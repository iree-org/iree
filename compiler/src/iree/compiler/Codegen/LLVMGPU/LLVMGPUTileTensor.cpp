// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <numeric>

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgExt/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Dialect/LoweringConfig.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using mlir::iree_compiler::IREE::LinalgExt::TilingPatterns;

#define DEBUG_TYPE "iree-llvmgpu-tile-tensor"

namespace mlir {
namespace iree_compiler {

/// Patterns for workgroup level tiling. Workgroup tiling is done at the flow
/// level but we may have extra tiling for the reduction dimension. Therefore we
/// tile again without distributing.
static void populateTilingReductionPatterns(RewritePatternSet &patterns) {
  auto tileSizesFn = [&](OpBuilder &builder,
                         Operation *op) -> SmallVector<Value, 4> {
    auto interfaceOp = cast<PartitionableLoopsInterface>(*op);
    auto partitionedLoops =
        interfaceOp.getPartitionableLoops(kNumMaxParallelDims);
    SmallVector<Value, 4> tileSizes = getTileSizes(builder, op, 0);
    auto zero = builder.create<arith::ConstantIndexOp>(op->getLoc(), 0);
    for (unsigned depth : partitionedLoops) {
      if (depth < tileSizes.size()) {
        tileSizes[depth] = zero;
      }
    }
    return tileSizes;
  };

  auto tilingOptions = linalg::LinalgTilingOptions()
                           .setLoopType(linalg::LinalgTilingLoopType::Loops)
                           .setTileSizeComputationFunction(tileSizesFn);
  MLIRContext *context = patterns.getContext();

  linalg::LinalgTransformationFilter filter(
      ArrayRef<StringAttr>{
          StringAttr::get(context, getWorkgroupMemoryMarker())},
      StringAttr::get(context, getWorkgroupKTiledMarker()));
  filter.setMatchByDefault();
  TilingPatterns<linalg::MatmulOp, linalg::BatchMatmulOp, linalg::GenericOp,
                 linalg::Conv2DNhwcHwcfOp>::insert(patterns, tilingOptions,
                                                   filter);
}

LogicalResult tileReduction(func::FuncOp funcOp) {
  {
    // Tile again at the workgroup level since redution dimension were
    // ignored. Dimensions already tiled will be ignore since we tile to the
    // same size.
    RewritePatternSet wgTilingPatterns(funcOp.getContext());
    populateTilingReductionPatterns(wgTilingPatterns);
    if (failed(applyPatternsAndFoldGreedily(funcOp,
                                            std::move(wgTilingPatterns)))) {
      return failure();
    }
  }

  {
    RewritePatternSet wgTilingCanonicalizationPatterns =
        linalg::getLinalgTilingCanonicalizationPatterns(funcOp.getContext());
    populateAffineMinSCFCanonicalizationPattern(
        wgTilingCanonicalizationPatterns);
    scf::populateSCFForLoopCanonicalizationPatterns(
        wgTilingCanonicalizationPatterns);
    if (failed(applyPatternsAndFoldGreedily(
            funcOp, std::move(wgTilingCanonicalizationPatterns)))) {
      return failure();
    }
    return success();
  }
}

/// Tile parallel dimensions according to the attribute tile sizes attached to
/// each op.
static LogicalResult tileParallelDims(func::FuncOp funcOp,
                                      SmallVectorImpl<int64_t> &workgroupSize,
                                      bool distributeToWarp) {
  std::array<int64_t, 3> elementPerWorkgroup = {
      distributeToWarp ? workgroupSize[0] / kWarpSize : workgroupSize[0],
      workgroupSize[1], workgroupSize[2]};
  SmallVector<TilingInterface> computeOps;
  funcOp.walk([&](TilingInterface op) { computeOps.push_back(op); });

  for (TilingInterface tilingOp : computeOps) {
    size_t numLoops = 0;
    for (auto type : tilingOp.getLoopIteratorTypes()) {
      if (type == getParallelIteratorTypeName()) numLoops++;
    }
    IRRewriter rewriter(tilingOp->getContext());
    rewriter.setInsertionPoint(tilingOp);
    auto interfaceOp =
        cast<PartitionableLoopsInterface>(*tilingOp.getOperation());
    auto partitionedLoops =
        interfaceOp.getPartitionableLoops(kNumMaxParallelDims);
    // If there are no dimensions to tile skip the transformation.
    if (partitionedLoops.empty()) continue;
    SmallVector<OpFoldResult> numThreads(numLoops, rewriter.getIndexAttr(0));
    int64_t id = 0;
    int64_t threadId = 0;
    SmallVector<int64_t> idDims;
    for (unsigned loop : llvm::reverse(partitionedLoops)) {
      int64_t num = elementPerWorkgroup[id++];
      if (num > 1) {
        numThreads[loop] = rewriter.getIndexAttr(num);
        idDims.push_back(threadId++);
      }
    }
    std::reverse(idDims.begin(), idDims.end());
    for (int64_t i = threadId; i < 3; i++) idDims.push_back(i);

    auto tilingResult =
        linalg::tileToForeachThreadOp(rewriter, tilingOp, numThreads, idDims);
    rewriter.replaceOp(tilingOp, tilingResult->tileOp->getResults());
  }
  return success();
}

namespace {
struct LLVMGPUTileTensorPass
    : public LLVMGPUTileTensorBase<LLVMGPUTileTensorPass> {
 private:
  // Distribute the workloads to warp if true otherwise distribute to threads.
  bool distributeToWarp = false;

 public:
  LLVMGPUTileTensorPass(bool distributeToWarp)
      : distributeToWarp(distributeToWarp) {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, gpu::GPUDialect>();
  }
  void runOnOperation() override {
    auto funcOp = getOperation();
    if (!isEntryPoint(funcOp)) return;

    if (failed(tileReduction(funcOp))) {
      return signalPassFailure();
    }

    LLVM_DEBUG({
      llvm::dbgs() << "--- After tile reductions:";
      funcOp.dump();
    });

    auto workgroupSize = llvm::to_vector<4>(llvm::map_range(
        getEntryPoint(funcOp)->getWorkgroupSize().value(),
        [&](Attribute attr) { return attr.cast<IntegerAttr>().getInt(); }));
    if (failed(tileParallelDims(funcOp, workgroupSize, distributeToWarp))) {
      return signalPassFailure();
    }

    LLVM_DEBUG({
      llvm::dbgs() << "--- After second level of tiling";
      funcOp.dump();
    });
  }
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createLLVMGPUTileTensor(
    bool distributeToWarp) {
  return std::make_unique<LLVMGPUTileTensorPass>(distributeToWarp);
}

}  // namespace iree_compiler
}  // namespace mlir

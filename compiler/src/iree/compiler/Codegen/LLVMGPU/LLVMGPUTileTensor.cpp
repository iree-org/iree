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
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using mlir::iree_compiler::IREE::LinalgExt::TilingPatterns;

#define DEBUG_TYPE "iree-llvmgpu-tile-tensor"

namespace mlir {
namespace iree_compiler {

/// Patterns for workgroup level tiling. Workgroup tiling is done at the flow
/// level but we may have extra tiling for the reduction dimension. Therefore we
/// tile again without distributing.
static void populateTilingPatterns(RewritePatternSet &patterns,
                                   bool onlyReduction) {
  auto tileSizesFn = [onlyReduction](OpBuilder &builder,
                                     Operation *op) -> SmallVector<Value, 4> {
    auto interfaceOp = cast<PartitionableLoopsInterface>(*op);
    auto partitionedLoops =
        interfaceOp.getPartitionableLoops(kNumMaxParallelDims);
    SmallVector<Value, 4> tileSizes = getTileSizes(builder, op, 0);
    if (onlyReduction) {
      auto zero = builder.create<arith::ConstantIndexOp>(op->getLoc(), 0);
      for (unsigned depth : partitionedLoops) {
        if (depth < tileSizes.size()) {
          tileSizes[depth] = zero;
        }
      }
    }
    return tileSizes;
  };

  auto tilingOptions = linalg::LinalgTilingOptions()
                           .setLoopType(linalg::LinalgTilingLoopType::Loops)
                           .setTileSizeComputationFunction(tileSizesFn);
  MLIRContext *context = patterns.getContext();

  IREE::LinalgExt::LinalgTransformationFilter filter(
      ArrayRef<StringAttr>{
          StringAttr::get(context, getWorkgroupMemoryMarker())},
      StringAttr::get(context, getWorkgroupKTiledMarker()));
  filter.setMatchByDefault();
  TilingPatterns<linalg::MatmulOp, linalg::BatchMatmulOp, linalg::GenericOp,
                 linalg::Conv2DNhwcHwcfOp,
                 linalg::Conv2DNchwFchwOp>::insert(patterns, tilingOptions,
                                                   filter);
}

LogicalResult tileToSerialLoops(func::FuncOp funcOp, bool onlyReduction) {
  {
    // Tile again at the workgroup level since redution dimension were
    // ignored. Dimensions already tiled will be ignore since we tile to the
    // same size.
    RewritePatternSet wgTilingPatterns(funcOp.getContext());
    populateTilingPatterns(wgTilingPatterns, onlyReduction);
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
      if (type == utils::IteratorType::parallel) numLoops++;
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
    int64_t id = 0, threadDim = 0;
    SmallVector<Attribute> idDims;
    auto getThreadMapping = [&](int64_t dim) {
      return mlir::gpu::GPUThreadMappingAttr::get(
          tilingOp->getContext(), dim == 0   ? mlir::gpu::Threads::DimX
                                  : dim == 1 ? mlir::gpu::Threads::DimY
                                             : mlir::gpu::Threads::DimZ);
    };
    for (unsigned loop : llvm::reverse(partitionedLoops)) {
      int64_t num = elementPerWorkgroup[id++];
      if (num > 1) {
        numThreads[loop] = rewriter.getIndexAttr(num);
        idDims.push_back(getThreadMapping(threadDim++));
      }
    }
    std::reverse(idDims.begin(), idDims.end());
    ArrayAttr mapping = rewriter.getArrayAttr(idDims);
    auto tilingResult =
        linalg::tileToForallOp(rewriter, tilingOp, numThreads, mapping);
    rewriter.replaceOp(tilingOp, tilingResult->tileOp->getResults());
  }
  return success();
}

// Tile convolution output window dimension by 1 to prepare downsizing.
static LogicalResult tileAndUnrollConv(func::FuncOp funcOp) {
  SmallVector<linalg::ConvolutionOpInterface, 1> convOps;
  funcOp.walk([&convOps](linalg::ConvolutionOpInterface convOp) {
    convOps.push_back(convOp);
  });
  for (linalg::ConvolutionOpInterface convOp : convOps) {
    auto consumerOp = cast<linalg::LinalgOp>(*convOp);
    OpBuilder builder(funcOp.getContext());
    SmallVector<int64_t> tileSizes = getTileSizes(consumerOp, 1);
    if (tileSizes.empty()) return success();
    auto identityLoopOrder =
        llvm::to_vector<4>(llvm::seq<int64_t>(0, tileSizes.size()));

    FailureOr<linalg::TileLoopNest> loopNest =
        IREE::LinalgExt::tileConsumerAndFuseProducers(
            builder, consumerOp, tileSizes, identityLoopOrder, std::nullopt);
    if (failed(loopNest)) {
      consumerOp.emitOpError("failed tiling and fusing producers");
      return failure();
    }

    consumerOp->replaceAllUsesWith(loopNest->getRootOpReplacementResults());

    // Fully unroll the generated loop. This allows us to remove the loop
    // for parallel output window dimension, so it helps future vector
    // transformations.
    if (!loopNest->getLoopOps().empty()) {
      assert(loopNest->getLoopOps().size() == 1);
      scf::ForOp loopOp = loopNest->getLoopOps().front();
      IntegerAttr ub;
      if (!matchPattern(loopOp.getUpperBound(), m_Constant(&ub))) {
        loopOp.emitOpError("upper bound should be a constant");
        return failure();
      }
      if (failed(mlir::loopUnrollByFactor(loopOp, ub.getInt()))) {
        loopOp.emitOpError("failed unrolling by factor 1");
        return failure();
      }
    }
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
    registry.insert<AffineDialect, gpu::GPUDialect, scf::SCFDialect>();
  }
  void runOnOperation() override {
    auto funcOp = getOperation();
    if (!isEntryPoint(funcOp)) return;

    funcOp->walk([&](linalg::LinalgOp op) {
      op->removeAttr(IREE::LinalgExt::LinalgTransforms::kLinalgTransformMarker);
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

    // Tile to serial loops to the wg tile size to handle reductions and other
    // dimension that have not been distributed.
    if (failed(tileToSerialLoops(funcOp, /*onlyReduction=*/true))) {
      return signalPassFailure();
    }

    LLVM_DEBUG({
      llvm::dbgs() << "--- After tile reductions:";
      funcOp.dump();
    });

    if (failed(tileAndUnrollConv(funcOp))) {
      return signalPassFailure();
    }

    LLVM_DEBUG({
      llvm::dbgs() << "--- After conv unrolling:";
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

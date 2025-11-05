// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- SPIRVTileAndDistribute.cpp -----------------------------------------===//
//
// This pass tiles and distributes Linalg ops with buffer semantics to
// invocations.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Codegen/SPIRV/Passes.h"
#include "iree/compiler/Codegen/SPIRV/Utils.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/JSON.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-spirv-tile-and-distribute"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_SPIRVTILEANDDISTRIBUTEPASS
#include "iree/compiler/Codegen/SPIRV/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Invocation tiling utils
//===----------------------------------------------------------------------===//

/// Returns the workgroup sizes from the export op. Returns empty list if it can
/// not be found.
static SmallVector<int64_t>
getWorkgroupSizes(mlir::FunctionOpInterface funcOp) {
  std::optional<IREE::HAL::ExecutableExportOp> exportOp = getEntryPoint(funcOp);
  if (!exportOp) {
    return {};
  }
  std::optional<ArrayAttr> workgroupSizeAttr = exportOp->getWorkgroupSize();
  if (!workgroupSizeAttr) {
    return {};
  }
  return llvm::map_to_vector(workgroupSizeAttr.value(), [](Attribute attr) {
    return cast<IntegerAttr>(attr).getInt();
  });
}

static llvm::SmallBitVector
computeSkipTileLoops(TilingInterface op, ArrayRef<int64_t> configTileSizes,
                     ArrayRef<int64_t> workgroupSize) {
  size_t numLoops = op.getLoopIteratorTypes().size();
  llvm::SmallBitVector notTiledLoops(numLoops, false);
  for (size_t index = 0; index < numLoops; ++index) {
    if (index >= workgroupSize.size()) {
      continue;
    }
    int64_t tileSizeVal =
        index < configTileSizes.size() ? configTileSizes[index] : 0;
    if (tileSizeVal == 0)
      continue;
    // `workgroupSize` and `tileSizeVal` are in opposite order.
    if (tileSizeVal < workgroupSize[workgroupSize.size() - index - 1])
      notTiledLoops.set(index);
  }
  return notTiledLoops;
}

/// Tiles LinalgOp to target invocations.
static LogicalResult
tileToInvocation(mlir::FunctionOpInterface funcOp,
                 ArrayRef<int64_t> configTileSizes,
                 const linalg::TileSizeComputationFunction &computeFn) {
  auto getThreadProcInfoFn = [](OpBuilder &builder, Location loc,
                                ArrayRef<Range> parallelLoopRanges) {
    return getGPUProcessorIdsAndCounts<gpu::ThreadIdOp, gpu::BlockDimOp>(
        builder, loc, parallelLoopRanges.size());
  };
  linalg::LinalgLoopDistributionOptions distributionOptions;
  distributionOptions.procInfo = getThreadProcInfoFn;

  auto tilingOptions = linalg::LinalgTilingOptions()
                           .setLoopType(linalg::LinalgTilingLoopType::Loops)
                           .setTileSizeComputationFunction(computeFn)
                           .setDistributionOptions(distributionOptions);

  MLIRContext *context = funcOp.getContext();
  IRRewriter rewriter(context);
  auto marker = StringAttr::get(context, getTileReductionMarker());
  auto filter = LinalgTransformationFilter(ArrayRef<StringAttr>(), marker);

  SmallVector<TilingInterface> candidates;
  funcOp.walk([&](TilingInterface op) { candidates.push_back(op); });
  SmallVector<int64_t> workgroupSize = getWorkgroupSizes(funcOp);
  for (auto op : candidates) {
    llvm::SmallBitVector skipTileLoops;
    if (tilingOptions.tileSizeComputationFunction && !workgroupSize.empty()) {
      skipTileLoops = computeSkipTileLoops(op, configTileSizes, workgroupSize);
    }

    FailureOr<IREETilingResult> res =
        tileDispatchUsingSCFForOp(rewriter, op, tilingOptions, skipTileLoops);
    if (failed(res)) {
      return failure();
    }
    for (auto tiledOp : res->tiledOps) {
      filter.replaceLinalgTransformationFilter(rewriter, tiledOp);
    }
  }

  return success();
}

//====---------------------------------------------------------------------===//
// Reduction tiling utils
//====---------------------------------------------------------------------===//

static LogicalResult
tileReduction(mlir::FunctionOpInterface funcOp,
              const scf::SCFTileSizeComputationFunction &computeFn) {
  MLIRContext *context = funcOp.getContext();
  IRRewriter rewriter(context);
  auto filter = LinalgTransformationFilter(
      StringAttr::get(context, getTileReductionMarker()), std::nullopt);
  auto options =
      scf::SCFTilingOptions().setTileSizeComputationFunction(computeFn);
  return tileLinalgOpsWithFilter(funcOp, options, filter);
}

//===----------------------------------------------------------------------===//
// Main pass
//===----------------------------------------------------------------------===//

namespace {
/// Function pass that implements tiling and distributing Linalg ops with
/// buffer semantics.
class SPIRVTileAndDistributePass final
    : public impl::SPIRVTileAndDistributePassBase<SPIRVTileAndDistributePass> {
public:
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, gpu::GPUDialect,
                    linalg::LinalgDialect, memref::MemRefDialect,
                    scf::SCFDialect, vector::VectorDialect>();
  }

  void runOnOperation() override;
};
} // namespace

//====---------------------------------------------------------------------===//
// Main pass implementation
//====---------------------------------------------------------------------===//

void SPIRVTileAndDistributePass::runOnOperation() {
  MLIRContext *context = &getContext();
  mlir::FunctionOpInterface funcOp = getOperation();
  if (!isEntryPoint(funcOp))
    return;

  auto threadTileSizes = getSPIRVTileSize(funcOp, 1);
  if (failed(threadTileSizes))
    return signalPassFailure();
  auto threadTileComputeFn = getSPIRVTileSizeComputeFn(funcOp, 1);
  if (failed(threadTileComputeFn))
    return signalPassFailure();
  auto reductionTileComputeFn = getSPIRVScfTileSizeComputeFn(funcOp, 2);
  if (failed(reductionTileComputeFn))
    return signalPassFailure();

  { // Tile and distribute to invocations.
    if (failed(
            tileToInvocation(funcOp, *threadTileSizes, *threadTileComputeFn))) {
      funcOp.emitOpError() << "failed to tile to invocations";
      return signalPassFailure();
    }

    LLVM_DEBUG({
      llvm::dbgs() << "--- After tiling to invocations ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });
  }

  {
    RewritePatternSet canonicalizationPatterns =
        linalg::getLinalgTilingCanonicalizationPatterns(context);

    SmallVector<int64_t> numWorkgroups = getStaticNumWorkgroups(funcOp);
    populateFoldAffineMinInDistributedLoopsPatterns(canonicalizationPatterns,
                                                    numWorkgroups);

    if (failed(applyPatternsGreedily(funcOp,
                                     std::move(canonicalizationPatterns)))) {
      // TODO(#4759): This does not converge after the max number of iterations.
      // It indicates that some pattern upstream is generating ops even when the
      // pattern failed to match. Not related to correctness, but would be good
      // to figure out and fix.
      // return signalPassFailure();
    }

    LLVM_DEBUG({
      llvm::dbgs() << "--- After tiling canonicalization ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });
  }

  { // Tile reduction dimensions.
    if (failed(tileReduction(funcOp, *reductionTileComputeFn))) {
      funcOp.emitOpError() << "failing in tile reduction";
      return signalPassFailure();
    }

    RewritePatternSet canonicalizationPatterns =
        linalg::getLinalgTilingCanonicalizationPatterns(context);
    scf::populateSCFForLoopCanonicalizationPatterns(canonicalizationPatterns);
    if (failed(applyPatternsGreedily(funcOp,
                                     std::move(canonicalizationPatterns)))) {
      funcOp.emitOpError() << "failing canonicalizing after tile reduction";
      return signalPassFailure();
    }

    LLVM_DEBUG({
      llvm::dbgs() << "--- After tiling reduction dimensions  ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });
  }
}

} // namespace mlir::iree_compiler

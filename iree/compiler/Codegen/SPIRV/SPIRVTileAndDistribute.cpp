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

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgExt/Passes/Transforms.h"
#include "iree/compiler/Codegen/Dialect/LoweringConfig.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/SPIRV/Utils.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SCF/Transforms.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-spirv-tile-and-distribute"

namespace mlir {
namespace iree_compiler {

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

/// Returns a Linalg marker that matches any of the `matchMarkers` and replaces
/// it with `replaceMarker`.
static linalg::LinalgTransformationFilter getLinalgMatchAndReplaceMarker(
    ArrayRef<StringRef> matchMarkers, Optional<StringRef> replaceMarker,
    MLIRContext *context) {
  SmallVector<StringAttr, 2> matchIds;
  matchIds.reserve(matchMarkers.size());
  for (StringRef marker : matchMarkers) {
    matchIds.emplace_back(StringAttr::get(context, marker));
  }

  Optional<StringAttr> replaceId;
  if (replaceMarker) replaceId = StringAttr::get(context, *replaceMarker);

  return linalg::LinalgTransformationFilter(matchIds, replaceId);
}

//===----------------------------------------------------------------------===//
// Invocation tiling patterns
//===----------------------------------------------------------------------===//

/// Patterns for third level tiling to target invocations.
static void populateTilingToInvocationPatterns(MLIRContext *context,
                                               RewritePatternSet &patterns) {
  linalg::TileSizeComputationFunction getInnerTileSizeFn =
      [&](OpBuilder &builder, Operation *op) {
        return getTileSizes(builder, op, 1);
      };

  auto getThreadProcInfoFn = [](OpBuilder &builder, Location loc,
                                ArrayRef<Range> parallelLoopRanges) {
    return getGPUProcessorIdsAndCounts<gpu::ThreadIdOp, gpu::BlockDimOp>(
        builder, loc, parallelLoopRanges.size());
  };
  linalg::LinalgLoopDistributionOptions invocationDistributionOptions;
  invocationDistributionOptions.procInfo = getThreadProcInfoFn;
  invocationDistributionOptions.distributionMethod = {
      {linalg::DistributionMethod::Cyclic, linalg::DistributionMethod::Cyclic,
       linalg::DistributionMethod::Cyclic}};

  auto tilingOptions =
      linalg::LinalgTilingOptions()
          .setLoopType(linalg::LinalgTilingLoopType::Loops)
          .setTileSizeComputationFunction(getInnerTileSizeFn)
          .setDistributionOptions(invocationDistributionOptions);

  SmallVector<StringRef, 2> matchMarkers = {getWorkgroupMemoryMarker()};

  linalg::LinalgTransformationFilter filterVectorized =
      getLinalgMatchAndReplaceMarker(matchMarkers, getVectorizeMarker(),
                                     context)
          .setMatchByDefault();
  linalg::TilingPatterns<linalg::Conv1DNwcWcfOp, linalg::Conv3DNdhwcDhwcfOp,
                         linalg::DepthwiseConv2DNhwcHwcmOp, linalg::FillOp,
                         linalg::GenericOp, linalg::PoolingNhwcMaxOp,
                         linalg::PoolingNhwcMinOp,
                         linalg::PoolingNhwcSumOp>::insert(patterns,
                                                           tilingOptions,
                                                           filterVectorized);

  linalg::LinalgTransformationFilter filterTiled =
      getLinalgMatchAndReplaceMarker(matchMarkers, getTileReductionMarker(),
                                     context)
          .setMatchByDefault();
  linalg::TilingPatterns<linalg::BatchMatmulOp, linalg::Conv2DNhwcHwcfOp,
                         linalg::DepthwiseConv2DNhwcHwcOp,
                         linalg::MatmulOp>::insert(patterns, tilingOptions,
                                                   filterTiled);

  patterns.insert<IREE::LinalgExt::TiledOpInterfaceTilingPattern>(
      context, tilingOptions,
      getLinalgMatchAndReplaceMarker(matchMarkers, getVectorizeMarker(),
                                     context)
          .setMatchByDefault());
}

//====---------------------------------------------------------------------===//
// Reduction tiling patterns
//====---------------------------------------------------------------------===//

static void populateTilingReductionPatterns(
    MLIRContext *context, RewritePatternSet &patterns,
    linalg::LinalgTransformationFilter marker) {
  auto getTileSizeFn = [&](OpBuilder &builder, Operation *op) {
    return getTileSizes(builder, op, 2);
  };

  auto tilingOptions = linalg::LinalgTilingOptions()
                           .setLoopType(linalg::LinalgTilingLoopType::Loops)
                           .setTileSizeComputationFunction(getTileSizeFn);

  linalg::TilingPatterns<linalg::BatchMatmulOp, linalg::Conv2DNhwcHwcfOp,
                         linalg::DepthwiseConv2DNhwcHwcOp,
                         linalg::MatmulOp>::insert(patterns, tilingOptions,
                                                   marker);
}

//===----------------------------------------------------------------------===//
// Main pass
//===----------------------------------------------------------------------===//

namespace {
/// Function pass that implements tiling and distributing Linalg ops with
/// buffer semantics.
class SPIRVTileAndDistributePass
    : public SPIRVTileAndDistributeBase<SPIRVTileAndDistributePass> {
 public:
  SPIRVTileAndDistributePass() = default;
  SPIRVTileAndDistributePass(const SPIRVTileAndDistributePass &pass) = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, gpu::GPUDialect, linalg::LinalgDialect,
                    memref::MemRefDialect, scf::SCFDialect,
                    vector::VectorDialect>();
  }

  void runOnOperation() override;
};
}  // namespace

//====---------------------------------------------------------------------===//
// Main pass implementation
//====---------------------------------------------------------------------===//

void SPIRVTileAndDistributePass::runOnOperation() {
  MLIRContext *context = &getContext();
  FuncOp funcOp = getOperation();
  auto entryPointOp = getEntryPoint(funcOp);
  if (!entryPointOp) return;

  {  // Tile and distribute to invocations.
    RewritePatternSet invocationTilingPatterns(&getContext());
    populateTilingToInvocationPatterns(context, invocationTilingPatterns);
    if (failed(applyPatternsAndFoldGreedily(
            funcOp, std::move(invocationTilingPatterns)))) {
      funcOp.emitOpError() << "failure in tiling";
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

    populateFoldAffineMinInDistributedLoopsPatterns(canonicalizationPatterns);

    if (failed(applyPatternsAndFoldGreedily(
            funcOp, std::move(canonicalizationPatterns)))) {
      // TODO(#4759): Terrifyingly, this fails. Errors here were ignored for a
      // long time and now tests for this pass actually fail if we propagate the
      // failure correctly. Fix this.
      // funcOp.emitOpError() << "failure canonicalizing after tiling";
      // return signalPassFailure();
    }

    LLVM_DEBUG({
      llvm::dbgs() << "--- After tiling canonicalization ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });
  }

  {  // Tile reduction dimensions.
    RewritePatternSet reductionTilingPatterns(&getContext());
    auto marker = getLinalgMatchAndReplaceMarker(getTileReductionMarker(),
                                                 getVectorizeMarker(), context);
    populateTilingReductionPatterns(context, reductionTilingPatterns, marker);
    if (failed(applyPatternsAndFoldGreedily(
            funcOp, std::move(reductionTilingPatterns)))) {
      funcOp.emitOpError() << "failing in tile reduction";
      return signalPassFailure();
    }

    RewritePatternSet canonicalizationPatterns =
        linalg::getLinalgTilingCanonicalizationPatterns(context);
    scf::populateSCFForLoopCanonicalizationPatterns(canonicalizationPatterns);
    if (failed(applyPatternsAndFoldGreedily(
            funcOp, std::move(canonicalizationPatterns)))) {
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

//===----------------------------------------------------------------------===//
// Pass entry point and registration
//===----------------------------------------------------------------------===//

std::unique_ptr<OperationPass<FuncOp>> createSPIRVTileAndDistributePass() {
  return std::make_unique<SPIRVTileAndDistributePass>();
}

}  // namespace iree_compiler
}  // namespace mlir

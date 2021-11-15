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
#include "iree-dialects/Dialect/LinalgExt/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Dialect/LoweringConfig.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/SPIRV/Utils.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SCF/Transforms.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorTransforms.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopUtils.h"

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
  SmallVector<Identifier, 2> matchIds;
  matchIds.reserve(matchMarkers.size());
  for (StringRef marker : matchMarkers) {
    matchIds.emplace_back(Identifier::get(marker, context));
  }

  Optional<Identifier> replaceId;
  if (replaceMarker) replaceId = Identifier::get(*replaceMarker, context);

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

  patterns.insert<linalg::LinalgTilingPattern<linalg::CopyOp>,
                  linalg::LinalgTilingPattern<linalg::Conv1DNwcWcfOp>,
                  linalg::LinalgTilingPattern<linalg::Conv3DNdhwcDhwcfOp>,
                  linalg::LinalgTilingPattern<linalg::DepthwiseConv2DNhwcOp>,
                  linalg::LinalgTilingPattern<linalg::FillOp>,
                  linalg::LinalgTilingPattern<linalg::GenericOp>,
                  linalg::LinalgTilingPattern<linalg::PoolingNhwcMaxOp>,
                  linalg::LinalgTilingPattern<linalg::PoolingNhwcMinOp>,
                  linalg::LinalgTilingPattern<linalg::PoolingNhwcSumOp>>(
      context, tilingOptions,
      getLinalgMatchAndReplaceMarker(matchMarkers, getVectorizeMarker(),
                                     context)
          .setMatchByDefault());

  patterns.insert<linalg::LinalgTilingPattern<linalg::BatchMatmulOp>,
                  linalg::LinalgTilingPattern<linalg::Conv2DNhwcHwcfOp>,
                  linalg::LinalgTilingPattern<linalg::DepthwiseConv2DNhwOp>,
                  linalg::LinalgTilingPattern<linalg::MatmulOp>>(
      context, tilingOptions,
      getLinalgMatchAndReplaceMarker(matchMarkers, getTileReductionMarker(),
                                     context)
          .setMatchByDefault());

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

  patterns.insert<linalg::LinalgTilingPattern<linalg::BatchMatmulOp>,
                  linalg::LinalgTilingPattern<linalg::Conv2DNhwcHwcfOp>,
                  linalg::LinalgTilingPattern<linalg::DepthwiseConv2DNhwOp>,
                  linalg::LinalgTilingPattern<linalg::MatmulOp>>(
      context, tilingOptions, marker);
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
    (void)applyPatternsAndFoldGreedily(funcOp,
                                       std::move(invocationTilingPatterns));

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

    (void)applyPatternsAndFoldGreedily(funcOp,
                                       std::move(canonicalizationPatterns));

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
    (void)applyPatternsAndFoldGreedily(funcOp,
                                       std::move(reductionTilingPatterns));

    RewritePatternSet canonicalizationPatterns =
        linalg::getLinalgTilingCanonicalizationPatterns(context);
    scf::populateSCFForLoopCanonicalizationPatterns(canonicalizationPatterns);
    (void)applyPatternsAndFoldGreedily(funcOp,
                                       std::move(canonicalizationPatterns));

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

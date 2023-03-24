// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/Transforms/Transforms.h"

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgExt/Passes/PassDetail.h"
#include "iree-dialects/Dialect/LinalgExt/Passes/Passes.h"
#include "iree-dialects/Dialect/LinalgExt/Utils/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace LinalgExt {

//===----------------------------------------------------------------------===//
// CodegenStrategy patterns and passes.
//===----------------------------------------------------------------------===//

FailureOr<linalg::TileLoopNest> tileConsumerAndFuseProducers(
    OpBuilder &b, linalg::LinalgOp consumerOp, ArrayRef<int64_t> tileSizes,
    ArrayRef<int64_t> tileInterchange,
    const std::optional<linalg::LinalgLoopDistributionOptions>
        &tileDistribution) {
  assert(tileSizes.size() == tileInterchange.size() &&
         "expect the number of tile sizes and interchange dims to match");
  assert(isPermutationVector(tileInterchange) &&
         "expect tile interchange is a permutation");

  // Create an empty tile loop nest.
  linalg::TileLoopNest tileLoopNest(consumerOp);

  // Search the number of outer parallel loops to separate them from possible
  // inner reduction dimensions.
  auto iterTypes = consumerOp.getIteratorTypesArray();
  // Make sure to only look at the leading loops for tiling---we will scan this
  // array to find the first non-parallel loop later and use that for indexing
  // into the tile sizes.
  if (iterTypes.size() > tileSizes.size()) {
    iterTypes.resize(tileSizes.size());
  }
  applyPermutationToVector(iterTypes, tileInterchange);
  auto *it = find_if_not(iterTypes, linalg::isParallelIterator);
  int64_t split = std::distance(iterTypes.begin(), it);

  // Helper to fuse the producers greedily using a queue of fusion candidates.
  auto fuseProducersGreedily = [&](ArrayRef<OpOperand *> operands) {
    SmallVector<OpOperand *> candidates(operands.begin(), operands.end());
    while (!candidates.empty()) {
      FailureOr<linalg::LinalgOp> fusedProducer =
          tileLoopNest.fuseProducer(b, candidates.pop_back_val());
      if (failed(fusedProducer))
        continue;
      candidates.append(fusedProducer->getDpsInputOperands());
      candidates.append(fusedProducer->getDpsInitOperands());
    }
  };

  // Perform tiling and fusion in two steps. We need to respect the loop
  // interchange here; filter parellel dimensions based on their order *after*
  // permutation but pass in the original configuration *before* permuation,
  // given the tiling and interchange happen together.
  SmallVector<int64_t> outerTileSizes(tileSizes.size(), 0);
  SmallVector<int64_t> innerTileSizes(tileSizes.size(), 0);
  for (int64_t i : tileInterchange.take_front(split))
    outerTileSizes[i] = tileSizes[i];
  for (int64_t i : tileInterchange.drop_front(split))
    innerTileSizes[i] = tileSizes[i];

  // Tile the outer parallel loops and fuse the output operands.
  if (failed(tileLoopNest.tileRootOp(b, outerTileSizes, tileInterchange,
                                     tileDistribution)))
    return failure();
  fuseProducersGreedily(tileLoopNest.getRootOp().getDpsInitOperands());

  // Tile the remaining loops and fuse the input operands.
  if (failed(tileLoopNest.tileRootOp(b, innerTileSizes, tileInterchange,
                                     tileDistribution)))
    return failure();
  fuseProducersGreedily(tileLoopNest.getRootOp().getDpsInputOperands());

  // Exit if the tile loop nest is empty since all tile sizes are zero.
  if (tileLoopNest.isEmpty())
    return failure();

  return tileLoopNest;
}

/// Peel loops after tiling.
static void peelTiledLinalgOp(RewriterBase &rewriter,
                              linalg::TiledLinalgOp &res,
                              ArrayRef<int64_t> peeledLoops,
                              linalg::LinalgTilingLoopType loopType) {
  for (int64_t loop : peeledLoops) {
    assert(loop < static_cast<int64_t>(res.loops.size()) &&
           "requested peeling of non-existing loop");
    SmallVector<Value, 4> loopResults;
    Operation *loopOp = res.loops[loop];
    loopResults = linalg::peelLoop(rewriter, loopOp);

    // The result of the loop nest may change with peeling.
    if (res.tensorResults.size() == loopOp->getNumResults() &&
        std::equal(res.tensorResults.begin(), res.tensorResults.end(),
                   loopOp->getResults().begin()))
      res.tensorResults = loopResults;
  }
}

/// Linalg tiling pattern.
LinalgTilingPattern::LinalgTilingPattern(
    MLIRContext *context, linalg::LinalgTilingOptions options,
    LinalgExt::LinalgTransformationFilter f, PatternBenefit benefit)
    : OpInterfaceRewritePattern<linalg::LinalgOp>(context, benefit),
      filter(std::move(f)), options(std::move(options)) {}

LinalgTilingPattern::LinalgTilingPattern(
    StringRef opName, MLIRContext *context, linalg::LinalgTilingOptions options,
    LinalgExt::LinalgTransformationFilter f, PatternBenefit benefit)
    : OpInterfaceRewritePattern<linalg::LinalgOp>(context, benefit),
      filter(f.addOpNameFilter(opName)), options(std::move(options)) {}

FailureOr<linalg::TiledLinalgOp>
LinalgTilingPattern::returningMatchAndRewrite(linalg::LinalgOp op,
                                              PatternRewriter &rewriter) const {
  if (failed(filter.checkAndNotify(rewriter, op)))
    return failure();

  FailureOr<linalg::TiledLinalgOp> res =
      linalg::tileLinalgOp(rewriter, op, options);
  if (failed(res))
    return failure();

  // Clear filter to stop recursive pattern application.
  // This must be done here to properly propagate to peeling branches.
  filter.replaceLinalgTransformationFilter(rewriter, res->op);

  // Peel the loops of the TiledLinalgOp.
  peelTiledLinalgOp(rewriter, *res, options.peeledLoops, options.loopType);

  if (res->tensorResults.empty())
    rewriter.eraseOp(op);
  else
    rewriter.replaceOp(op, res->tensorResults);

  return res;
}

/// Linalg SCF tiling pattern.
SCFTilingPattern::SCFTilingPattern(MLIRContext *context,
                                   scf::SCFTilingOptions options,
                                   LinalgExt::LinalgTransformationFilter f,
                                   PatternBenefit benefit)
    : OpInterfaceRewritePattern<TilingInterface>(context, benefit),
      filter(std::move(f)), options(std::move(options)) {}

SCFTilingPattern::SCFTilingPattern(StringRef opName, MLIRContext *context,
                                   scf::SCFTilingOptions options,
                                   LinalgExt::LinalgTransformationFilter f,
                                   PatternBenefit benefit)
    : OpInterfaceRewritePattern<TilingInterface>(context, benefit),
      filter(f.addOpNameFilter(opName)), options(std::move(options)) {}

LogicalResult
SCFTilingPattern::returningMatchAndRewrite(TilingInterface op,
                                           PatternRewriter &rewriter) const {
  if (failed(filter.checkAndNotify(rewriter, op)))
    return failure();

  FailureOr<scf::SCFTilingResult> tiledResults =
      scf::tileUsingSCFForOp(rewriter, op, options);
  if (failed(tiledResults))
    return failure();

  rewriter.replaceOp(op, tiledResults->replacements);

  for (auto tiledOp : tiledResults->tiledOps) {
    filter.replaceLinalgTransformationFilter(rewriter, tiledOp);
  }

  return success();
}

LinalgVectorizationPattern::LinalgVectorizationPattern(
    MLIRContext *context, LinalgVectorizationOptions opts,
    LinalgExt::LinalgTransformationFilter f, PatternBenefit benefit)
    : OpInterfaceRewritePattern<linalg::LinalgOp>(context, benefit),
      options(std::move(opts)), filter(std::move(f)) {}

LinalgVectorizationPattern::LinalgVectorizationPattern(
    StringRef opName, MLIRContext *context, LinalgVectorizationOptions opts,
    LinalgExt::LinalgTransformationFilter f, PatternBenefit benefit)
    : OpInterfaceRewritePattern<linalg::LinalgOp>(context, benefit),
      options(std::move(opts)), filter(f.addOpNameFilter(opName)) {}

LogicalResult
LinalgVectorizationPattern::matchAndRewrite(linalg::LinalgOp linalgOp,
                                            PatternRewriter &rewriter) const {
  if (failed(filter.checkAndNotify(rewriter, linalgOp)))
    return failure();
  SmallVector<int64_t> vectorSizes;
  if (options.enableVectorMasking)
    vectorSizes.append(options.vectorSizeComputationFunction(
        linalgOp, options.canonicalVectorSizes));
  return vectorize(rewriter, linalgOp, vectorSizes);
}

namespace {
/// Configurable pass to apply pattern-based linalg tiling.
struct LinalgStrategyTilePass
    : public LinalgStrategyTilePassBase<LinalgStrategyTilePass> {

  LinalgStrategyTilePass() = default;

  LinalgStrategyTilePass(StringRef opName, scf::SCFTilingOptions options,
                         LinalgExt::LinalgTransformationFilter filt)
      : options(std::move(options)), filter(std::move(filt)) {
    this->anchorOpName.setValue(opName.str());
  }

  void runOnOperation() override {
    auto funcOp = getOperation();
    if (!anchorFuncName.empty() && funcOp.getName() != anchorFuncName)
      return;

    MLIRContext *ctx = funcOp.getContext();
    RewritePatternSet tilingPattern(ctx);
    if (!anchorOpName.empty())
      tilingPattern.add<SCFTilingPattern>(anchorOpName, ctx, options, filter);
    else
      tilingPattern.add<SCFTilingPattern>(ctx, options, filter);

    (void)applyPatternsAndFoldGreedily(funcOp, std::move(tilingPattern));
  }

  scf::SCFTilingOptions options;
  LinalgExt::LinalgTransformationFilter filter;
};

/// Configurable pass to apply lowering of coarser-grained named linalg ops into
/// finer-grained named versions.
struct LinalgStrategyDecomposePass
    : public LinalgStrategyDecomposePassBase<LinalgStrategyDecomposePass> {

  LinalgStrategyDecomposePass() = default;

  LinalgStrategyDecomposePass(LinalgExt::LinalgTransformationFilter filter)
      : filter(std::move(filter)) {}

  void runOnOperation() override {
    auto funcOp = getOperation();
    if (!anchorFuncName.empty() && funcOp.getName() != anchorFuncName)
      return;
    RewritePatternSet decompositionPattern(funcOp.getContext());
    linalg::populateDecomposeConvolutionPatterns(decompositionPattern);
    if (failed(applyPatternsAndFoldGreedily(funcOp,
                                            std::move(decompositionPattern))))
      signalPassFailure();
  }

  LinalgExt::LinalgTransformationFilter filter;
};

/// Configurable pass to apply pattern-based linalg vectorization.
struct LinalgStrategyVectorizePass
    : public LinalgStrategyVectorizePassBase<LinalgStrategyVectorizePass> {

  LinalgStrategyVectorizePass() = default;

  LinalgStrategyVectorizePass(StringRef opName, LinalgVectorizationOptions opts,
                              LinalgExt::LinalgTransformationFilter filt)
      : options(std::move(opts)), filter(std::move(filt)) {
    this->vectorizePadding = opts.vectorizePadding;
  };

  void runOnOperation() override {
    auto funcOp = getOperation();
    if (!anchorFuncName.empty() && funcOp.getName() != anchorFuncName)
      return;

    RewritePatternSet vectorizationPatterns(funcOp.getContext());
    if (!anchorOpName.empty()) {
      vectorizationPatterns.add<LinalgVectorizationPattern>(
          anchorOpName, funcOp.getContext(), options, filter);
    } else {
      vectorizationPatterns.add<LinalgVectorizationPattern>(funcOp.getContext(),
                                                            options, filter);
    }

    // TODO: Move this down the pipeline once we have the ODM-based masking
    // representation.
    vector::populateVectorMaskLoweringPatternsForSideEffectingOps(
        vectorizationPatterns);

    vector::populateVectorTransferPermutationMapLoweringPatterns(
        vectorizationPatterns);
    vector::populateVectorReductionToContractPatterns(vectorizationPatterns);
    vectorizationPatterns.add<linalg::LinalgCopyVTRForwardingPattern,
                              linalg::LinalgCopyVTWForwardingPattern>(
        funcOp.getContext(), /*benefit=*/2);
    vector::TransferReadOp::getCanonicalizationPatterns(vectorizationPatterns,
                                                        funcOp.getContext());
    vector::TransferWriteOp::getCanonicalizationPatterns(vectorizationPatterns,
                                                         funcOp.getContext());
    (void)applyPatternsAndFoldGreedily(funcOp,
                                       std::move(vectorizationPatterns));

    // Apply the pad tensor op vectorization separately to avoid running the
    // GenericPadOpVectorizationPattern too early.
    // TODO: Improve once we have better infrastructure to control pattern
    // application.
    if (vectorizePadding) {
      RewritePatternSet patterns(funcOp.getContext());
      linalg::populatePadOpVectorizationPatterns(patterns);
      (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
    }
  }

  LinalgVectorizationOptions options;
  LinalgExt::LinalgTransformationFilter filter;
};

/// Configurable pass to enable the application of other pattern-based linalg
/// passes.
struct LinalgStrategyEnablePass
    : public LinalgStrategyEnablePassBase<LinalgStrategyEnablePass> {

  LinalgStrategyEnablePass(LinalgEnablingOptions opt,
                           LinalgExt::LinalgTransformationFilter filt)
      : options(opt), filter(std::move(filt)) {}

  void runOnOperation() override {
    auto funcOp = getOperation();
    if (!anchorFuncName.empty() && funcOp.getName() != anchorFuncName)
      return;

    MLIRContext *context = funcOp.getContext();
    RewritePatternSet patterns =
        linalg::getLinalgTilingCanonicalizationPatterns(context);
    scf::populateSCFForLoopCanonicalizationPatterns(patterns);
    tensor::populateFoldTensorEmptyPatterns(patterns);
    memref::populateResolveRankedShapeTypeResultDimsPatterns(patterns);
    // Pull in tensor dialect canonicalization patterns to fold tensor.cast
    // into producers when possible.
    context->getLoadedDialect<tensor::TensorDialect>()
        ->getCanonicalizationPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns))))
      return signalPassFailure();

    if (options.licm) {
      funcOp->walk([&](LoopLikeOpInterface loopLike) {
        moveLoopInvariantCode(loopLike);
      });
    }

    // Gathers all innermost loops through a post order pruned walk.
    funcOp.walk([](Operation *op) {
      if (auto forOp = dyn_cast<AffineForOp>(op))
        (void)promoteIfSingleIteration(forOp);
      else if (auto forOp = dyn_cast<scf::ForOp>(op))
        (void)promoteIfSingleIteration(forOp);
    });
    if (options.hoistRedundantVectorTransfers)
      linalg::hoistRedundantVectorTransfers(funcOp);

    if (options.hoistRedundantVectorTransfersOnTensor)
      linalg::hoistRedundantVectorTransfersOnTensor(funcOp);

    // Run CSE to cleanup after canonicalization.
    OpPassManager dynamicPM("func.func");
    dynamicPM.addPass(createCSEPass());
    if (failed(runPipeline(dynamicPM, funcOp)))
      return signalPassFailure();
  }

  LinalgEnablingOptions options;
  LinalgExt::LinalgTransformationFilter filter;
};

/// Configurable pass to lower vector operations.
struct LinalgStrategyLowerVectorsPass
    : public LinalgStrategyLowerVectorsPassBase<
          LinalgStrategyLowerVectorsPass> {

  LinalgStrategyLowerVectorsPass(LinalgVectorLoweringOptions opt,
                                 LinalgExt::LinalgTransformationFilter filt)
      : options(opt), filter(std::move(filt)) {}

  void runOnOperation() override {
    auto funcOp = getOperation();
    if (!anchorFuncName.empty() && funcOp.getName() != anchorFuncName)
      return;

    MLIRContext *context = funcOp.getContext();
    RewritePatternSet patterns(context);
    vector::populateVectorToVectorCanonicalizationPatterns(patterns);
    // In a progressive lowering of vectors, this would be the 1st step.
    if (options.contractionLowering) {
      patterns.add<vector::ContractionOpToOuterProductOpLowering,
                   vector::ContractionOpToMatmulOpLowering,
                   vector::ContractionOpLowering>(
          options.vectorTransformOptions, context);
      vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);
    }
    // In a progressive lowering of vectors, this would be the 2nd step.
    if (options.multiReductionLowering) {
      vector::populateVectorMultiReductionLoweringPatterns(
          patterns,
          options.vectorTransformOptions.vectorMultiReductionLowering);
    }
    // In a progressive lowering of vectors, this would be the 3rd step.
    if (options.transferPartialRewrite) {
      patterns.add<vector::VectorTransferFullPartialRewriter>(
          context, options.vectorTransformOptions);
    }
    // In a progressive lowering of vectors, this would be the 4th step.
    if (options.transferLowering) {
      vector::populateVectorTransferLoweringPatterns(patterns,
                                                     options.maxTransferRank);
    }
    // In a progressive lowering of vectors, this would be the 5th step.
    if (options.transferToSCFConversion) {
      populateVectorToSCFConversionPatterns(
          patterns, options.vectorTransferToSCFOptions.setTargetRank(
                        options.maxTransferRank));
    }
    // In a progressive lowering of vectors, this would be the 6th step.
    if (options.shapeCastLowering) {
      vector::populateVectorShapeCastLoweringPatterns(patterns);
    }
    // In a progressive lowering of vectors, this would be the 7th step.
    if (options.transposeLowering) {
      vector::populateVectorTransposeLoweringPatterns(
          patterns, options.vectorTransformOptions);
      if (options.avx2Lowering)
        x86vector::avx2::populateSpecializedTransposeLoweringPatterns(
            patterns, options.avx2LoweringOptions, /*benefit=*/10);
    }
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
  }

  LinalgVectorLoweringOptions options;
  LinalgExt::LinalgTransformationFilter filter;
};

/// Configurable pass to lower vector operations.
struct LinalgStrategyRemoveMarkersPass
    : public LinalgStrategyRemoveMarkersPassBase<
          LinalgStrategyRemoveMarkersPass> {

  void runOnOperation() override {
    auto funcOp = getOperation();
    if (!anchorFuncName.empty() && funcOp.getName() != anchorFuncName)
      return;
    funcOp.walk([](linalg::LinalgOp op) {
      op->removeAttr(LinalgTransforms::kLinalgTransformMarker);
    });
  }
};
} // namespace

/// Create a LinalgStrategyTilePass.
std::unique_ptr<OperationPass<func::FuncOp>> createLinalgStrategyTilePass(
    StringRef opName, const scf::SCFTilingOptions &options,
    const LinalgExt::LinalgTransformationFilter &filter) {
  return std::make_unique<LinalgStrategyTilePass>(opName, options, filter);
}

/// Create a LinalgStrategyDecomposePass.
// TODO: if/when we need finer control add an `opName` parameter.
std::unique_ptr<OperationPass<func::FuncOp>> createLinalgStrategyDecomposePass(
    const LinalgExt::LinalgTransformationFilter &filter) {
  return std::make_unique<LinalgStrategyDecomposePass>(filter);
}

/// Create a LinalgStrategyVectorizePass.
std::unique_ptr<OperationPass<func::FuncOp>> createLinalgStrategyVectorizePass(
    StringRef opName, const LinalgVectorizationOptions &options,
    const LinalgExt::LinalgTransformationFilter &filter) {
  return std::make_unique<LinalgStrategyVectorizePass>(opName, options, filter);
}

/// Create a LinalgStrategyEnablePass.
std::unique_ptr<OperationPass<func::FuncOp>> createLinalgStrategyEnablePass(
    LinalgEnablingOptions opt,
    const LinalgExt::LinalgTransformationFilter &filter) {
  return std::make_unique<LinalgStrategyEnablePass>(opt, filter);
}

/// Create a LinalgStrategyLowerVectorsPass.
std::unique_ptr<OperationPass<func::FuncOp>>
createLinalgStrategyLowerVectorsPass(
    LinalgVectorLoweringOptions opt,
    const LinalgExt::LinalgTransformationFilter &filter) {
  return std::make_unique<LinalgStrategyLowerVectorsPass>(opt, filter);
}

/// Create a LinalgStrategyRemoveMarkersPass.
std::unique_ptr<OperationPass<func::FuncOp>>
createLinalgStrategyRemoveMarkersPass() {
  return std::make_unique<LinalgStrategyRemoveMarkersPass>();
}

} // namespace LinalgExt
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir

// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/Transforms/Transforms.h"

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgExt/Passes/PassDetail.h"
#include "iree-dialects/Dialect/LinalgExt/Passes/Passes.h"
#include "iree-dialects/Dialect/LinalgExt/Passes/Transforms.h"
#include "iree-dialects/Dialect/LinalgExt/Utils/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
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
    const Optional<linalg::LinalgLoopDistributionOptions> &tileDistribution) {
  assert(tileSizes.size() == tileInterchange.size() &&
         "expect the number of tile sizes and interchange dims to match");
  assert(linalg::isPermutation(tileInterchange) &&
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

namespace {
///
/// Linalg tile and fuse tensor ops pattern.
///
/// Apply tiling and fusion as a pattern.
/// See `tileConsumerAndFuseProducers` for more details.
struct LinalgTileAndFuseTensorOpsBasePattern : public RewritePattern {
  // Entry point to match any LinalgOp.
  LinalgTileAndFuseTensorOpsBasePattern(
      MLIRContext *context, linalg::LinalgTilingAndFusionOptions options,
      PatternBenefit benefit = 1)
      : RewritePattern(MatchAnyOpTypeTag(), benefit, context),
        options(std::move(options)) {}
  // Entry point to match a specific LinalgOp.
  LinalgTileAndFuseTensorOpsBasePattern(
      StringRef opName, MLIRContext *context,
      linalg::LinalgTilingAndFusionOptions options, PatternBenefit benefit = 1)
      : RewritePattern(opName, benefit, context), options(std::move(options)) {}

  /// `matchAndRewrite` implementation that returns the significant transformed
  /// pieces of IR.
  FailureOr<linalg::TileLoopNest>
  returningMatchAndRewrite(Operation *op, PatternRewriter &rewriter) const;

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    return returningMatchAndRewrite(op, rewriter);
  }

private:
  /// Tile sizes and interchange used to tile the root operation.
  linalg::LinalgTilingAndFusionOptions options;
};
} // namespace

FailureOr<mlir::linalg::TileLoopNest>
LinalgTileAndFuseTensorOpsBasePattern::returningMatchAndRewrite(
    Operation *op, PatternRewriter &rewriter) const {
  linalg::LinalgOp rootOp = dyn_cast<linalg::LinalgOp>(op);
  if (!rootOp)
    return failure();

  // Check `tileSizes` contains a tile size for every `rootOp` loop dimension.
  if (options.tileSizes.size() < rootOp.getNumLoops())
    return rewriter.notifyMatchFailure(op, "expect #tile sizes >= #loops");

  // Check `tileInterchange` contains no entries or as many as `tileSizes`.
  if (!options.tileInterchange.empty() &&
      options.tileInterchange.size() != options.tileSizes.size())
    return rewriter.notifyMatchFailure(
        op, "expect the number of tile sizes and interchange dims to match");

  // Copy the `tileSizes` and `tileInterchange` prefixes needed for `rootOp`.
  SmallVector<int64_t> rootTileSizes(options.tileSizes.begin(),
                                     options.tileSizes.begin() +
                                         rootOp.getNumLoops());
  SmallVector<int64_t> rootInterchange =
      options.tileInterchange.empty()
          ? llvm::to_vector<6>(llvm::seq<int64_t>(0, rootOp.getNumLoops()))
          : SmallVector<int64_t>(options.tileInterchange.begin(),
                                 options.tileInterchange.begin() +
                                     rootOp.getNumLoops());

  // Check `rootTileSizes` contains non-zero tile sizes.
  if (llvm::count(rootTileSizes, 0) == static_cast<long>(rootTileSizes.size()))
    return rewriter.notifyMatchFailure(
        op, "expect at least one non-zero tile size");

  // Check `rootInterchange` is a permutation of the `rootOp` loop dimensions.
  // It has to be a permutation since the tiling cannot tile the same loop
  // dimension multiple times.
  if (!linalg::isPermutation(rootInterchange))
    return rewriter.notifyMatchFailure(
        op, "expect the tile interchange permutes the root loops");

  // Tile `rootOp` and fuse its producers.
  FailureOr<linalg::TileLoopNest> tileLoopNest =
      IREE::LinalgExt::tileConsumerAndFuseProducers(
          rewriter, rootOp, rootTileSizes, rootInterchange,
          options.tileDistribution);
  if (failed(tileLoopNest))
    return rewriter.notifyMatchFailure(
        op, "tileConsumerAndFuseProducers failed unexpectedly");

  // Replace all uses of the tiled loop operation.
  rootOp->replaceAllUsesWith(tileLoopNest->getRootOpReplacementResults());

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

LinalgVectorizationPattern::LinalgVectorizationPattern(
    MLIRContext *context, LinalgExt::LinalgTransformationFilter f,
    PatternBenefit benefit)
    : OpInterfaceRewritePattern<linalg::LinalgOp>(context, benefit),
      filter(std::move(f)) {}

LinalgVectorizationPattern::LinalgVectorizationPattern(
    StringRef opName, MLIRContext *context,
    LinalgExt::LinalgTransformationFilter f, PatternBenefit benefit)
    : OpInterfaceRewritePattern<linalg::LinalgOp>(context, benefit),
      filter(f.addOpNameFilter(opName)) {}

LogicalResult
LinalgVectorizationPattern::matchAndRewrite(linalg::LinalgOp linalgOp,
                                            PatternRewriter &rewriter) const {
  if (failed(filter.checkAndNotify(rewriter, linalgOp)))
    return failure();
  return vectorize(rewriter, linalgOp);
}

namespace {

///
/// Linalg peeling patterns.
///

/// Compute the loops to peel and return them in a SmallVector. Loops will be
/// peeled in order of appearance in the SmallVector. This order will impact the
/// output IR. If an inner-to-outer order is provided, the peeled iterations of
/// the outer loops will also contain the peeled inner loops. If an
/// outer-to-inner order is provided, the peeled iterations of the outer loops
/// will not contain any peeled inner loops.

/// `filter` controls LinalgTransformMarker matching and update when specified.
struct LinalgPeelingPattern
    : public OpInterfaceRewritePattern<linalg::LinalgOp> {
  /// Construct a generic pattern applied to all LinalgOp that verify `filter`.
  LinalgPeelingPattern(MLIRContext *context,
                       LinalgExt::LinalgTransformationFilter f =
                           LinalgExt::LinalgTransformationFilter(),
                       LinalgPeelOptions options = LinalgPeelOptions(),
                       PatternBenefit benefit = 1);

  /// Construct a pattern specifically applied to `opName`.
  LinalgPeelingPattern(StringRef opName, MLIRContext *context,
                       LinalgPeelOptions options = LinalgPeelOptions(),
                       LinalgExt::LinalgTransformationFilter f =
                           LinalgExt::LinalgTransformationFilter(),
                       PatternBenefit benefit = 1);

  LogicalResult matchAndRewrite(linalg::LinalgOp linalgOp,
                                PatternRewriter &rewriter) const override;

private:
  /// LinalgTransformMarker handles special attribute manipulations.
  const LinalgExt::LinalgTransformationFilter filter;
  /// Peeling options.
  const LinalgPeelOptions options;
};

LinalgPeelingPattern::LinalgPeelingPattern(
    MLIRContext *context, LinalgExt::LinalgTransformationFilter f,
    LinalgPeelOptions options, PatternBenefit benefit)
    : OpInterfaceRewritePattern<linalg::LinalgOp>(context, benefit),
      filter(std::move(f)), options(std::move(options)) {}

LinalgPeelingPattern::LinalgPeelingPattern(
    StringRef opName, MLIRContext *context, LinalgPeelOptions options,
    LinalgExt::LinalgTransformationFilter f, PatternBenefit benefit)
    : OpInterfaceRewritePattern<linalg::LinalgOp>(context, benefit),
      filter(f.addOpNameFilter(opName)), options(std::move(options)) {}

LogicalResult
LinalgPeelingPattern::matchAndRewrite(linalg::LinalgOp linalgOp,
                                      PatternRewriter &rewriter) const {
  if (failed(filter.checkAndNotify(rewriter, linalgOp)))
    return failure();

  // Increase marker counter even if peeling doesn't happen for this op.
  filter.replaceLinalgTransformationFilter(rewriter, linalgOp);

  if (!options.loopsToPeelComputationFunction)
    return failure();

  SmallVector<scf::ForOp, 4> loopsToPeel;
  options.loopsToPeelComputationFunction(rewriter, linalgOp, loopsToPeel);
  linalg::peelLoops(rewriter, loopsToPeel);
  return success();
}

namespace {
///
/// Linalg tile and fuse tensor ops pattern.
///
/// Apply tiling and fusion as a pattern.
/// `filter` controls LinalgTransformMarker matching and update when specified.
/// See `tileConsumerAndFuseProducers` for more details.
struct LinalgTileAndFuseTensorOpsPattern : public RewritePattern {
  // Entry point to match any LinalgOp.
  LinalgTileAndFuseTensorOpsPattern(
      MLIRContext *context, linalg::LinalgTilingAndFusionOptions options,
      LinalgExt::LinalgTransformationFilter f =
          LinalgExt::LinalgTransformationFilter(),
      PatternBenefit benefit = 1)
      : RewritePattern(MatchAnyOpTypeTag(), benefit, context),
        filter(std::move(f)), options(std::move(options)) {}
  // Entry point to match a specific LinalgOp.
  LinalgTileAndFuseTensorOpsPattern(
      StringRef opName, MLIRContext *context,
      linalg::LinalgTilingAndFusionOptions options,
      LinalgExt::LinalgTransformationFilter f =
          LinalgExt::LinalgTransformationFilter(),
      PatternBenefit benefit = 1)
      : RewritePattern(opName, benefit, context), filter(std::move(f)),
        options(std::move(options)) {}

  /// `matchAndRewrite` implementation that returns the significant transformed
  /// pieces of IR.
  FailureOr<linalg::TileLoopNest>
  returningMatchAndRewrite(Operation *op, PatternRewriter &rewriter) const {
    if (failed(filter.checkAndNotify(rewriter, op)))
      return failure();
    LinalgTileAndFuseTensorOpsBasePattern p(op->getContext(), options);
    auto maybeTileLoopNest = p.returningMatchAndRewrite(op, rewriter);
    if (failed(maybeTileLoopNest))
      return failure();
    // Apply the filter if specified.
    for (linalg::LinalgOp linalgOp :
         maybeTileLoopNest->getAllTiledAndFusedOps())
      filter.replaceLinalgTransformationFilter(rewriter, linalgOp);
    return maybeTileLoopNest;
  }

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    return returningMatchAndRewrite(op, rewriter);
  }

private:
  /// LinalgTransformMarker handles special attribute manipulations.
  LinalgExt::LinalgTransformationFilter filter;
  /// Tile sizes and interchange used to tile the root operation.
  linalg::LinalgTilingAndFusionOptions options;
};
} // namespace
  //
/// Configurable pass to apply pattern-based tiling and fusion.
struct LinalgStrategyTileAndFusePass
    : public LinalgStrategyTileAndFusePassBase<LinalgStrategyTileAndFusePass> {

  LinalgStrategyTileAndFusePass() = default;

  LinalgStrategyTileAndFusePass(StringRef opName,
                                linalg::LinalgTilingAndFusionOptions opt,
                                LinalgExt::LinalgTransformationFilter filt)
      : options(std::move(opt)), filter(std::move(filt)) {
    this->anchorOpName.setValue(opName.str());
  }

  void runOnOperation() override {
    auto funcOp = getOperation();
    if (!anchorFuncName.empty() && funcOp.getName() != anchorFuncName)
      return;

    RewritePatternSet tilingAndFusionPattern(funcOp.getContext());
    if (!anchorOpName.empty()) {
      tilingAndFusionPattern.add<LinalgTileAndFuseTensorOpsPattern>(
          anchorOpName, funcOp.getContext(), options, filter);
    } else {
      tilingAndFusionPattern.add<LinalgTileAndFuseTensorOpsPattern>(
          funcOp.getContext(), options, filter);
    }
    // Search the root operation using bottom up traversal.
    GreedyRewriteConfig config;
    config.useTopDownTraversal = false;
    (void)applyPatternsAndFoldGreedily(
        funcOp, std::move(tilingAndFusionPattern), config);
  }

  linalg::LinalgTilingAndFusionOptions options;
  LinalgExt::LinalgTransformationFilter filter;
};

/// Configurable pass to apply pattern-based linalg tiling.
struct LinalgStrategyTilePass
    : public LinalgStrategyTilePassBase<LinalgStrategyTilePass> {

  LinalgStrategyTilePass() = default;

  LinalgStrategyTilePass(StringRef opName, linalg::LinalgTilingOptions opt,
                         LinalgExt::LinalgTransformationFilter filt)
      : options(std::move(opt)), filter(std::move(filt)) {
    this->anchorOpName.setValue(opName.str());
  }

  void runOnOperation() override {
    auto funcOp = getOperation();
    if (!anchorFuncName.empty() && funcOp.getName() != anchorFuncName)
      return;

    MLIRContext *ctx = funcOp.getContext();
    RewritePatternSet tilingPattern(ctx);
    if (!anchorOpName.empty())
      tilingPattern.add<LinalgTilingPattern>(anchorOpName, ctx, options,
                                             filter);
    else
      tilingPattern.add<LinalgTilingPattern>(ctx, options, filter);
    if (anchorOpName == tensor::PadOp::getOperationName())
      populatePadTensorTilingPatterns(tilingPattern, options);
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(tilingPattern));
  }

  linalg::LinalgTilingOptions options;
  LinalgExt::LinalgTransformationFilter filter;
};

/// Configurable pass to apply hoisting and padding.
struct LinalgStrategyPadPass
    : public LinalgStrategyPadPassBase<LinalgStrategyPadPass> {

  LinalgStrategyPadPass() = default;

  LinalgStrategyPadPass(StringRef opName, linalg::LinalgPaddingOptions opt,
                        LinalgExt::LinalgTransformationFilter filt)
      : options(std::move(opt)), filter(std::move(filt)) {
    this->anchorOpName.setValue(opName.str());
  }

  void runOnOperation() override {
    auto funcOp = getOperation();
    if (!anchorFuncName.empty() && funcOp.getName() != anchorFuncName)
      return;

    RewritePatternSet paddingPattern(funcOp.getContext());
    if (!anchorOpName.empty()) {
      paddingPattern.add<LinalgPaddingPattern>(
          anchorOpName, funcOp.getContext(), options, filter);
    } else {
      paddingPattern.add<LinalgPaddingPattern>(funcOp.getContext(), options,
                                               filter);
    }
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(paddingPattern));
  }

  linalg::LinalgPaddingOptions options;
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
    decompositionPattern
        .add<DownscaleSizeOneWindowed2DConvolution<linalg::Conv2DNhwcHwcfOp,
                                                   linalg::Conv1DNwcWcfOp>,
             DownscaleSizeOneWindowed2DConvolution<linalg::Conv2DNchwFchwOp,
                                                   linalg::Conv1DNcwFcwOp>,
             DownscaleDepthwiseConv2DNhwcHwcOp>(funcOp.getContext(), filter);
    if (failed(applyPatternsAndFoldGreedily(funcOp,
                                            std::move(decompositionPattern))))
      signalPassFailure();
  }

  LinalgExt::LinalgTransformationFilter filter;
};

/// Configurable pass to apply pattern-based linalg peeling.
struct LinalgStrategyPeelPass
    : public LinalgStrategyPeelPassBase<LinalgStrategyPeelPass> {

  LinalgStrategyPeelPass() = default;

  LinalgStrategyPeelPass(StringRef opName, LinalgPeelOptions opt,
                         LinalgExt::LinalgTransformationFilter filt)
      : options(std::move(opt)), filter(std::move(filt)) {
    this->anchorOpName.setValue(opName.str());
  }

  void runOnOperation() override {
    auto funcOp = getOperation();
    if (!anchorFuncName.empty() && funcOp.getName() != anchorFuncName)
      return;

    RewritePatternSet peelingPatterns(funcOp.getContext());
    if (!anchorOpName.empty()) {
      peelingPatterns.add<LinalgPeelingPattern>(
          anchorOpName, funcOp.getContext(), options, filter);
    } else {
      peelingPatterns.add<LinalgPeelingPattern>(funcOp.getContext(), filter,
                                                options);
    }
    if (failed(
            applyPatternsAndFoldGreedily(funcOp, std::move(peelingPatterns))))
      return signalPassFailure();
  }

  LinalgPeelOptions options;
  LinalgExt::LinalgTransformationFilter filter;
};

/// Configurable pass to apply pattern-based linalg vectorization.
struct LinalgStrategyVectorizePass
    : public LinalgStrategyVectorizePassBase<LinalgStrategyVectorizePass> {

  LinalgStrategyVectorizePass() = default;

  LinalgStrategyVectorizePass(StringRef opName,
                              LinalgExt::LinalgTransformationFilter filt,
                              bool padVectorize = false)
      : filter(std::move(filt)) {
    this->anchorOpName.setValue(opName.str());
    this->vectorizePadding.setValue(padVectorize);
  }

  void runOnOperation() override {
    auto funcOp = getOperation();
    if (!anchorFuncName.empty() && funcOp.getName() != anchorFuncName)
      return;

    RewritePatternSet vectorizationPatterns(funcOp.getContext());
    if (!anchorOpName.empty()) {
      vectorizationPatterns.add<LinalgVectorizationPattern>(
          anchorOpName, funcOp.getContext(), filter);
    } else {
      vectorizationPatterns.add<LinalgVectorizationPattern>(funcOp.getContext(),
                                                            filter);
    }
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

/// Create a LinalgStrategyTileAndFusePass.
std::unique_ptr<OperationPass<func::FuncOp>>
createLinalgStrategyTileAndFusePass(
    StringRef opName, const linalg::LinalgTilingAndFusionOptions &options,
    const LinalgExt::LinalgTransformationFilter &filter) {
  return std::make_unique<LinalgStrategyTileAndFusePass>(opName, options,
                                                         filter);
}

/// Create a LinalgStrategyTilePass.
std::unique_ptr<OperationPass<func::FuncOp>> createLinalgStrategyTilePass(
    StringRef opName, const linalg::LinalgTilingOptions &opt,
    const LinalgExt::LinalgTransformationFilter &filter) {
  return std::make_unique<LinalgStrategyTilePass>(opName, opt, filter);
}

/// Create a LinalgStrategyPadPass.
std::unique_ptr<OperationPass<func::FuncOp>> createLinalgStrategyPadPass(
    StringRef opName, const linalg::LinalgPaddingOptions &opt,
    const LinalgExt::LinalgTransformationFilter &filter) {
  return std::make_unique<LinalgStrategyPadPass>(opName, opt, filter);
}

/// Create a LinalgStrategyDecomposePass.
// TODO: if/when we need finer control add an `opName` parameter.
std::unique_ptr<OperationPass<func::FuncOp>> createLinalgStrategyDecomposePass(
    const LinalgExt::LinalgTransformationFilter &filter) {
  return std::make_unique<LinalgStrategyDecomposePass>(filter);
}

/// Create a LinalgStrategyPeelPass.
std::unique_ptr<OperationPass<func::FuncOp>> createLinalgStrategyPeelPass(
    StringRef opName, const LinalgPeelOptions &opt,
    const LinalgExt::LinalgTransformationFilter &filter) {
  return std::make_unique<LinalgStrategyPeelPass>(opName, opt, filter);
}

/// Create a LinalgStrategyVectorizePass.
std::unique_ptr<OperationPass<func::FuncOp>> createLinalgStrategyVectorizePass(
    StringRef opName, const LinalgExt::LinalgTransformationFilter &filter,
    bool padVectorize) {
  return std::make_unique<LinalgStrategyVectorizePass>(opName, filter,
                                                       padVectorize);
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

//===----------------------------------------------------------------------===//
// LinalgExt patterns and passes.
//===----------------------------------------------------------------------===//

namespace {

/// A simple pattern rewriter that implements no special logic.
class SimpleRewriter : public PatternRewriter {
public:
  SimpleRewriter(MLIRContext *context) : PatternRewriter(context) {}
};

/// Returns a tensor.pad op if padding value is set. Otherwise, returns the
/// input directly. The method assumes that the `packOp` has static shapes.
Value getInputOrPaddedInput(OpBuilder &builder, PackOp packOp) {
  Value input = packOp.getInput();
  if (!packOp.getPaddingValue()) {
    return input;
  }

  Location loc = packOp.getLoc();
  ShapedType inputType = packOp.getInputType();
  int64_t inputRank = inputType.getRank();
  assert(llvm::all_of(packOp.getOutputShape().take_front(inputRank),
                      [](int64_t val) { return val == 1; }));

  SmallVector<int64_t> paddedShape;
  DenseMap<int64_t, OpFoldResult> tileAndPosMapping =
      packOp.getDimAndTileMapping();
  for (int64_t dim = 0; dim < inputRank; ++dim) {
    int64_t size = inputType.getDimSize(dim);
    if (!tileAndPosMapping.count(dim)) {
      paddedShape.push_back(size);
      continue;
    }

    // The size is less than or equal to tileSize because outer dims are all 1s.
    Optional<int64_t> tileSize =
        getConstantIntValue(tileAndPosMapping.lookup(dim));
    assert(tileSize.hasValue() && "dynamic inner tile size is not supported");
    paddedShape.push_back(tileSize.value());
  }
  auto resultType =
      RankedTensorType::get(paddedShape, inputType.getElementType());
  return tensor::createPadHighOp(resultType, input, packOp.getPaddingValue(),
                                 /*nofold=*/false, loc, builder);
}

/// Rewrites iree_linalg_ext.pack to tensor.pad + rank-up linalg.generic
/// (transpose) ops.
struct GeneralizePackOpPattern : OpRewritePattern<PackOp> {
  using OpRewritePattern<PackOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(PackOp packOp,
                                PatternRewriter &rewriter) const final {
    if (!packOp.hasTensorSemantics()) {
      return rewriter.notifyMatchFailure(packOp, "require tensor semantics");
    }

    // The expand_shape op can be avoided if outer dimensions of result are all
    // 1s. This can be relaxed if needed. A tensor.expand_shape will be
    // generated in that case.
    int64_t inputRank = packOp.getInputRank();
    if (llvm::any_of(packOp.getOutputShape().take_front(inputRank),
                     [](int64_t val) { return val != 1; })) {
      return rewriter.notifyMatchFailure(
          packOp, "require the outer dimension of the result are all 1s");
    }

    Value input = getInputOrPaddedInput(rewriter, packOp);

    SmallVector<AffineExpr> inputExprs;
    for (int64_t dim = 0; dim < inputRank; ++dim) {
      inputExprs.push_back(rewriter.getAffineDimExpr(dim));
    }
    // The dimensions map in the order of output dimensions. Since the
    // interchange is applied, we have to undo it for input.
    if (auto outerDims = packOp.getOuterDimsPerm()) {
      inputExprs = undoInterchange<AffineExpr>(
          inputExprs, extractFromI64ArrayAttr(outerDims));
    }
    for (auto en :
         llvm::enumerate(extractFromI64ArrayAttr(packOp.getInnerDimsPos()))) {
      inputExprs[en.value()] =
          rewriter.getAffineDimExpr(inputRank + en.index());
    }

    Location loc = packOp.getLoc();
    auto inputType = input.getType().cast<RankedTensorType>();
    auto nloops = packOp.getOutputRank();

    Value empty = rewriter.create<tensor::EmptyOp>(loc, packOp.getOutputShape(),
                                                   inputType.getElementType());
    SmallVector<utils::IteratorType, 4> loopAttributeTypes(
        nloops, utils::IteratorType::parallel);
    SmallVector<AffineMap, 2> indexingMaps = {
        AffineMap::get(nloops, 0, inputExprs, rewriter.getContext()),
        AffineMap::getMultiDimIdentityMap(nloops, rewriter.getContext())};

    auto transposedOp = rewriter.create<linalg::GenericOp>(
        loc, empty.getType(), input, empty, indexingMaps, loopAttributeTypes,
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          nestedBuilder.create<linalg::YieldOp>(nestedLoc, args[0]);
        });
    rewriter.replaceOp(packOp, transposedOp.getResult(0));
    return success();
  }
};

/// Rewrites iree_linalg_ext.unpack to rank-reduced extract_slice op + transpose
/// op + insert_slice op. It requires the outer dims are all 1s.
struct GeneralizeUnPackOpPattern : OpRewritePattern<UnPackOp> {
  using OpRewritePattern<UnPackOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(UnPackOp unpackOp,
                                PatternRewriter &rewriter) const final {
    if (!unpackOp.hasTensorSemantics()) {
      return rewriter.notifyMatchFailure(unpackOp, "require tensor semantics");
    }

    int64_t outputRank = unpackOp.getOutputRank();
    if (llvm::any_of(unpackOp.getInputShape().take_front(outputRank),
                     [](int64_t val) { return val != 1; })) {
      return rewriter.notifyMatchFailure(
          unpackOp, "require the outer dimension of the result are all 1s");
    }

    int64_t inputRank = unpackOp.getInputRank();

    Location loc = unpackOp.getLoc();
    Attribute zeroIdxAttr = rewriter.getIndexAttr(0);
    Attribute oneIdxAttr = rewriter.getIndexAttr(1);
    SmallVector<OpFoldResult> readOffsets(inputRank, zeroIdxAttr);
    SmallVector<OpFoldResult> readStrides(inputRank, oneIdxAttr);

    auto mixedTiles = unpackOp.getMixedTiles();
    SmallVector<OpFoldResult> readSizes(outputRank, oneIdxAttr);
    readSizes.append(mixedTiles.begin(), mixedTiles.end());

    // Explicitly create the type for extract_slice op because the inner tile
    // size could be 1. We want to represent the whole inner tile in this case.
    ArrayRef<int64_t> readShape =
        unpackOp.getInputShape().drop_front(outputRank);
    Type elemType = unpackOp.getInputType().getElementType();
    auto readType = RankedTensorType::get(readShape, elemType);
    Value innerTile = rewriter.create<tensor::ExtractSliceOp>(
        loc, readType, unpackOp.getInput(), readOffsets, readSizes,
        readStrides);

    SmallVector<int64_t> innerDimsPos =
        extractFromI64ArrayAttr(unpackOp.getInnerDimsPos());
    auto interchangeVector =
        computeInterchangeFromDimPos(innerDimsPos, outputRank);
    SmallVector<int64_t> transpShape =
        interchange<int64_t>(readShape, interchangeVector);

    Value empty = rewriter.create<tensor::EmptyOp>(loc, transpShape, elemType);
    auto transposedOp = rewriter.create<linalg::TransposeOp>(
        loc, innerTile, empty, interchangeVector);

    // Handle in-complete tiles.
    int numLoops = transpShape.size();
    SmallVector<OpFoldResult> tileStrides(numLoops, oneIdxAttr);
    SmallVector<OpFoldResult> tileOffsets(numLoops, zeroIdxAttr);
    SmallVector<OpFoldResult> tileSizes;
    for (int dim : innerDimsPos) {
      tileSizes.push_back(getDim(rewriter, loc, unpackOp.getOutput(), dim));
    }
    tileSizes = interchange<OpFoldResult>(tileSizes, interchangeVector);
    auto partialTile = rewriter.create<tensor::ExtractSliceOp>(
        loc, transposedOp.getResult()[0], tileOffsets, tileSizes, tileStrides);

    SmallVector<OpFoldResult> writeSizes;
    SmallVector<OpFoldResult> writeStrides(outputRank, oneIdxAttr);
    SmallVector<OpFoldResult> writeOffsets(outputRank, zeroIdxAttr);
    DenseMap<int64_t, OpFoldResult> dimAndTileMapping =
        unpackOp.getDimAndTileMapping();
    for (int i = 0, idx = 0; i < outputRank; ++i) {
      if (dimAndTileMapping.count(i)) {
        writeSizes.push_back(tileSizes[idx++]);
      } else {
        writeSizes.push_back(oneIdxAttr);
      }
    }
    auto insert = rewriter.create<tensor::InsertSliceOp>(
        loc, partialTile, unpackOp.getOutput(), writeOffsets, writeSizes,
        writeStrides);
    rewriter.replaceOp(unpackOp, insert.getResult());

    return success();
  }
};

struct LinalgExtVectorizationPass
    : public LinalgExtVectorizationBase<LinalgExtVectorizationPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, func::FuncDialect,
                    arith::ArithDialect, scf::SCFDialect, tensor::TensorDialect,
                    vector::VectorDialect>();
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    // Apply tiling to make outer dims be all 1s.
    {
      SimpleRewriter rewriter(ctx);
      auto packTilingOptions =
          scf::SCFTilingOptions().setTileSizeComputationFunction(
              [](OpBuilder &builder, Operation *op) {
                Location loc = op->getLoc();
                int inputRank = cast<PackOp>(op).getInputRank();
                SmallVector<Value> tileSizes(
                    inputRank, builder.create<arith::ConstantIndexOp>(loc, 1));
                return tileSizes;
              });
      auto funcOp = getOperation();
      funcOp->walk([&](LinalgExt::PackOp op) {
        FailureOr<scf::SCFTilingResult> tilingResult = scf::tileUsingSCFForOp(
            rewriter, cast<TilingInterface>(op.getOperation()),
            packTilingOptions);
        if (failed(tilingResult))
          return signalPassFailure();
        rewriter.replaceOp(op, tilingResult->replacements);
      });

      auto unpackTilingOptions =
          scf::SCFTilingOptions().setTileSizeComputationFunction(
              [](OpBuilder &builder, Operation *op) {
                Location loc = op->getLoc();
                auto unpackOp = cast<UnPackOp>(op);
                int numLoops = unpackOp.getOutputRank();
                auto dimAndTileMapping = unpackOp.getDimAndTileMapping();
                SmallVector<Value> tileSizes;
                for (int i = 0; i < numLoops; ++i) {
                  if (dimAndTileMapping.count(i)) {
                    tileSizes.push_back(getValueOrCreateConstantIndexOp(
                        builder, loc, dimAndTileMapping[i]));
                  } else {
                    tileSizes.push_back(
                        getDimValue(builder, loc, unpackOp.getOutput(), i));
                  }
                }
                return tileSizes;
              });
      funcOp->walk([&](LinalgExt::UnPackOp op) {
        FailureOr<scf::SCFTilingResult> tilingResult = scf::tileUsingSCFForOp(
            rewriter, cast<TilingInterface>(op.getOperation()),
            unpackTilingOptions);
        if (failed(tilingResult))
          return signalPassFailure();
        rewriter.replaceOp(op, tilingResult->replacements);
      });
    }

    // Generalize pack and unpack ops and canonicalize tiled ops.
    {
      RewritePatternSet patterns(ctx);
      linalg::populateLinalgTilingCanonicalizationPatterns(patterns);
      patterns.add<GeneralizePackOpPattern, GeneralizeUnPackOpPattern>(ctx);
      if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                              std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    // Kick in generic vectorizer.
    {
      RewritePatternSet patterns(ctx);
      patterns.add<LinalgVectorizationPattern>(ctx);
      linalg::populatePadOpVectorizationPatterns(patterns);
      vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);
      vector::TransferReadOp::getCanonicalizationPatterns(patterns, ctx);
      vector::TransferWriteOp::getCanonicalizationPatterns(patterns, ctx);
      if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                              std::move(patterns)))) {
        return signalPassFailure();
      }
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createLinalgExtVectorizationPass() {
  return std::make_unique<LinalgExtVectorizationPass>();
}

} // namespace LinalgExt
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir

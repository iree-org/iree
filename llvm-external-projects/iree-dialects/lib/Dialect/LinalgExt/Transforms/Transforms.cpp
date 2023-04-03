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
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
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
  return vectorize(rewriter, linalgOp, vectorSizes,
                   options.vectorizeGatherAccesses);
}

namespace {
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
      vector::populateVectorContractLoweringPatterns(
          patterns, options.vectorTransformOptions,
          /*benefit=*/1,
          /*disableOuterProductLowering=*/true);
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
      populateVectorTransferFullPartialPatterns(patterns,
                                                options.vectorTransformOptions);
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

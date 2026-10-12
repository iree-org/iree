// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- PropagateLinalgTranspose.cpp - Pass to propagate transposes ---------==//
//
// The pass is to propagate linalg.transpose operations through a restricted
// set of operations based on a set of local propagation decisions.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Dialect/Flow/Conversion/TensorToFlow/Utils.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Transforms.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "iree/compiler/GlobalOptimization/TransposePropagationPatterns.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-global-opt-propagate-linalg-transpose"

namespace mlir::iree_compiler::GlobalOptimization {

#define GEN_PASS_DEF_PROPAGATELINALGTRANSPOSEPASS
#include "iree/compiler/GlobalOptimization/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Propagation control
//===----------------------------------------------------------------------===//

static bool isReshapeBlockingFusion(Operation *producer, Operation *consumer) {
  auto isFusableOp = [](Operation *op) {
    if (!op) {
      return false;
    }
    return isa_and_nonnull<linalg::LinalgDialect,
                           IREE::LinalgExt::IREELinalgExtDialect,
                           tensor::TensorDialect>(op->getDialect());
  };
  return isFusableOp(producer) && isFusableOp(consumer);
}

// Restricts propagation to edges whose ops are both outside of pre-formed
// dispatches.
static bool isOutsideDispatch(OpOperand *operand) {
  return IREE::Flow::isNonNullAndOutsideDispatch(
      {operand->get().getDefiningOp(), operand->getOwner()});
}

// Profitability heuristic for sinking a transpose through a
// tensor.extract_slice: accepts the edge only when the slice, taken from the
// untransposed source instead, is still mappable to Flow.
static bool isUntransposedSliceMappableToFlow(OpOperand *operand) {
  if (!isOutsideDispatch(operand)) {
    return false;
  }
  auto transposeOp = cast<linalg::TransposeOp>(operand->get().getDefiningOp());
  auto extractOp = cast<tensor::ExtractSliceOp>(operand->getOwner());
  auto invPerm = invertPermutationVector(transposeOp.getPermutation());
  SmallVector<OpFoldResult> offsets = extractOp.getMixedOffsets();
  SmallVector<OpFoldResult> sizes = extractOp.getMixedSizes();
  SmallVector<OpFoldResult> strides = extractOp.getMixedStrides();
  ArrayRef<int64_t> srcShape = extractOp.getSourceType().getShape();

  // Permute the offsets, sizes, and strides to pre-transpose ordering.
  applyPermutationToVector(offsets, invPerm);
  applyPermutationToVector(sizes, invPerm);
  applyPermutationToVector(strides, invPerm);
  SmallVector<int64_t> baseShape = applyPermutation(srcShape, invPerm);

  // Check if the resulting offsets, sizes, and strides correspond to a
  // contiguous slice and can thus be mappable to a `flow.tensor.update` op.
  // This should always be worth doing because this can remove a dispatch for
  // the slice, and the transpose is on the slice rather than the full tensor.
  return IREE::Flow::isOffsetSizeAndStrideMappableToFlow(offsets, sizes,
                                                         strides, baseShape);
}

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

namespace {
struct PropagateLinalgTransposePass
    : impl::PropagateLinalgTransposePassBase<PropagateLinalgTransposePass> {
  using Base::Base;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, tensor::TensorDialect>();
  }
  explicit PropagateLinalgTransposePass(bool enableAggressivePropagation) {
    this->enableAggressivePropagation = enableAggressivePropagation;
  }

  void runOnOperation() override;
};
} // namespace

static void populateMatmulSinkingPatterns(RewritePatternSet &sinkingPatterns) {
  populateFoldTransposeIntoNamedMatmulPatterns(sinkingPatterns,
                                               isOutsideDispatch);
  populateFuseTransposeThroughGenericReductionPattern(sinkingPatterns,
                                                      isOutsideDispatch);
}

static void
populateCommonCanonicalizationPatterns(MLIRContext *context,
                                       RewritePatternSet &patterns) {
  linalg::FillOp::getCanonicalizationPatterns(patterns, context);
  tensor::EmptyOp::getCanonicalizationPatterns(patterns, context);
  tensor::ExpandShapeOp::getCanonicalizationPatterns(patterns, context);
  tensor::CollapseShapeOp::getCanonicalizationPatterns(patterns, context);
  memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
  tensor::populateFoldTensorEmptyPatterns(patterns,
                                          /*foldSingleUseOnly=*/false);
}

void PropagateLinalgTransposePass::runOnOperation() {
  MLIRContext *context = &getContext();
  mlir::FunctionOpInterface funcOp = getOperation();

  // Unless edge reshape propagation is enabled, only move transposes through
  // reshapes when the transpose blocks fusion of the reshape with its
  // neighbors.
  ControlTransposePropagationFn bubbleThroughCollapseShapeFn =
      [this](OpOperand *operand) {
        if (!isOutsideDispatch(operand)) {
          return false;
        }
        auto collapseOp =
            cast<tensor::CollapseShapeOp>(operand->get().getDefiningOp());
        return enableEdgeReshapePropagation ||
               isReshapeBlockingFusion(operand->getOwner(),
                                       collapseOp.getSrc().getDefiningOp());
      };
  ControlTransposePropagationFn sinkThroughExpandShapeFn =
      [this](OpOperand *operand) {
        if (!isOutsideDispatch(operand)) {
          return false;
        }
        Operation *transposeOp = operand->get().getDefiningOp();
        return enableEdgeReshapePropagation ||
               llvm::any_of(
                   operand->getOwner()->getUsers(), [&](Operation *consumer) {
                     return isReshapeBlockingFusion(transposeOp, consumer);
                   });
      };

  // First, specialize all transposes to `linalg.transpose`. This dramatically
  // simplifies all subsequent propagation patterns, both in matching and
  // rewriting.
  {
    SmallVector<linalg::GenericOp> genericCandidates;
    funcOp.walk([&](linalg::GenericOp genericOp) {
      if (IREE::Flow::isNonNullAndOutsideDispatch(genericOp)) {
        genericCandidates.push_back(genericOp);
      }
    });
    IRRewriter rewriter(&getContext());
    for (auto genericOp : genericCandidates) {
      rewriter.setInsertionPoint(genericOp);
      (void)specializeGenericTransposeOp(rewriter, genericOp);
    }
  }

  LLVM_DEBUG({
    llvm::dbgs() << "\n--- After specializing transpose ops ---\n";
    funcOp->print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });

  // First try to fuse transposes with some consumer linalg named ops before
  // any reshape propagation. Some transposes may be adjacent to named ops,
  // and it is more canonical if we can fuse the ops into a new named op.
  if (!testBubblingOnly) {
    RewritePatternSet sinkingPatterns(context);
    populateSinkTransposeThroughExtractSlicePattern(
        sinkingPatterns, isUntransposedSliceMappableToFlow);
    populateSinkTransposeThroughExpandShapePattern(sinkingPatterns,
                                                   sinkThroughExpandShapeFn);
    populateMatmulSinkingPatterns(sinkingPatterns);
    populateCommonCanonicalizationPatterns(context, sinkingPatterns);
    populateSinkTransposeThroughUnaryElementwiseInputPattern(
        sinkingPatterns, isOutsideDispatch, /*benefit=*/2);
    if (failed(applyPatternsGreedily(funcOp, std::move(sinkingPatterns)))) {
      funcOp.emitError("Transpose initial sinking patterns failed");
      return signalPassFailure();
    }
  }

  LLVM_DEBUG({
    llvm::dbgs() << "\n--- After canonicalizing transpose in place ---\n";
    funcOp->print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });

  // Propagate transposes upwards, and fuse with any producer generic ops. Also
  // propagate reshapes upwards to open up more transpose fusion opportunities.
  if (!testSinkingOnly) {
    linalg::ControlFusionFn reshapePropagationFn =
        [&](OpOperand *fusedOperand) {
          Operation *producer = fusedOperand->get().getDefiningOp();
          Operation *consumer = fusedOperand->getOwner();
          if (!IREE::Flow::isNonNullAndOutsideDispatch({producer, consumer})) {
            return false;
          }

          // Do not reshape producer linalg op if it has more than one user.
          auto producerLinalgOp = dyn_cast<linalg::LinalgOp>(producer);
          if (!producerLinalgOp || !producerLinalgOp->hasOneUse()) {
            return false;
          }
          // Only reshape generic ops, or any op if aggressive propagation is
          // enabled.
          if (!enableAggressivePropagation &&
              !isa<linalg::GenericOp>(producerLinalgOp)) {
            return false;
          }
          // Only propagate expand_shape ops up through producers because it
          // is always possible to bubble a transpose through an collapse_shape
          // and thus is handled separately.
          if (!isa<tensor::ExpandShapeOp>(consumer)) {
            return false;
          }

          if (!enableEdgeReshapePropagation &&
              llvm::none_of(
                  consumer->getUsers(), [&](Operation *expandConsumer) {
                    return isReshapeBlockingFusion(producer, expandConsumer);
                  })) {
            return false;
          }
          // Only propagate if the immediate consumer of the reshape is a
          // transpose.
          return consumer->hasOneUse() &&
                 isa<linalg::TransposeOp>(*(consumer->user_begin()));
        };
    RewritePatternSet bubblingPatterns(context);
    linalg::populateFoldReshapeOpsByExpansionPatterns(bubblingPatterns,
                                                      reshapePropagationFn);
    linalg::FillOp::getCanonicalizationPatterns(bubblingPatterns, context);

    if (enableAttentionVTranspose) {
      IREE::LinalgExt::populateBubbleTransposeFromLinalgExtOps(
          bubblingPatterns, isOutsideDispatch);
    }
    populateFuseTransposeWithProducerLinalgOpPattern(
        bubblingPatterns, isOutsideDispatch, enableAggressivePropagation,
        enableConvolutionPropagation);
    populateBubbleTransposeThroughCollapseShapePattern(
        bubblingPatterns, bubbleThroughCollapseShapeFn);
    populateBubbleTransposeThroughUnaryElementwiseDpsInitPattern(
        bubblingPatterns, isOutsideDispatch, /*benefit=*/2);
    populateComposeTransposesPattern(bubblingPatterns, isOutsideDispatch);
    populateFuseTransposeIntoDequantizePattern(bubblingPatterns,
                                               isOutsideDispatch);
    populateCommonCanonicalizationPatterns(context, bubblingPatterns);

    GreedyRewriteConfig config;
    config.setMaxIterations(GreedyRewriteConfig::kNoLimit);
    if (failed(applyPatternsGreedily(funcOp, std::move(bubblingPatterns),
                                     config))) {
      funcOp.emitError("Transpose bubbling patterns failed");
      return signalPassFailure();
    }
  }

  LLVM_DEBUG({
    llvm::dbgs() << "\n--- After bubbling transpose ops up ---\n";
    funcOp->print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });

  // Propagate transposes downwards, and fuse with any non-unary generic ops
  // or linalg named ops. Also propagate reshapes downwards to open up more
  // transpose fusion opportunities.
  if (!testBubblingOnly) {
    RewritePatternSet sinkingPatterns(context);
    linalg::ControlFusionFn reshapePropagationFn =
        [&](OpOperand *fusedOperand) {
          Operation *producer = fusedOperand->get().getDefiningOp();
          Operation *consumer = fusedOperand->getOwner();
          if (!IREE::Flow::isNonNullAndOutsideDispatch({producer, consumer})) {
            return false;
          }
          auto consumerLinalgOp = dyn_cast<linalg::LinalgOp>(consumer);
          if (!consumerLinalgOp || consumerLinalgOp.getNumReductionLoops()) {
            return false;
          }
          // Only reshape generic ops.
          if (!enableAggressivePropagation &&
              !isa<linalg::GenericOp>(consumerLinalgOp)) {
            return false;
          }
          // Only propagate collapse_shape ops down through consumers because it
          // is always possible to sink a transpose through an expand_shape and
          // thus is handled separately.
          if (!isa<tensor::CollapseShapeOp>(producer)) {
            return false;
          }

          if (!enableEdgeReshapePropagation &&
              !isReshapeBlockingFusion(producer->getOperand(0).getDefiningOp(),
                                       consumer)) {
            return false;
          }

          // Require that the immediate producer of the reshape is a transpose.
          return isa_and_nonnull<linalg::TransposeOp>(
              producer->getOperand(0).getDefiningOp());
        };
    linalg::populateFoldReshapeOpsByExpansionPatterns(sinkingPatterns,
                                                      reshapePropagationFn);
    populateSinkTransposeThroughExtractSlicePattern(
        sinkingPatterns, isUntransposedSliceMappableToFlow);
    populateSinkTransposeThroughExpandShapePattern(sinkingPatterns,
                                                   sinkThroughExpandShapeFn);
    if (enableSinkTransposeThroughPad) {
      populateSinkTransposeThroughPadPattern(sinkingPatterns,
                                             isOutsideDispatch);
    }
    populateFuseTransposeWithLinalgOpConsumerPattern(
        sinkingPatterns, isOutsideDispatch, enableAggressivePropagation,
        enableConvolutionPropagation);
    populateComposeTransposesPattern(sinkingPatterns, isOutsideDispatch);
    populateMatmulSinkingPatterns(sinkingPatterns);
    populateCommonCanonicalizationPatterns(context, sinkingPatterns);
    populateSinkTransposeThroughUnaryElementwiseInputPattern(
        sinkingPatterns, isOutsideDispatch, /*benefit=*/2);
    GreedyRewriteConfig config;
    // TODO: This is inefficient. Consider rewriting this pass to use a
    // worklist of just the transpose operations.
    config.setMaxIterations(GreedyRewriteConfig::kNoLimit);
    if (failed(applyPatternsGreedily(funcOp, std::move(sinkingPatterns),
                                     config))) {
      funcOp.emitError("Transpose sinking patterns failed");
      return signalPassFailure();
    }
  }

  LLVM_DEBUG({
    llvm::dbgs() << "\n--- After sinking transpose ops down ---\n";
    funcOp->print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });
}

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createPropagateLinalgTransposePass(bool enableAggressivePropagation) {
  return std::make_unique<PropagateLinalgTransposePass>(
      enableAggressivePropagation);
}

} // namespace mlir::iree_compiler::GlobalOptimization

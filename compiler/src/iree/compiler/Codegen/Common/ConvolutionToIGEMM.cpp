// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_CONVOLUTIONTOIGEMMPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

using iree_compiler::IREE::LinalgExt::IREELinalgExtDialect;

/// Generalize a specific named op to a linalg.generic.
template <typename OpTy>
struct GeneralizeNamedOp : OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    return linalg::generalizeNamedOp(rewriter,
                                     cast<linalg::LinalgOp>(op.getOperation()));
  }
};

/// Pattern to set a lowering configuration on an IGEMM convolution. Searches
/// for a contraction with a linalg_ext.im2col producer, and calls the configFn
/// to set the configuration.
/// TODO(Max191): Use a funcOp walk instead of a pattern for this.
struct SetIGEMMConfiguration final : OpRewritePattern<linalg::GenericOp> {
  using Base::Base;

  SetIGEMMConfiguration(MLIRContext *context, IGEMMConfigFn configFn)
      : OpRewritePattern(context), configFn(configFn) {}

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    if (!linalg::isaContractionOpInterface(genericOp)) {
      return failure();
    }

    IREE::LinalgExt::Im2colOp im2colOp;
    for (auto operand : genericOp.getDpsInputs()) {
      im2colOp = operand.getDefiningOp<IREE::LinalgExt::Im2colOp>();
      if (im2colOp) {
        break;
      }
    }
    if (!im2colOp) {
      return rewriter.notifyMatchFailure(genericOp, "no im2colOp producer.");
    }

    if (getLoweringConfig(genericOp)) {
      return rewriter.notifyMatchFailure(genericOp,
                                         "genericOp has a lowering config.");
    }
    if (getLoweringConfig(im2colOp)) {
      return rewriter.notifyMatchFailure(im2colOp,
                                         "im2colOp has a lowering config.");
    }

    if (failed(configFn(genericOp, im2colOp))) {
      return rewriter.notifyMatchFailure(genericOp,
                                         "failed to set config on igemm_conv.");
    }

    return success();
  }

private:
  IGEMMConfigFn configFn;
};

class ConvolutionToIGEMMPass final
    : public impl::ConvolutionToIGEMMPassBase<ConvolutionToIGEMMPass> {
public:
  using ConvolutionToIGEMMPassBase::ConvolutionToIGEMMPassBase;

  ConvolutionToIGEMMPass(std::optional<IGEMMConfigFn> configFn,
                         std::optional<IGEMMControlFn> controlFn)
      : configFn(configFn), controlFn(controlFn) {}

  void runOnOperation() override;

private:
  std::optional<IGEMMConfigFn> configFn;
  std::optional<IGEMMControlFn> controlFn;
};

} // namespace

LogicalResult
convertToIGEMMAndSetConfig(FunctionOpInterface funcOp,
                           std::optional<IGEMMConfigFn> configFn,
                           std::optional<IGEMMControlFn> controlFn) {
  // Rewrite convolutions into a im2col and GEMM.
  MLIRContext *context = funcOp->getContext();
  {
    RewritePatternSet patterns(context);
    iree_compiler::IREE::LinalgExt::populateConvToIm2colOpPatterns(patterns,
                                                                   controlFn);
    if (configFn.has_value()) {
      patterns.add<SetIGEMMConfiguration>(context, configFn.value());
    }
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      return failure();
    }
  }

  // Materialize implicit broadcasts in element-wise consumer ops. Consumer
  // generics with non-identity indexing maps (e.g., per-row bias with map
  // (d0,d1,d2,d3) -> (d1)) cannot be folded through by the reshape
  // propagation patterns below. Materialize the broadcasts explicitly to
  // turn consumers into pure element-wise ops with identity maps.
  {
    RewritePatternSet materializeBroadcastPatterns(context);
    linalg::populateDecomposeProjectedPermutationPatterns(
        materializeBroadcastPatterns);
    if (failed(applyPatternsGreedily(
            funcOp, std::move(materializeBroadcastPatterns)))) {
      return failure();
    }
  }

  // The im2col transformation collapses some of the dimensions of the
  // convolution operands. Try to push the reshape ops towards the boundaries
  // of the function and fold with interface tensor ops.
  //
  // TODO(Max191): Allow for the im2col op to have multiple M dimensions, and
  //   generate a multi-M dim contraction instead of collapsing and
  //   propagating reshapes. It should ultimately become a pass option to
  //   decide whether to collapse the contraction dimensions into a single
  //   M/N/K dimension.
  {
    RewritePatternSet bubbleCollapseShapePatterns(context);
    linalg::ControlFusionFn bubbleUpExpansionControlFn =
        [](OpOperand *fusedOperand) {
          Operation *producer = fusedOperand->get().getDefiningOp();
          Operation *consumer = fusedOperand->getOwner();

          // Block only if one of the operations has a lowering configuration
          // which means it likely expects tiling specific to its original
          // shape.
          if (getLoweringConfig(producer) || getLoweringConfig(consumer)) {
            return false;
          }
          return true;
        };
    linalg::populateFoldReshapeOpsByCollapsingPatterns(
        bubbleCollapseShapePatterns, bubbleUpExpansionControlFn);
    // Add patterns to do some additional cleanup (on top of canonicalizations
    // that can be done later) of reshape ops.
    tensor::populateFoldTensorEmptyPatterns(bubbleCollapseShapePatterns);
    linalg::FillOp::getCanonicalizationPatterns(bubbleCollapseShapePatterns,
                                                context);
    tensor::CollapseShapeOp::getCanonicalizationPatterns(
        bubbleCollapseShapePatterns, context);
    tensor::EmptyOp::getCanonicalizationPatterns(bubbleCollapseShapePatterns,
                                                 context);
    tensor::ExpandShapeOp::getCanonicalizationPatterns(
        bubbleCollapseShapePatterns, context);
    populateReshapeToInterfaceTensorPatterns(bubbleCollapseShapePatterns);
    populateFoldTensorReshapeIntoBufferPatterns(bubbleCollapseShapePatterns);
    populateSwapExtractWithExpandPattern(bubbleCollapseShapePatterns,
                                         bubbleUpExpansionControlFn);
    populateCollapseDestinationForallPatterns(bubbleCollapseShapePatterns);
    if (failed(applyPatternsGreedily(funcOp,
                                     std::move(bubbleCollapseShapePatterns)))) {
      return failure();
    }
  }
  // Re-fuse the materialized broadcasts back into their element-wise
  // consumers. The decomposition above created explicit linalg.broadcast
  // and linalg.transpose ops so reshape propagation could fold through
  // identity-map generics. Now that reshapes have been pushed to the
  // boundaries, generalize those named ops to generics and fuse them back
  // into their consumers to produce compact element-wise generics.
  {
    RewritePatternSet fusionPatterns(context);
    // Generalize only broadcast/transpose to generics so elementwise
    // fusion can fold them into their consumers.
    fusionPatterns.add<GeneralizeNamedOp<linalg::BroadcastOp>,
                       GeneralizeNamedOp<linalg::TransposeOp>>(context);
    linalg::populateElementwiseOpsFusionPatterns(
        fusionPatterns, [](OpOperand *) { return true; });
    if (failed(applyPatternsGreedily(funcOp, std::move(fusionPatterns)))) {
      return failure();
    }
  }
  return success();
}

void ConvolutionToIGEMMPass::runOnOperation() {
  if (failed(convertToIGEMMAndSetConfig(getOperation()))) {
    return signalPassFailure();
  }
}

} // namespace mlir::iree_compiler

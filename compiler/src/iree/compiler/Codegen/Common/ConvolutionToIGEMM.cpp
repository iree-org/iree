// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_CONVOLUTIONTOIGEMMPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

using iree_compiler::IREE::LinalgExt::IREELinalgExtDialect;

struct SetIGEMMConfiguration final : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  SetIGEMMConfiguration(MLIRContext *context, ConfigFn configFn)
      : OpRewritePattern(context), configFn(configFn) {}

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    if (!linalg::isaContractionOpInterface(genericOp)) {
      return failure();
    }

    auto im2colOp =
        genericOp.getOperand(0).getDefiningOp<IREE::LinalgExt::Im2colOp>();
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
  ConfigFn configFn;
};

class ConvolutionToIGEMMPass final
    : public impl::ConvolutionToIGEMMPassBase<ConvolutionToIGEMMPass> {
public:
  using ConvolutionToIGEMMPassBase::ConvolutionToIGEMMPassBase;

  explicit ConvolutionToIGEMMPass(ConfigFn configFn) : configFn(configFn) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tensor::TensorDialect, IREELinalgExtDialect>();
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();

    // Rewrite convolutions into a im2col and GEMM.
    {
      auto conv2dToIm2colControlFn = [](Operation *conv) {
        // Don't transform convolutions that have a preset lowering config.
        if (getLoweringConfig(conv)) {
          return false;
        }
        return true;
      };
      MLIRContext *context = &getContext();
      RewritePatternSet patterns(context);
      iree_compiler::IREE::LinalgExt::populateConv2DToIm2colOpPatterns(
          patterns, conv2dToIm2colControlFn);
      patterns.add<SetIGEMMConfiguration>(context, configFn);
      if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                              std::move(patterns)))) {
        return signalPassFailure();
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
      if (failed(applyPatternsAndFoldGreedily(
              getOperation(), std::move(bubbleCollapseShapePatterns)))) {
        return signalPassFailure();
      }
    }
  }

private:
  ConfigFn configFn = [](linalg::GenericOp genericOp,
                         IREE::LinalgExt::Im2colOp im2colOp) {
    return failure();
  };
};

} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createConvolutionToIGEMMPass(ConfigFn configFn) {
  return std::make_unique<ConvolutionToIGEMMPass>(configFn);
}

} // namespace mlir::iree_compiler

// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/PassDetail.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

namespace {

using iree_compiler::IREE::LinalgExt::IREELinalgExtDialect;

class ConvolutionToIGEMMPass
    : public ConvolutionToIGEMMBase<ConvolutionToIGEMMPass> {
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
      RewritePatternSet patterns(&getContext());
      iree_compiler::IREE::LinalgExt::populateConv2DToIm2colOpPatterns(
          patterns, conv2dToIm2colControlFn);
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
};

} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createConvolutionToIGEMMPass() {
  return std::make_unique<ConvolutionToIGEMMPass>();
}

} // namespace mlir::iree_compiler

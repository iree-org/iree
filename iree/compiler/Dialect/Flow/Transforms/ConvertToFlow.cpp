// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Conversion/TensorToFlow/ConvertTensorToFlow.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowTypes.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-flow-convert-to-flow"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

namespace {

/// Generates `tensor.dim` operations to get the dynamic sizes of a value `v`.
static SmallVector<Value, 4> getDynamicDimValues(OpBuilder &b, Location loc,
                                                 Value v) {
  SmallVector<Value, 4> dynamicDims;
  for (auto dim : llvm::enumerate(v.getType().cast<ShapedType>().getShape())) {
    if (dim.value() != ShapedType::kDynamicSize) continue;
    dynamicDims.push_back(b.createOrFold<tensor::DimOp>(loc, v, dim.index()));
  }
  return dynamicDims;
}

/// Converts linalg.tensor_reshape operations into flow.tensor.reshape
/// operations.
template <typename TensorReshapeOp>
struct LinalgTensorReshapeToFlowTensorReshape
    : public OpRewritePattern<TensorReshapeOp> {
  using OpRewritePattern<TensorReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TensorReshapeOp reshapeOp,
                                PatternRewriter &rewriter) const override {
    if (reshapeOp->template getParentOfType<Flow::DispatchWorkgroupsOp>()) {
      return failure();
    }
    SmallVector<SmallVector<Value>> outputShape;
    ReifyRankedShapedTypeOpInterface reifyShapedTypeInterface =
        cast<ReifyRankedShapedTypeOpInterface>(reshapeOp.getOperation());
    if (failed(reifyShapedTypeInterface.reifyResultShapes(rewriter,
                                                          outputShape))) {
      return failure();
    }
    SmallVector<Value> outputDynamicShapes;
    for (auto shape :
         llvm::zip(reshapeOp.getResultType().getShape(), outputShape[0])) {
      if (std::get<0>(shape) != ShapedType::kDynamicSize) continue;
      outputDynamicShapes.push_back(std::get<1>(shape));
    }
    rewriter.replaceOpWithNewOp<IREE::Flow::TensorReshapeOp>(
        reshapeOp, reshapeOp.getResultType(), reshapeOp.src(),
        outputDynamicShapes);
    return success();
  }
};

/// Converts linalg.fill ops into flow.tensor.splat ops.
///
/// This is expected to improve performance because we can use DMA
/// functionalities for the fill, instead of dispatching kernels.
struct LinalgFillToFlowTensorSplat final
    : public OpRewritePattern<linalg::FillOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::FillOp fillOp,
                                PatternRewriter &rewriter) const override {
    if (fillOp->getParentOfType<IREE::Flow::DispatchWorkgroupsOp>()) {
      // Don't convert linalg.fill ops that were fused together with other ops.
      return failure();
    }

    SmallVector<Value, 4> dynamicDims =
        getDynamicDimValues(rewriter, fillOp.getLoc(), fillOp.output());
    rewriter.replaceOpWithNewOp<TensorSplatOp>(
        fillOp, fillOp.output().getType(), fillOp.value(), dynamicDims);
    return success();
  }
};

struct ConvertToFlowBeforeDispatchFormation
    : public ConvertToFlowBeforeDispatchFormationBase<
          ConvertToFlowBeforeDispatchFormation> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Flow::FlowDialect, tensor::TensorDialect,
                    linalg::LinalgDialect, mlir::StandardOpsDialect,
                    mlir::arith::ArithmeticDialect, mlir::math::MathDialect>();
  }
  void runOnOperation() override {
    auto funcOp = getOperation();
    MLIRContext *context = funcOp->getContext();
    context->allowUnregisteredDialects(true);
    RewritePatternSet patterns(&getContext());

    patterns
        .insert<LinalgTensorReshapeToFlowTensorReshape<tensor::CollapseShapeOp>,
                LinalgTensorReshapeToFlowTensorReshape<tensor::ExpandShapeOp>>(
            context);
    populateTensorToFlowPatternsBeforeDispatchFormation(context, patterns);
    IREE::Flow::TensorReshapeOp::getCanonicalizationPatterns(patterns, context);

    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

struct ConvertToFlowAfterDispatchFormation
    : public ConvertToFlowAfterDispatchFormationBase<
          ConvertToFlowAfterDispatchFormation> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Flow::FlowDialect, tensor::TensorDialect,
                    linalg::LinalgDialect, mlir::StandardOpsDialect,
                    mlir::arith::ArithmeticDialect, mlir::math::MathDialect>();
  }
  void runOnOperation() override {
    auto funcOp = getOperation();
    MLIRContext *context = funcOp->getContext();
    context->allowUnregisteredDialects(true);
    RewritePatternSet patterns(&getContext());

    patterns.insert<LinalgFillToFlowTensorSplat>(context);
    populateTensorToFlowPatternsAfterDispatchFormation(context, patterns);
    IREE::Flow::TensorReshapeOp::getCanonicalizationPatterns(patterns, context);

    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<Pass> createConvertToFlowBeforeDispatchFormation() {
  return std::make_unique<ConvertToFlowBeforeDispatchFormation>();
}

std::unique_ptr<Pass> createConvertToFlowAfterDispatchFormation() {
  return std::make_unique<ConvertToFlowAfterDispatchFormation>();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

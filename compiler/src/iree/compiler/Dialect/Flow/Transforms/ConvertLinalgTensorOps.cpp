// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowTypes.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-flow-convert-linalg-tensor-ops"

namespace mlir::iree_compiler::IREE::Flow {

namespace {

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
    for (auto shape : llvm::zip_equal(reshapeOp.getResultType().getShape(),
                                      outputShape[0])) {
      if (!ShapedType::isDynamic(std::get<0>(shape))
        continue;
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
    SmallVector<Value> dynamicDims = tensor::createDynamicDimValues(
        rewriter, fillOp.getLoc(), fillOp.output());
    rewriter.replaceOpWithNewOp<TensorSplatOp>(
        fillOp, fillOp.output().getType(), fillOp.value(), dynamicDims);
    return success();
  }
};

struct ConvertSplatConstantOp
    : public OpRewritePattern<mlir::func::ConstantOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(mlir::func::ConstantOp op,
                                PatternRewriter &rewriter) const override {
    if (op->getParentOfType<IREE::Flow::DispatchWorkgroupsOp>()) {
      return rewriter.notifyMatchFailure(op, "ignoring dispatch ops");
    }
    auto splatAttr = op.getValue().dyn_cast<SplatElementsAttr>();
    if (!splatAttr) {
      return rewriter.notifyMatchFailure(op, "only looking for splats");
    }
    auto tensorType = op.getType().cast<TensorType>();
    auto elementValue = rewriter.createOrFold<mlir::func::ConstantOp>(
        op.getLoc(), tensorType.getElementType(),
        splatAttr.getSplatValue<Attribute>());
    rewriter.replaceOpWithNewOp<IREE::Flow::TensorSplatOp>(
        op, tensorType, elementValue, ValueRange{});
    return success();
  }
};

/// Converts linalg operations that can map to flow.tensor.* operations.
struct ConvertLinalgTensorOpsPass
    : public ConvertLinalgTensorOpsBase<ConvertLinalgTensorOpsPass> {
  ConvertLinalgTensorOpsPass(bool runBefore) {
    runBeforeDispatchRegionFormation = runBefore;
  }
  ConvertLinalgTensorOpsPass(const ConvertLinalgTensorOpsPass &that) {
    runBeforeDispatchRegionFormation = that.runBeforeDispatchRegionFormation;
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Flow::FlowDialect, tensor::TensorDialect,
                    linalg::LinalgDialect, mlir::func::FuncDialect,
                    mlir::arith::ArithDialect, mlir::math::MathDialect>();
  }
  void runOnOperation() override {
    auto funcOp = getOperation();
    MLIRContext *context = funcOp->getContext();
    RewritePatternSet patterns(&getContext());
    if (runBeforeDispatchRegionFormation) {
      patterns.insert<
          LinalgTensorReshapeToFlowTensorReshape<linalg::TensorCollapseShapeOp>,
          LinalgTensorReshapeToFlowTensorReshape<linalg::TensorExpandShapeOp>>(
          context);
    } else {
      patterns.insert<LinalgFillToFlowTensorSplat, ConvertSplatConstantOp>(
          context);
    }
    IREE::Flow::TensorReshapeOp::getCanonicalizationPatterns(patterns, context);
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<mlir::func::FuncOp>>
createConvertLinalgTensorOpsPass(bool runBeforeDispatchRegionFormation) {
  return std::make_unique<ConvertLinalgTensorOpsPass>(
      runBeforeDispatchRegionFormation);
}

} // namespace mlir::iree_compiler::IREE::Flow

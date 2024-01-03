// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler::IREE::Flow {

namespace {

/// tensor::ExtractOp will be lowered to IREE::Flow::TensorLoadOp. If the type
/// is i1, it's not valid to load. In this case, we need to cast it to i8 before
/// the load, and truncate the value after the load.
struct ExtractElementOpLowering
    : public OpConversionPattern<tensor::ExtractOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tensor::ExtractOp op, ArrayRef<Value> args,
                  ConversionPatternRewriter &rewriter) const override {
    // tensor<i1> is not valid to load, it needs to be converted to i8 or
    // something else instead.
    auto tensorType = op.tensor().getType().cast<TensorType>();
    if (tensorType.getElementType().isInteger(1)) {
      auto i1Type = rewriter.getI1Type();
      auto i8Type = rewriter.getIntegerType(8);
      auto convertedOperand = rewriter.createOrFold<arith::ExtUIOp>(
          op.getLoc(), args[0],
          RankedTensorType::get(tensorType.getShape(), i8Type));
      auto i8Value = rewriter.createOrFold<IREE::Flow::TensorLoadOp>(
          op.getLoc(), i8Type, convertedOperand, op.getIndices());
      rewriter.replaceOpWithNewOp<arith::TruncIOp>(op, i1Type, i8Value);
    } else {
      rewriter.replaceOpWithNewOp<IREE::Flow::TensorLoadOp>(
          op, tensorType.getElementType(), op.tensor(), op.getIndices());
    }
    return success();
  }
};

void setupStandardToFlowTensorLoadLegality(MLIRContext *context,
                                           ConversionTarget &conversionTarget) {
  conversionTarget.addIllegalOp<tensor::ExtractOp>();
}

void populateStandardToFlowTensorLoadPatterns(MLIRContext *context,
                                              RewritePatternSet &patterns) {
  patterns.insert<ExtractElementOpLowering>(context);
}

} // namespace

class PromoteTensorLoadsPass
    : public PromoteTensorLoadsBase<PromoteTensorLoadsPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<FlowDialect, func::FuncDialect, mlir::arith::ArithDialect,
                    mlir::math::MathDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override {
    auto *context = &getContext();
    ConversionTarget conversionTarget(*context);
    RewritePatternSet conversionPatterns(&getContext());

    conversionTarget.addLegalDialect<IREE::Flow::FlowDialect>();
    conversionTarget
        .addLegalDialect<func::FuncDialect, mlir::arith::ArithDialect,
                         mlir::math::MathDialect>();
    setupStandardToFlowTensorLoadLegality(context, conversionTarget);
    populateStandardToFlowTensorLoadPatterns(context, conversionPatterns);

    if (failed(applyPartialConversion(getOperation(), conversionTarget,
                                      std::move(conversionPatterns)))) {
      return signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<mlir::func::FuncOp>>
createPromoteTensorLoadsPass() {
  return std::make_unique<PromoteTensorLoadsPass>();
}

} // namespace mlir::iree_compiler::IREE::Flow

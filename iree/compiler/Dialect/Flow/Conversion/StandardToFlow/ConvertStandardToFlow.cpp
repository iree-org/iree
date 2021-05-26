// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Conversion/StandardToFlow/ConvertStandardToFlow.h"

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace iree_compiler {

namespace {

/// tensor::ExtractOp will be lowered to IREE::Flow::TensorLoadOp. If the type
/// is i1, it's not valid to load. In this case, we need to cast it to i8 before
/// the load, and truncate the value after the load.
struct ExtractElementOpLowering
    : public OpConversionPattern<tensor::ExtractOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      tensor::ExtractOp op, ArrayRef<Value> args,
      ConversionPatternRewriter &rewriter) const override {
    // tensor<i1> is not valid to load, it needs to be converted to i8 or
    // something else instead.
    auto tensorType = op.tensor().getType().cast<TensorType>();
    if (tensorType.getElementType().isInteger(1)) {
      auto i1Type = rewriter.getI1Type();
      auto i8Type = rewriter.getIntegerType(8);
      auto convertedOperand = rewriter.createOrFold<ZeroExtendIOp>(
          op.getLoc(), args[0],
          RankedTensorType::get(tensorType.getShape(), i8Type));
      auto i8Value = rewriter.createOrFold<IREE::Flow::TensorLoadOp>(
          op.getLoc(), i8Type, convertedOperand, op.indices());
      rewriter.replaceOpWithNewOp<TruncateIOp>(op, i1Type, i8Value);
    } else {
      rewriter.replaceOpWithNewOp<IREE::Flow::TensorLoadOp>(
          op, tensorType.getElementType(), op.tensor(), op.indices());
    }
    return success();
  }
};

}  // namespace

void setupStandardToFlowTensorLoadLegality(MLIRContext *context,
                                           ConversionTarget &conversionTarget) {
  conversionTarget.addIllegalOp<tensor::ExtractOp>();
}

void populateStandardToFlowTensorLoadPatterns(
    MLIRContext *context, OwningRewritePatternList &patterns) {
  patterns.insert<ExtractElementOpLowering>(context);
}

}  // namespace iree_compiler
}  // namespace mlir

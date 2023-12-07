// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Stream/Conversion/StandardToStream/Patterns.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

namespace {

struct ConvertTensorConstantOp : public OpConversionPattern<arith::ConstantOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arith::ConstantOp constantOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Only handle tensor types - other arith.constant types (like i32) are
    // ignored.
    if (!llvm::isa<TensorType>(constantOp.getType()))
      return failure();

    Type constantType = IREE::Stream::ResourceType::get(
        getContext(), IREE::Stream::Lifetime::Constant);
    auto newOp = rewriter.create<IREE::Stream::TensorConstantOp>(
        constantOp.getLoc(), constantType,
        llvm::cast<ElementsAttr>(constantOp.getValue()),
        TypeAttr::get(constantOp.getType()),
        /*result_encoding_dims=*/ValueRange{},
        /*affinity=*/nullptr);

    Type unknownType = IREE::Stream::ResourceType::get(getContext());
    auto constantSize = rewriter.createOrFold<IREE::Stream::ResourceSizeOp>(
        constantOp.getLoc(), rewriter.getIndexType(), newOp.getResult());
    rewriter.replaceOpWithNewOp<IREE::Stream::AsyncTransferOp>(
        constantOp, unknownType, newOp.getResult(), constantSize, constantSize,
        /*source_affinity=*/nullptr,
        /*result_affinity=*/nullptr);
    return success();
  }
};

} // namespace

void populateStandardConstantToStreamPatterns(
    MLIRContext *context, ConversionTarget &conversionTarget,
    TypeConverter &typeConverter, RewritePatternSet &patterns) {
  conversionTarget.addDynamicallyLegalOp<arith::ConstantOp>(
      [](arith::ConstantOp op) {
        return !llvm::isa<TensorType>(op.getType());
      });

  patterns.insert<ConvertTensorConstantOp>(typeConverter, context);
}

} // namespace mlir::iree_compiler

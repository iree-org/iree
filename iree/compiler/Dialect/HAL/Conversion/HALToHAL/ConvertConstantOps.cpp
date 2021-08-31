// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace {

class ConstantSubspanConversion
    : public OpConversionPattern<IREE::HAL::ConstantSubspanOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      IREE::HAL::ConstantSubspanOp op, llvm::ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto bufferValue = rewriter.createOrFold<IREE::Util::GlobalLoadOp>(
        op.getLoc(), IREE::HAL::BufferType::get(rewriter.getContext()),
        op.runtime_buffer().getLeafReference().getValue());
    auto offsetValue = rewriter.createOrFold<mlir::ConstantIndexOp>(
        op.getLoc(), op.runtime_range().getOffset());
    auto lengthValue = rewriter.createOrFold<mlir::ConstantIndexOp>(
        op.getLoc(), op.runtime_range().getLength());
    rewriter.replaceOpWithNewOp<IREE::HAL::BufferSubspanOp>(
        op, bufferValue.getType(), bufferValue, offsetValue, lengthValue);
    return success();
  }
};

}  // namespace

void populateHALConstantToHALPatterns(MLIRContext *context,
                                      OwningRewritePatternList &patterns,
                                      TypeConverter &typeConverter) {
  patterns.insert<ConstantSubspanConversion>(typeConverter, context);
}

}  // namespace iree_compiler
}  // namespace mlir

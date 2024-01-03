// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/Dialect/VM/Conversion/TypeConverter.h"
#include "iree/compiler/Dialect/VM/Conversion/UtilToVM/ConvertUtilToVM.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

namespace {

//===----------------------------------------------------------------------===//
// util.switch
//===----------------------------------------------------------------------===//

struct SwitchOpConversion : public OpConversionPattern<IREE::Util::SwitchOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Util::SwitchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto index = rewriter.createOrFold<IREE::VM::TruncI64I32Op>(
        op.getLoc(), rewriter.getI32Type(), adaptor.getIndex());
    auto type = adaptor.getDefaultValue().getType();
    if (type.isInteger(32)) {
      rewriter.replaceOpWithNewOp<IREE::VM::SwitchI32Op>(
          op, type, index, adaptor.getDefaultValue(), adaptor.getValues());
    } else if (type.isInteger(64)) {
      rewriter.replaceOpWithNewOp<IREE::VM::SwitchI64Op>(
          op, type, index, adaptor.getDefaultValue(), adaptor.getValues());
    } else if (type.isF32()) {
      rewriter.replaceOpWithNewOp<IREE::VM::SwitchF32Op>(
          op, type, index, adaptor.getDefaultValue(), adaptor.getValues());
    } else if (type.isF64()) {
      rewriter.replaceOpWithNewOp<IREE::VM::SwitchF64Op>(
          op, type, index, adaptor.getDefaultValue(), adaptor.getValues());
    } else {
      // TODO(benvanik): support other types by coercing the values.
      return rewriter.notifyMatchFailure(
          op, "unsupported type; needs widening/narrowing");
    }
    return success();
  }
};

} // namespace

void populateUtilAssignmentToVMPatterns(MLIRContext *context,
                                        ConversionTarget &conversionTarget,
                                        TypeConverter &typeConverter,
                                        RewritePatternSet &patterns) {
  conversionTarget.addIllegalOp<IREE::Util::SwitchOp>();

  patterns.insert<SwitchOpConversion>(typeConverter, context);
}

} // namespace mlir::iree_compiler

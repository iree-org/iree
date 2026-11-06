// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

class StatusCheckOkOpConversion final
    : public OpConversionPattern<IREE::Util::StatusCheckOkOp> {
public:
  StatusCheckOkOpConversion(MLIRContext *context, TypeConverter &typeConverter)
      : OpConversionPattern(context) {}

  LogicalResult
  matchAndRewrite(IREE::Util::StatusCheckOkOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    APInt status;
    if (matchPattern(adaptor.getStatus(), m_ConstantInt(&status)) &&
        status.isZero()) {
      rewriter.eraseOp(op);
      return success();
    }
    // If status value is non-zero, fail.
    rewriter.replaceOpWithNewOp<IREE::VM::CondFailOp>(
        op, adaptor.getStatus(), op.getMessage().value_or(""));
    return success();
  }
};

void populateUtilStatusToVMPatterns(MLIRContext *context,
                                    ConversionTarget &conversionTarget,
                                    TypeConverter &typeConverter,
                                    RewritePatternSet &patterns) {
  conversionTarget.addIllegalOp<IREE::Util::StatusCheckOkOp>();
  patterns.insert<StatusCheckOkOpConversion>(context, typeConverter);
}

} // namespace mlir::iree_compiler

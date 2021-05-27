// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

class CheckSuccessOpConversion
    : public OpConversionPattern<IREE::HAL::CheckSuccessOp> {
 public:
  CheckSuccessOpConversion(MLIRContext *context, SymbolTable &importSymbols,
                           TypeConverter &typeConverter, StringRef importName)
      : OpConversionPattern(context) {}

  LogicalResult matchAndRewrite(
      IREE::HAL::CheckSuccessOp op, llvm::ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // If status value is non-zero, fail.
    rewriter.replaceOpWithNewOp<IREE::VM::CondFailOp>(
        op, op.status(), op.message().getValueOr(""));
    return success();
  }
};

void populateHALControlFlowToVMPatterns(MLIRContext *context,
                                        SymbolTable &importSymbols,
                                        TypeConverter &typeConverter,
                                        OwningRewritePatternList &patterns) {
  patterns.insert<CheckSuccessOpConversion>(context, importSymbols,
                                            typeConverter, "hal.check_success");
}

}  // namespace iree_compiler
}  // namespace mlir

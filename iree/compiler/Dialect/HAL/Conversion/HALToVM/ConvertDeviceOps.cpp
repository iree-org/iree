// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/VM/Conversion/ImportUtils.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

class DeviceQueryI32OpConversion
    : public OpConversionPattern<IREE::HAL::DeviceQueryOp> {
 public:
  DeviceQueryI32OpConversion(MLIRContext *context, SymbolTable &importSymbols,
                             TypeConverter &typeConverter, StringRef importName)
      : OpConversionPattern(typeConverter, context) {
    importOp = importSymbols.lookup<IREE::VM::ImportOp>(importName);
    assert(importOp);
  }

  LogicalResult matchAndRewrite(
      IREE::HAL::DeviceQueryOp op, llvm::ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    if (!op.value().getType().isInteger(32)) return failure();
    IREE::HAL::DeviceQueryOp::Adaptor adaptor(operands);
    return rewriteToCall(op, adaptor, importOp, *getTypeConverter(), rewriter);
  }

 private:
  mutable IREE::VM::ImportOp importOp;
};

void populateHALDeviceToVMPatterns(MLIRContext *context,
                                   SymbolTable &importSymbols,
                                   TypeConverter &typeConverter,
                                   OwningRewritePatternList &patterns) {
  patterns.insert<VMImportOpConversion<IREE::HAL::DeviceAllocatorOp>>(
      context, importSymbols, typeConverter, "hal.device.allocator");
  patterns.insert<VMImportOpConversion<IREE::HAL::DeviceMatchIDOp>>(
      context, importSymbols, typeConverter, "hal.device.match.id");
  patterns.insert<DeviceQueryI32OpConversion>(
      context, importSymbols, typeConverter, "hal.device.query.i32");
}

}  // namespace iree_compiler
}  // namespace mlir

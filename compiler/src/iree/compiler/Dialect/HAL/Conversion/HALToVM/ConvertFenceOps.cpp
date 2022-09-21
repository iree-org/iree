// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/VM/Conversion/ImportUtils.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

namespace {

struct FenceCreateOpConversion
    : public OpConversionPattern<IREE::HAL::FenceCreateOp> {
  FenceCreateOpConversion(MLIRContext *context, SymbolTable &importSymbols,
                          TypeConverter &typeConverter, StringRef importName)
      : OpConversionPattern(context) {
    importOp = importSymbols.lookup<IREE::VM::ImportOp>(importName);
    assert(importOp);
  }
  LogicalResult matchAndRewrite(
      IREE::HAL::FenceCreateOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto importType = importOp.getFunctionType();

    SmallVector<Value, 8> callOperands;
    SmallVector<int16_t, 5> segmentSizes = {
        /*timepoints=*/
        static_cast<int16_t>(adaptor.getSemaphores().size()),
    };
    for (auto it : llvm::zip(adaptor.getSemaphores(), adaptor.getMinValues())) {
      callOperands.push_back(std::get<0>(it));
      callOperands.push_back(std::get<1>(it));
    }

    auto callOp = rewriter.replaceOpWithNewOp<IREE::VM::CallVariadicOp>(
        op, SymbolRefAttr::get(importOp), importType.getResults(), segmentSizes,
        importType.getInputs(), callOperands);
    copyImportAttrs(importOp, callOp);
    return success();
  }

  mutable IREE::VM::ImportOp importOp;
};

}  // namespace

void populateHALFenceToVMPatterns(MLIRContext *context,
                                  SymbolTable &importSymbols,
                                  TypeConverter &typeConverter,
                                  RewritePatternSet &patterns) {
  patterns.insert<FenceCreateOpConversion>(context, importSymbols,
                                           typeConverter, "hal.fence.create");
  patterns.insert<VMImportOpConversion<IREE::HAL::FenceJoinOp>>(
      context, importSymbols, typeConverter, "hal.fence.join");
  patterns.insert<VMImportOpConversion<IREE::HAL::FenceSignalOp>>(
      context, importSymbols, typeConverter, "hal.fence.signal");
  patterns.insert<VMImportOpConversion<IREE::HAL::FenceFailOp>>(
      context, importSymbols, typeConverter, "hal.fence.fail");
  patterns.insert<VMImportOpConversion<IREE::HAL::FenceAwaitOp>>(
      context, importSymbols, typeConverter, "hal.fence.await");
}

}  // namespace iree_compiler
}  // namespace mlir

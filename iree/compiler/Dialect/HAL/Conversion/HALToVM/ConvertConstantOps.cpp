// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace {

class ConstantPoolOpConversion
    : public OpConversionPattern<IREE::HAL::ConstantPoolOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      IREE::HAL::ConstantPoolOp op, llvm::ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    for (auto storageOp : op.getOps<IREE::HAL::ConstantStorageOp>()) {
      auto rodataName = (op.sym_name() + storageOp.sym_name()).str();
      auto rodataOp = rewriter.create<IREE::VM::RodataOp>(
          storageOp.getLoc(), rodataName, storageOp.value());
      rodataOp.setPrivate();
    }
    rewriter.eraseOp(op);
    return success();
  }
};

class ConstantStorageLookupOpConversion
    : public OpConversionPattern<IREE::HAL::ConstantStorageLookupOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      IREE::HAL::ConstantStorageLookupOp op, llvm::ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // I don't like this, but I can't figure out what to do.
    // Matches the logic above.
    auto rodataName =
        (op.constant().getRootReference() + op.constant().getLeafReference())
            .str();
    rewriter.replaceOpWithNewOp<IREE::VM::ConstRefRodataOp>(op, rodataName);
    return success();
  }
};

}  // namespace

void populateHALConstantToVMPatterns(MLIRContext *context,
                                     SymbolTable &importSymbols,
                                     TypeConverter &typeConverter,
                                     OwningRewritePatternList &patterns) {
  patterns.insert<ConstantPoolOpConversion, ConstantStorageLookupOpConversion>(
      typeConverter, context);
}

}  // namespace iree_compiler
}  // namespace mlir

// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string>

#include "iree/compiler/Dialect/HAL/Conversion/HALToVM/ConvertHALToVM.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/Dialect/VM/Conversion/ImportUtils.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace {

class RemoveExecutableOpConversion
    : public OpConversionPattern<IREE::HAL::ExecutableOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      IREE::HAL::ExecutableOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

class ExecutableCreateOpConversion
    : public OpConversionPattern<IREE::HAL::ExecutableCreateOp> {
 public:
  ExecutableCreateOpConversion(MLIRContext *context, SymbolTable &importSymbols,
                               TypeConverter &typeConverter,
                               StringRef importName)
      : OpConversionPattern(context) {
    importOp = importSymbols.lookup<IREE::VM::ImportOp>(importName);
    assert(importOp);
  }

  LogicalResult matchAndRewrite(
      IREE::HAL::ExecutableCreateOp createOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = createOp.getLoc();

    // Materialize vm.rodata for the binary.
    auto executableBinaryOp =
        SymbolTable::lookupNearestSymbolFrom<IREE::HAL::ExecutableBinaryOp>(
            createOp, createOp.executable_target());
    auto executableOp = executableBinaryOp.getOperation()
                            ->getParentOfType<IREE::HAL::ExecutableOp>();
    auto insertPoint = rewriter.saveInsertionPoint();
    rewriter.setInsertionPoint(rewriter.getInsertionBlock()->getParentOp());
    std::string rodataName =
        (executableOp.getName() + "_" + executableBinaryOp.getName()).str();
    std::replace(rodataName.begin(), rodataName.end(), '-', '_');
    auto rodataOp = rewriter.create<IREE::VM::RodataOp>(
        executableBinaryOp.getLoc(), rodataName, executableBinaryOp.data());
    rodataOp.setPrivate();
    if (executableBinaryOp.mime_type().hasValue()) {
      rodataOp.mime_typeAttr(executableBinaryOp.mime_typeAttr());
    }
    // TODO(benvanik): should these be page aligned? memcpy fastpath is fine for
    // now.
    rodataOp.alignmentAttr(rewriter.getI64IntegerAttr(16));
    rewriter.restoreInsertionPoint(insertPoint);

    auto executableFormatString = detail::rewriteAttrToOperands(
        createOp.getLoc(), executableBinaryOp.formatAttr(),
        importOp.getType().getInput(1), rewriter);
    assert(executableFormatString.hasValue() &&
           executableFormatString.getValue().size() == 1);

    SmallVector<int16_t, 4> segmentSizes = {
        /*device=*/-1,
        /*executable_format=*/-1,
        /*executable_data=*/-1,
        /*executable_layouts=*/
        static_cast<int16_t>(llvm::size(adaptor.layouts())),
    };
    SmallVector<Value, 8> callOperands = {
        adaptor.device(),
        executableFormatString.getValue().front(),
        rewriter.createOrFold<IREE::VM::ConstRefRodataOp>(loc, rodataOp),
    };
    callOperands.append(adaptor.layouts().begin(), adaptor.layouts().end());

    auto importType = importOp.getType();
    auto callOp = rewriter.replaceOpWithNewOp<IREE::VM::CallVariadicOp>(
        createOp, SymbolRefAttr::get(importOp), importType.getResults(),
        segmentSizes, importType.getInputs(), callOperands);
    copyImportAttrs(importOp, callOp);

    return success();
  }

 private:
  mutable IREE::VM::ImportOp importOp;
};  // namespace

}  // namespace

void populateHALExecutableToVMPatterns(MLIRContext *context,
                                       SymbolTable &importSymbols,
                                       TypeConverter &typeConverter,
                                       RewritePatternSet &patterns) {
  // hal.executables are not needed after conversion as we extract their
  // contents during conversion of the ops that use them.
  patterns.insert<RemoveExecutableOpConversion>(context);

  patterns.insert<ExecutableCreateOpConversion>(
      context, importSymbols, typeConverter, "hal.executable.create");

  patterns.insert<VMImportOpConversion<IREE::HAL::DescriptorSetLayoutCreateOp>>(
      context, importSymbols, typeConverter,
      "hal.descriptor_set_layout.create");
  patterns.insert<VMImportOpConversion<IREE::HAL::ExecutableLayoutCreateOp>>(
      context, importSymbols, typeConverter, "hal.executable_layout.create");
}

}  // namespace iree_compiler
}  // namespace mlir

// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VMVX/IR/VMVXDialect.h"

#include "iree/compiler/Dialect/VM/Conversion/ConversionDialectInterface.h"
#include "iree/compiler/Dialect/VMVX/Conversion/VMVXToVM/ConvertVMVXToVM.h"
#include "iree/compiler/Dialect/VMVX/IR/VMVXOps.h"
#include "iree/compiler/Dialect/VMVX/IR/VMVXTypes.h"
#include "iree/compiler/Dialect/VMVX/vmvx.imports.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Parser/Parser.h"

namespace mlir::iree_compiler::IREE::VMVX {

namespace {

class VMVXToVMConversionInterface : public VMConversionDialectInterface {
public:
  using VMConversionDialectInterface::VMConversionDialectInterface;

  OwningOpRef<mlir::ModuleOp> parseVMImportModule() const override {
    return mlir::parseSourceString<mlir::ModuleOp>(
        StringRef(iree_vmvx_imports_create()->data,
                  iree_vmvx_imports_create()->size),
        getDialect()->getContext());
  }

  void
  populateVMConversionPatterns(SymbolTable &importSymbols,
                               RewritePatternSet &patterns,
                               ConversionTarget &conversionTarget,
                               TypeConverter &typeConverter) const override {
    conversionTarget.addIllegalDialect<IREE::VMVX::VMVXDialect>();
    populateVMVXToVMPatterns(getDialect()->getContext(), conversionTarget,
                             typeConverter, importSymbols, patterns);
  }
};

} // namespace

VMVXDialect::VMVXDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<VMVXDialect>()) {
  addInterfaces<VMVXToVMConversionInterface>();

#define GET_OP_LIST
  addOperations<
#include "iree/compiler/Dialect/VMVX/IR/VMVXOps.cpp.inc"
      >();
}

} // namespace mlir::iree_compiler::IREE::VMVX

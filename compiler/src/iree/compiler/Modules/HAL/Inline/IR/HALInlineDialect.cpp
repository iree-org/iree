// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Modules/HAL/Inline/IR/HALInlineDialect.h"

#include "iree/compiler/Dialect/VM/Conversion/ConversionDialectInterface.h"
#include "iree/compiler/Modules/HAL/Inline/Conversion/HALInlineToVM/Patterns.h"
#include "iree/compiler/Modules/HAL/Inline/IR/HALInlineOps.h"
#include "iree/compiler/Modules/HAL/Inline/hal_inline.imports.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Parser/Parser.h"

namespace mlir::iree_compiler::IREE::HAL {
namespace Inline {

namespace {

class HALInlineToVMConversionInterface : public VMConversionDialectInterface {
public:
  using VMConversionDialectInterface::VMConversionDialectInterface;

  OwningOpRef<mlir::ModuleOp> parseVMImportModule() const override {
    return mlir::parseSourceString<mlir::ModuleOp>(
        StringRef(iree_hal_inline_imports_create()->data,
                  iree_hal_inline_imports_create()->size),
        getDialect()->getContext());
  }

  void
  populateVMConversionPatterns(SymbolTable &importSymbols,
                               RewritePatternSet &patterns,
                               ConversionTarget &conversionTarget,
                               TypeConverter &typeConverter) const override {
    conversionTarget.addIllegalDialect<IREE::HAL::Inline::HALInlineDialect>();
    populateHALInlineToVMPatterns(getDialect()->getContext(), conversionTarget,
                                  typeConverter, importSymbols, patterns);
  }
};

} // namespace

HALInlineDialect::HALInlineDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<HALInlineDialect>()) {
  addInterfaces<HALInlineToVMConversionInterface>();

#define GET_OP_LIST
  addOperations<
#include "iree/compiler/Modules/HAL/Inline/IR/HALInlineOps.cpp.inc"
      >();
}

} // namespace Inline
} // namespace mlir::iree_compiler::IREE::HAL

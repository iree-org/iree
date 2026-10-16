// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Modules/HAL/Loader/IR/HALLoaderDialect.h"

#include "iree/compiler/Dialect/VM/Conversion/ConversionDialectInterface.h"
#include "iree/compiler/Modules/HAL/Loader/Conversion/HALLoaderToVM/Patterns.h"
#include "iree/compiler/Modules/HAL/Loader/IR/HALLoaderOps.h"
#include "iree/compiler/Modules/HAL/Loader/hal_loader.imports.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/InliningUtils.h"

namespace mlir::iree_compiler::IREE::HAL::Loader {

namespace {

class HALLoaderToVMConversionInterface : public VMConversionDialectInterface {
public:
  using VMConversionDialectInterface::VMConversionDialectInterface;

  OwningOpRef<mlir::ModuleOp> parseVMImportModule() const override {
    return mlir::parseSourceString<mlir::ModuleOp>(
        StringRef(iree_hal_loader_imports_create()->data,
                  iree_hal_loader_imports_create()->size),
        getDialect()->getContext());
  }

  void
  populateVMConversionPatterns(SymbolTable &importSymbols,
                               RewritePatternSet &patterns,
                               ConversionTarget &conversionTarget,
                               TypeConverter &typeConverter) const override {
    conversionTarget.addIllegalDialect<IREE::HAL::Loader::HALLoaderDialect>();
    populateHALLoaderToVMPatterns(getDialect()->getContext(), conversionTarget,
                                  typeConverter, importSymbols, patterns);
  }
};

struct HALLoaderInlinerInterface : DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    return true;
  }

  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       IRMapping &valueMapping) const final {
    return true;
  }

  bool isLegalToInline(Operation *op, Region *dest, bool wouldBeCloned,
                       IRMapping &valueMapping) const final {
    return true;
  }
};

} // namespace

HALLoaderDialect::HALLoaderDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<HALLoaderDialect>()) {
  addInterfaces<HALLoaderToVMConversionInterface, HALLoaderInlinerInterface>();

#define GET_OP_LIST
  addOperations<
#include "iree/compiler/Modules/HAL/Loader/IR/HALLoaderOps.cpp.inc"
      >();
}

} // namespace mlir::iree_compiler::IREE::HAL::Loader

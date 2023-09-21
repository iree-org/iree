// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Modules/IO/Parameters/IR/IOParametersDialect.h"

#include "iree/compiler/Dialect/HAL/Conversion/ConversionDialectInterface.h"
#include "iree/compiler/Dialect/VM/Conversion/ConversionDialectInterface.h"
#include "iree/compiler/Modules/IO/Parameters/Conversion/ParamsToVM/Patterns.h"
#include "iree/compiler/Modules/IO/Parameters/Conversion/StreamToParams/Patterns.h"
#include "iree/compiler/Modules/IO/Parameters/IR/IOParametersOps.h"
#include "iree/compiler/Modules/IO/Parameters/io_parameters.imports.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/InliningUtils.h"

namespace mlir::iree_compiler::IREE::IO::Parameters {

namespace {

// Used to control inlining behavior.
struct IOParametersInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    // Sure!
    return true;
  }
  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       IRMapping &valueMapping) const final {
    // Sure!
    return true;
  }

  bool isLegalToInline(Operation *op, Region *dest, bool wouldBeCloned,
                       IRMapping &valueMapping) const final {
    // Sure!
    return true;
  }
};

class StreamToIOParametersConversionInterface
    : public HALConversionDialectInterface {
public:
  using HALConversionDialectInterface::HALConversionDialectInterface;
  void setupConversionTarget(ConversionTarget &target,
                             RewritePatternSet &patterns,
                             TypeConverter &typeConverter) const override {
    populateStreamToIOParametersPatterns(getDialect()->getContext(), target,
                                         typeConverter, patterns);
  }
};

class IOParametersToVMConversionInterface
    : public VMConversionDialectInterface {
public:
  using VMConversionDialectInterface::VMConversionDialectInterface;

  OwningOpRef<mlir::ModuleOp> parseVMImportModule() const override {
    return mlir::parseSourceString<mlir::ModuleOp>(
        StringRef(iree_io_parameters_imports_create()->data,
                  iree_io_parameters_imports_create()->size),
        getDialect()->getContext());
  }

  void
  populateVMConversionPatterns(SymbolTable &importSymbols,
                               RewritePatternSet &patterns,
                               ConversionTarget &conversionTarget,
                               TypeConverter &typeConverter) const override {
    conversionTarget
        .addIllegalDialect<IREE::IO::Parameters::IOParametersDialect>();
    populateIOParametersToVMPatterns(getDialect()->getContext(),
                                     conversionTarget, typeConverter,
                                     importSymbols, patterns);
  }
};

} // namespace

IOParametersDialect::IOParametersDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context,
              TypeID::get<IOParametersDialect>()) {
  addInterfaces<IOParametersInlinerInterface>();
  addInterfaces<StreamToIOParametersConversionInterface>();
  addInterfaces<IOParametersToVMConversionInterface>();

#define GET_OP_LIST
  addOperations<
#include "iree/compiler/Modules/IO/Parameters/IR/IOParametersOps.cpp.inc"
      >();
}

} // namespace mlir::iree_compiler::IREE::IO::Parameters

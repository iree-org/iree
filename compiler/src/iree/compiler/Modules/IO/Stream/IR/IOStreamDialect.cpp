// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Modules/IO/Stream/IR/IOStreamDialect.h"

#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/VM/Conversion/ConversionDialectInterface.h"
#include "iree/compiler/Modules/IO/Stream/Conversion/StreamToVM/Patterns.h"
#include "iree/compiler/Modules/IO/Stream/IR/IOStreamOps.h"
#include "iree/compiler/Modules/IO/Stream/io_stream.imports.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/InliningUtils.h"

// clang-format off: must be included after all LLVM/MLIR headers.
#define GET_TYPEDEF_CLASSES
#include "iree/compiler/Modules/IO/Stream/IR/IOStreamTypes.cpp.inc" // IWYU pragma: keep
// clang-format on

namespace mlir::iree_compiler::IREE::IO::Stream {

namespace {

// Used to control inlining behavior.
struct IOStreamInlinerInterface : public DialectInlinerInterface {
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

class IOStreamToVMConversionInterface : public VMConversionDialectInterface {
public:
  using VMConversionDialectInterface::VMConversionDialectInterface;

  OwningOpRef<mlir::ModuleOp> parseVMImportModule() const override {
    return mlir::parseSourceString<mlir::ModuleOp>(
        StringRef(iree_io_stream_imports_create()->data,
                  iree_io_stream_imports_create()->size),
        getDialect()->getContext());
  }

  void
  populateVMConversionPatterns(SymbolTable &importSymbols,
                               RewritePatternSet &patterns,
                               ConversionTarget &conversionTarget,
                               TypeConverter &typeConverter) const override {
    conversionTarget.addIllegalDialect<IREE::IO::Stream::IOStreamDialect>();
    populateIOStreamToVMPatterns(getDialect()->getContext(), conversionTarget,
                                 typeConverter, importSymbols, patterns);
  }
};

} // namespace

IOStreamDialect::IOStreamDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<IOStreamDialect>()) {
  context->loadDialect<IREE::Util::UtilDialect>();

#define GET_TYPEDEF_LIST
  addTypes<
#include "iree/compiler/Modules/IO/Stream/IR/IOStreamTypes.cpp.inc"
      >();

#define GET_OP_LIST
  addOperations<
#include "iree/compiler/Modules/IO/Stream/IR/IOStreamOps.cpp.inc"
      >();

  addInterfaces<IOStreamInlinerInterface>();
  addInterfaces<IOStreamToVMConversionInterface>();
}

Type IOStreamDialect::parseType(DialectAsmParser &parser) const {
  StringRef mnemonic;
  Type type;
  OptionalParseResult parseResult =
      generatedTypeParser(parser, &mnemonic, type);
  if (parseResult.has_value())
    return type;
  parser.emitError(parser.getCurrentLocation())
      << "unknown Flow type: " << mnemonic;
  return {};
}

void IOStreamDialect::printType(Type type, DialectAsmPrinter &os) const {
  if (succeeded(generatedTypePrinter(type, os)))
    return;
}

} // namespace mlir::iree_compiler::IREE::IO::Stream

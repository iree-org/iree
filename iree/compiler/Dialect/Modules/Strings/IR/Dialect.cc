// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Modules/Strings/IR/Dialect.h"

#include "iree/compiler/Dialect/HAL/Conversion/ConversionDialectInterface.h"
#include "iree/compiler/Dialect/HAL/Conversion/ConversionTarget.h"
#include "iree/compiler/Dialect/Modules/Strings/Conversion/ConversionPatterns.h"
#include "iree/compiler/Dialect/Modules/Strings/IR/Ops.h"
#include "iree/compiler/Dialect/Modules/Strings/strings.imports.h"
#include "iree/compiler/Dialect/VM/Conversion/ConversionDialectInterface.h"
#include "llvm/ADT/StringSwitch.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/InliningUtils.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Strings {

namespace {

struct StringsInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    // Sure!
    return true;
  }

  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       BlockAndValueMapping &valueMapping) const final {
    // Sure!
    return true;
  }

  bool isLegalToInline(Operation *op, Region *dest, bool wouldBeCloned,
                       BlockAndValueMapping &valueMapping) const final {
    // Sure!
    return true;
  }
};

class StringsToVMConversionInterface : public VMConversionDialectInterface {
 public:
  using VMConversionDialectInterface::VMConversionDialectInterface;

  OwningModuleRef parseVMImportModule() const override {
    return mlir::parseSourceString(
        StringRef(iree_strings_imports_create()->data,
                  iree_strings_imports_create()->size),
        getDialect()->getContext());
  }

  void populateVMConversionPatterns(
      SymbolTable &importSymbols, OwningRewritePatternList &patterns,
      TypeConverter &typeConverter) const override {
    populateStringsToVMPatterns(getDialect()->getContext(), importSymbols,
                                patterns, typeConverter);
  }
};

class StringsToHALConversionInterface : public HALConversionDialectInterface {
 public:
  using HALConversionDialectInterface::HALConversionDialectInterface;

  void setupConversionTarget(ConversionTarget &target,
                             OwningRewritePatternList &patterns,
                             TypeConverter &typeConverter) const override {
    populateStringsToHALPatterns(getDialect()->getContext(), patterns,
                                 typeConverter);
  };
};

}  // namespace

StringsDialect::StringsDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<StringsDialect>()) {
  addInterfaces<StringsToVMConversionInterface>();
  addInterfaces<StringsToHALConversionInterface>();
  addInterfaces<StringsInlinerInterface>();

  addTypes<StringType, StringTensorType>();

#define GET_OP_LIST
  addOperations<
#include "iree/compiler/Dialect/Modules/Strings/IR/Ops.cc.inc"
      >();
}

Type StringsDialect::parseType(DialectAsmParser &parser) const {
  StringRef typeName;
  if (failed(parser.parseKeyword(&typeName))) return {};
  auto type = llvm::StringSwitch<Type>(typeName)
                  .Case("string", StringType::get(getContext()))
                  .Case("string_tensor", StringTensorType::get(getContext()))
                  .Default(nullptr);

  if (!type) {
    parser.emitError(parser.getCurrentLocation())
        << "unknown type: " << typeName;
  }
  return type;
}

void StringsDialect::printType(Type type, DialectAsmPrinter &p) const {
  if (type.isa<StringType>()) {
    p << "string";
  } else if (type.isa<StringTensorType>()) {
    p << "string_tensor";
  } else {
    llvm_unreachable("unknown type");
  }
}

}  // namespace Strings
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

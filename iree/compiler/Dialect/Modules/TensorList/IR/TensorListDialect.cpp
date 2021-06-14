// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Modules/TensorList/IR/TensorListDialect.h"

#include "iree/compiler/Dialect/HAL/Conversion/ConversionDialectInterface.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/Modules/TensorList/Conversion/ConversionPatterns.h"
#include "iree/compiler/Dialect/Modules/TensorList/IR/TensorListOps.h"
#include "iree/compiler/Dialect/Modules/TensorList/tensorlist.imports.h"
#include "iree/compiler/Dialect/VM/Conversion/ConversionDialectInterface.h"
#include "llvm/ADT/StringSwitch.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/InliningUtils.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace TensorList {

namespace {

struct TensorListInlinerInterface : public DialectInlinerInterface {
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

class TensorListToHALConversionInterface
    : public HALConversionDialectInterface {
 public:
  using HALConversionDialectInterface::HALConversionDialectInterface;

  void setupConversionTarget(ConversionTarget &target,
                             OwningRewritePatternList &patterns,
                             TypeConverter &typeConverter) const override {
    populateTensorListToHALPatterns(getDialect()->getContext(), patterns,
                                    typeConverter);
  };
};

class TensorListToVMConversionInterface : public VMConversionDialectInterface {
 public:
  using VMConversionDialectInterface::VMConversionDialectInterface;

  OwningModuleRef parseVMImportModule() const override {
    return mlir::parseSourceString(
        StringRef(iree_tensorlist_imports_create()->data,
                  iree_tensorlist_imports_create()->size),
        getDialect()->getContext());
  }

  void populateVMConversionPatterns(
      SymbolTable &importSymbols, OwningRewritePatternList &patterns,
      TypeConverter &typeConverter) const override {
    populateTensorListToVMPatterns(getDialect()->getContext(), importSymbols,
                                   patterns, typeConverter);
  }
};

}  // namespace

TensorListDialect::TensorListDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context,
              TypeID::get<TensorListDialect>()) {
  addInterfaces<TensorListToHALConversionInterface>();
  addInterfaces<TensorListToVMConversionInterface>();
  addInterfaces<TensorListInlinerInterface>();

  addTypes<TensorListType>();

#define GET_OP_LIST
  addOperations<
#include "iree/compiler/Dialect/Modules/TensorList/IR/TensorListOps.cpp.inc"
      >();
}

Type TensorListDialect::parseType(DialectAsmParser &parser) const {
  StringRef typeName;
  if (failed(parser.parseKeyword(&typeName))) return {};
  auto type = llvm::StringSwitch<Type>(typeName)
                  .Case("list", TensorListType::get(getContext()))
                  .Default(nullptr);
  if (!type) {
    parser.emitError(parser.getCurrentLocation())
        << "unknown type: " << typeName;
  }
  return type;
}

void TensorListDialect::printType(Type type, DialectAsmPrinter &p) const {
  if (type.isa<TensorListType>()) {
    p << "list";
  } else {
    llvm_unreachable("unknown type");
  }
}

}  // namespace TensorList
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

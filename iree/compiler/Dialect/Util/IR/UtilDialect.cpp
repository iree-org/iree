// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"

#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Parser.h"
#include "mlir/Transforms/InliningUtils.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Util {

// Used to control inlining behavior.
struct IREEInlinerInterface : public DialectInlinerInterface {
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

UtilDialect::UtilDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<UtilDialect>()) {
  addInterfaces<IREEInlinerInterface>();
  registerTypes();
#define GET_OP_LIST
  addOperations<
#include "iree/compiler/Dialect/Util/IR/UtilOps.cpp.inc"
      >();
}

Type UtilDialect::parseType(DialectAsmParser &parser) const {
  Location loc = parser.getEncodedSourceLoc(parser.getNameLoc());
  llvm::StringRef spec = parser.getFullSymbolSpec();
  if (spec == "variant") {
    return IREE::Util::VariantType::get(getContext());
  } else if (spec.consume_front("ptr")) {
    if (!spec.consume_front("<") || !spec.consume_back(">")) {
      parser.emitError(parser.getCurrentLocation())
          << "malformed ptr type '" << parser.getFullSymbolSpec() << "'";
      return Type();
    }
    auto variableType = mlir::parseType(spec, getContext());
    if (!variableType) {
      parser.emitError(parser.getCurrentLocation())
          << "invalid ptr object type specification: '"
          << parser.getFullSymbolSpec() << "'";
      return Type();
    }
    return IREE::Util::PtrType::getChecked(variableType, loc);
  } else if (spec == "byte_buffer") {
    return IREE::Util::ByteBufferType::get(getContext());
  } else if (spec == "mutable_byte_buffer") {
    return IREE::Util::MutableByteBufferType::get(getContext());
  } else if (spec.consume_front("list")) {
    if (!spec.consume_front("<") || !spec.consume_back(">")) {
      parser.emitError(parser.getCurrentLocation())
          << "malformed list type '" << parser.getFullSymbolSpec() << "'";
      return Type();
    }
    Type elementType;
    if (spec == "?") {
      elementType = IREE::Util::VariantType::get(getContext());
    } else {
      elementType = mlir::parseType(spec, getContext());
    }
    if (!elementType) {
      parser.emitError(parser.getCurrentLocation())
          << "invalid list element type specification: '"
          << parser.getFullSymbolSpec() << "'";
      return Type();
    }
    return IREE::Util::ListType::getChecked(elementType, loc);
  }
  emitError(loc, "unknown IREE type: ") << spec;
  return Type();
}

void UtilDialect::printType(Type type, DialectAsmPrinter &os) const {
  if (type.isa<IREE::Util::VariantType>()) {
    os << "variant";
  } else if (auto ptrType = type.dyn_cast<IREE::Util::PtrType>()) {
    os << "ptr<" << ptrType.getTargetType() << ">";
  } else if (type.isa<IREE::Util::ByteBufferType>()) {
    os << "byte_buffer";
  } else if (type.isa<IREE::Util::MutableByteBufferType>()) {
    os << "mutable_byte_buffer";
  } else if (auto listType = type.dyn_cast<IREE::Util::ListType>()) {
    os << "list<";
    if (listType.getElementType().isa<IREE::Util::VariantType>()) {
      os << "?";
    } else {
      os << listType.getElementType();
    }
    os << ">";
  } else {
    llvm_unreachable("unhandled IREE type");
  }
}

}  // namespace Util
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

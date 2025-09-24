// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/FlowTypes.h"

#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"

// clang-format off: must be included after all LLVM/MLIR headers.
#define GET_ATTRDEF_CLASSES
#include "iree/compiler/Dialect/Flow/IR/FlowAttrs.cpp.inc" // IWYU pragma: keep
#include "iree/compiler/Dialect/Flow/IR/FlowEnums.cpp.inc" // IWYU pragma: keep
#define GET_TYPEDEF_CLASSES
#include "iree/compiler/Dialect/Flow/IR/FlowTypes.cpp.inc" // IWYU pragma: keep
// clang-format on

namespace mlir::iree_compiler::IREE::Flow {

//===----------------------------------------------------------------------===//
// Dialect registration
//===----------------------------------------------------------------------===//

#include "iree/compiler/Dialect/Flow/IR/FlowOpInterfaces.cpp.inc" // IWYU pragma: keep
#include "iree/compiler/Dialect/Flow/IR/FlowTypeInterfaces.cpp.inc" // IWYU pragma: keep

void FlowDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "iree/compiler/Dialect/Flow/IR/FlowAttrs.cpp.inc" // IWYU pragma: keep
      >();
}

void FlowDialect::registerTypes() {
  // addTypes<DispatchTensorType>();
  addTypes<
#define GET_TYPEDEF_LIST
#include "iree/compiler/Dialect/Flow/IR/FlowTypes.cpp.inc" // IWYU pragma: keep
      >();
}

//===----------------------------------------------------------------------===//
// Type printing and parsing
//===----------------------------------------------------------------------===//

Type FlowDialect::parseType(DialectAsmParser &parser) const {
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

void FlowDialect::printType(Type type, DialectAsmPrinter &p) const {
  if (failed(generatedTypePrinter(type, p))) {
    assert(false && "unknown Flow type");
  }
}

std::optional<IREE::Flow::CollectiveElementType>
convertToFlowCollectiveElementType(Type type) {
  if (isa<FloatType>(type)) {
    if (type.isF16()) {
      return IREE::Flow::CollectiveElementType::Float16;
    }
    if (type.isBF16()) {
      return IREE::Flow::CollectiveElementType::BFloat16;
    }
    if (type.isF32()) {
      return IREE::Flow::CollectiveElementType::Float32;
    }
    if (type.isF64()) {
      return IREE::Flow::CollectiveElementType::Float64;
    }
  } else if (isa<IntegerType>(type)) {
    if (type.isInteger(8)) {
      if (type.isSignedInteger()) {
        return IREE::Flow::CollectiveElementType::Sint8;
      }
      return IREE::Flow::CollectiveElementType::Uint8;
    }
    if (type.isInteger(16)) {
      if (type.isSignedInteger()) {
        return IREE::Flow::CollectiveElementType::Sint16;
      }
      return IREE::Flow::CollectiveElementType::Uint16;
    }
    if (type.isInteger(32)) {
      if (type.isSignedInteger()) {
        return IREE::Flow::CollectiveElementType::Sint32;
      }
      return IREE::Flow::CollectiveElementType::Uint32;
    }
    if (type.isInteger(64)) {
      if (type.isSignedInteger()) {
        return IREE::Flow::CollectiveElementType::Sint64;
      }
      return IREE::Flow::CollectiveElementType::Uint64;
    }
  }

  return std::nullopt;
}

IREE::Flow::CollectiveElementTypeAttr
getCollectiveElementTypeAttr(RankedTensorType type) {
  std::optional<IREE::Flow::CollectiveElementType> collectiveElemType =
      convertToFlowCollectiveElementType(type.getElementType());
  if (!collectiveElemType) {
    return IREE::Flow::CollectiveElementTypeAttr();
  }
  return IREE::Flow::CollectiveElementTypeAttr::get(type.getContext(),
                                                    *collectiveElemType);
}

//===----------------------------------------------------------------------===//
// custom<ParameterReference>($scope, $key)
//===----------------------------------------------------------------------===//

ParseResult parseParameterReference(AsmParser &parser, StringAttr &scopeAttr,
                                    StringAttr &keyAttr) {
  auto builder = parser.getBuilder();
  StringAttr firstAttr;
  if (failed(parser.parseCustomAttributeWithFallback(firstAttr,
                                                     builder.getNoneType()))) {
    return failure();
  }
  if (failed(parser.parseOptionalColon())) {
    keyAttr = firstAttr;
    return success();
  }
  scopeAttr = firstAttr;
  if (failed(parser.parseColon()) ||
      failed(parser.parseCustomAttributeWithFallback(keyAttr,
                                                     builder.getNoneType()))) {
    return failure();
  }
  return success();
}

void printParameterReference(AsmPrinter &p, StringAttr scopeAttr,
                             StringAttr keyAttr) {
  if (scopeAttr) {
    p << "\"" << scopeAttr.getValue() << "\"";
    p << "::";
  }
  p << "\"" << keyAttr.getValue() << "\"";
}

//===----------------------------------------------------------------------===//
// #flow.parameter.named<...>
//===----------------------------------------------------------------------===//

int64_t NamedParameterAttr::getStorageSize() const {
  if (auto configAttr = getConfig()) {
    if (auto lengthAttr = configAttr.getAs<IntegerAttr>("length")) {
      return lengthAttr.getInt();
    }
  }
  if (auto shapedType = llvm::dyn_cast<ShapedType>(getType())) {
    return IREE::Util::getRoundedPhysicalStorageSize(shapedType);
  } else {
    return IREE::Util::getTypePhysicalStorageBitWidth(getType());
  }
}

} // namespace mlir::iree_compiler::IREE::Flow

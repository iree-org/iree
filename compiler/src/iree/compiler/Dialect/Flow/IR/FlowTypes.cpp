// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/FlowTypes.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
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
// !flow.dispatch.tensor
//===----------------------------------------------------------------------===//

// static
DispatchTensorType DispatchTensorType::get(TensorAccess access,
                                           ArrayRef<int64_t> shape,
                                           Type elementType,
                                           Attribute encoding) {
  return Base::get(elementType.getContext(), static_cast<uint32_t>(access),
                   RankedTensorType::get(shape, elementType, encoding));
}

// static
DispatchTensorType DispatchTensorType::get(TensorAccess access,
                                           Type boundType) {
  return Base::get(boundType.getContext(), static_cast<uint32_t>(access),
                   boundType);
}

TensorAccess DispatchTensorType::getAccess() const {
  return static_cast<TensorAccess>(static_cast<ImplType *>(impl)->access);
}

Type DispatchTensorType::getBoundType() const {
  return static_cast<Type>(static_cast<ImplType *>(impl)->boundType);
}

Type DispatchTensorType::getBoundElementType() const {
  Type boundType = getBoundType();
  if (boundType.isIntOrFloat()) {
    return boundType;
  }
  return llvm::cast<RankedTensorType>(boundType).getElementType();
}

unsigned DispatchTensorType::getBoundElementTypeBitWidth() const {
  return getBoundElementType().getIntOrFloatBitWidth();
}

int64_t DispatchTensorType::getNumElements() const {
  assert(hasStaticShape() && "cannot get element count of dynamic shaped type");
  auto shape = getShape();
  int64_t num = 1;
  for (auto dim : shape)
    num *= dim;
  return num;
}

int64_t DispatchTensorType::getRank() const {
  Type boundType = getBoundType();
  if (boundType.isIntOrIndexOrFloat()) {
    return 0;
  }
  return llvm::cast<RankedTensorType>(boundType).getRank();
}

bool DispatchTensorType::hasRank() const { return true; }

int64_t DispatchTensorType::getDimSize(unsigned idx) const {
  assert(idx < getRank() && "invalid index for shaped type");
  return getShape()[idx];
}

bool DispatchTensorType::isDynamicDim(unsigned idx) const {
  assert(idx < getRank() && "invalid index for shaped type");
  return ShapedType::isDynamic(getShape()[idx]);
}

unsigned DispatchTensorType::getDynamicDimIndex(unsigned index) const {
  assert(index < getRank() && "invalid index");
  assert(ShapedType::isDynamic(getDimSize(index)) && "invalid index");
  return llvm::count_if(getShape().take_front(index), ShapedType::isDynamic);
}

ArrayRef<int64_t> DispatchTensorType::getShape() const {
  Type boundType = getBoundType();
  if (boundType.isIntOrIndexOrFloat()) {
    return {};
  }
  return llvm::cast<RankedTensorType>(boundType).getShape();
}

int64_t DispatchTensorType::getNumDynamicDims() const {
  return llvm::count_if(getShape(), ShapedType::isDynamic);
}

bool DispatchTensorType::hasStaticShape() const {
  return hasRank() && llvm::none_of(getShape(), ShapedType::isDynamic);
}

bool DispatchTensorType::hasStaticShape(ArrayRef<int64_t> shape) const {
  return hasStaticShape() && getShape() == shape;
}

LogicalResult
DispatchTensorType::verify(function_ref<InFlightDiagnostic()> emitError,
                           uint32_t access, Type boundType) {
  if (!boundType.isIntOrFloat() && !llvm::isa<RankedTensorType>(boundType)) {
    return emitError() << "unhandled bounded type in dispatch. Must by int, "
                          "float or ranked tensor type";
  }
  return success();
}

template <typename T>
static T parseShapedType(AsmParser &parser) {
  StringRef accessStr;
  Type boundType;
  if (failed(parser.parseLess()) || failed(parser.parseKeyword(&accessStr)) ||
      failed(parser.parseColon()) || failed(parser.parseType(boundType)) ||
      failed(parser.parseGreater())) {
    return {};
  }
  auto access = llvm::StringSwitch<TensorAccess>(accessStr)
                    .Case("readonly", TensorAccess::ReadOnly)
                    .Case("readwrite", TensorAccess::ReadWrite)
                    .Case("writeonly", TensorAccess::WriteOnly)
                    .Default(TensorAccess::ReadOnly);
  return T::get(access, boundType);
}

static void printShapedType(DispatchTensorType &type, AsmPrinter &p) {
  switch (type.getAccess()) {
  case TensorAccess::ReadOnly:
    p << "readonly";
    break;
  case TensorAccess::ReadWrite:
    p << "readwrite";
    break;
  case TensorAccess::WriteOnly:
    p << "writeonly";
    break;
  default:
    assert(false && "unhandled access");
  }
  p << ":" << type.getBoundType();
}

// static
DispatchTensorType DispatchTensorType::parse(AsmParser &parser) {
  return parseShapedType<DispatchTensorType>(parser);
}

void printType(DispatchTensorType &type, DialectAsmPrinter &p) {
  p << "dispatch.tensor<";
  printShapedType(type, p);
  p << '>';
}

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
  addTypes<DispatchTensorType>();
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
  if (mnemonic == "dispatch.tensor") {
    return DispatchTensorType::parse(parser);
  }
  parser.emitError(parser.getCurrentLocation())
      << "unknown Flow type: " << mnemonic;
  return {};
}

void FlowDialect::printType(Type type, DialectAsmPrinter &p) const {
  if (auto inputType = llvm::dyn_cast<DispatchTensorType>(type)) {
    IREE::Flow::printType(inputType, p);
  } else if (failed(generatedTypePrinter(type, p))) {
    assert(false && "unknown Flow type");
  }
}

} // namespace mlir::iree_compiler::IREE::Flow

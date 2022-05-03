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
#include "iree/compiler/Dialect/Flow/IR/FlowAttrs.cpp.inc"  // IWYU pragma: keep
#include "iree/compiler/Dialect/Flow/IR/FlowEnums.cpp.inc"  // IWYU pragma: keep
#define GET_TYPEDEF_CLASSES
#include "iree/compiler/Dialect/Flow/IR/FlowTypes.cpp.inc"  // IWYU pragma: keep
// clang-format on

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

//===----------------------------------------------------------------------===//
// !flow.dispatch.tensor
//===----------------------------------------------------------------------===//

// static
DispatchTensorType DispatchTensorType::get(TensorAccess access,
                                           ArrayRef<int64_t> shape,
                                           Type elementType) {
  return Base::get(elementType.getContext(), static_cast<uint32_t>(access),
                   shape, elementType);
}

// static
DispatchTensorType DispatchTensorType::get(TensorAccess access,
                                           TensorType tensorType) {
  return DispatchTensorType::get(access, tensorType.getShape(),
                                 tensorType.getElementType());
}

TensorAccess DispatchTensorType::getAccess() const {
  return static_cast<TensorAccess>(static_cast<ImplType *>(impl)->access);
}

Type DispatchTensorType::getElementType() const {
  return static_cast<ImplType *>(impl)->elementType;
}

unsigned DispatchTensorType::getElementTypeBitWidth() const {
  return getElementType().getIntOrFloatBitWidth();
}

int64_t DispatchTensorType::getNumElements() const {
  assert(hasStaticShape() && "cannot get element count of dynamic shaped type");
  auto shape = getShape();
  int64_t num = 1;
  for (auto dim : shape) num *= dim;
  return num;
}

int64_t DispatchTensorType::getRank() const { return getShape().size(); }

bool DispatchTensorType::hasRank() const { return true; }

int64_t DispatchTensorType::getDimSize(unsigned idx) const {
  assert(idx < getRank() && "invalid index for shaped type");
  return getShape()[idx];
}

bool DispatchTensorType::isDynamicDim(unsigned idx) const {
  assert(idx < getRank() && "invalid index for shaped type");
  return isDynamic(getShape()[idx]);
}

unsigned DispatchTensorType::getDynamicDimIndex(unsigned index) const {
  assert(index < getRank() && "invalid index");
  assert(DispatchTensorType::isDynamic(getDimSize(index)) && "invalid index");
  return llvm::count_if(getShape().take_front(index),
                        DispatchTensorType::isDynamic);
}

ArrayRef<int64_t> DispatchTensorType::getShape() const {
  return static_cast<ImplType *>(impl)->getShape();
}

int64_t DispatchTensorType::getNumDynamicDims() const {
  return llvm::count_if(getShape(), isDynamic);
}

bool DispatchTensorType::hasStaticShape() const {
  return hasRank() && llvm::none_of(getShape(), isDynamic);
}

bool DispatchTensorType::hasStaticShape(ArrayRef<int64_t> shape) const {
  return hasStaticShape() && getShape() == shape;
}

LogicalResult DispatchTensorType::verify(
    function_ref<InFlightDiagnostic()> emitError, uint32_t access,
    ArrayRef<int64_t> shape, Type elementType) {
  if (!isValidElementType(elementType)) {
    return emitError() << "dispatch tensor elements must be int or float type";
  }
  if (any_of(shape, [](int64_t i) { return i < -1; })) {
    return emitError()
           << "dispatch tensor dimensions must be positive if defined";
  }
  return success();
}

template <typename T>
static T parseShapedType(AsmParser &parser) {
  StringRef accessStr;
  SmallVector<int64_t, 4> shape;
  Type elementType;
  if (failed(parser.parseLess()) || failed(parser.parseKeyword(&accessStr)) ||
      failed(parser.parseColon()) ||
      failed(parser.parseDimensionList(shape, /*allowDynamic=*/true)) ||
      failed(parser.parseType(elementType)) || failed(parser.parseGreater())) {
    return {};
  }
  auto access = llvm::StringSwitch<TensorAccess>(accessStr)
                    .Case("readonly", TensorAccess::ReadOnly)
                    .Case("readwrite", TensorAccess::ReadWrite)
                    .Case("writeonly", TensorAccess::WriteOnly)
                    .Default(TensorAccess::ReadOnly);
  return T::get(access, shape, elementType);
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
  p << ":";
  for (int64_t dim : type.getShape()) {
    if (ShapedType::isDynamic(dim)) {
      p << '?';
    } else {
      p << dim;
    }
    p << 'x';
  }
  p << type.getElementType();
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

#include "iree/compiler/Dialect/Flow/IR/FlowOpInterfaces.cpp.inc"  // IWYU pragma: keep
#include "iree/compiler/Dialect/Flow/IR/FlowTypeInterfaces.cpp.inc"  // IWYU pragma: keep

void FlowDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "iree/compiler/Dialect/Flow/IR/FlowAttrs.cpp.inc"  // IWYU pragma: keep
      >();
}

void FlowDialect::registerTypes() {
  addTypes<DispatchTensorType>();
  addTypes<
#define GET_TYPEDEF_LIST
#include "iree/compiler/Dialect/Flow/IR/FlowTypes.cpp.inc"  // IWYU pragma: keep
      >();
}

//===----------------------------------------------------------------------===//
// Type printing and parsing
//===----------------------------------------------------------------------===//

Type FlowDialect::parseType(DialectAsmParser &parser) const {
  StringRef mnemonic;
  if (failed(parser.parseKeyword(&mnemonic))) return {};
  Type type;
  OptionalParseResult parseResult = generatedTypeParser(parser, mnemonic, type);
  if (parseResult.hasValue()) return type;
  if (mnemonic == "dispatch.tensor") {
    return DispatchTensorType::parse(parser);
  }
  parser.emitError(parser.getCurrentLocation())
      << "unknown Flow type: " << mnemonic;
  return {};
}

void FlowDialect::printType(Type type, DialectAsmPrinter &p) const {
  if (auto inputType = type.dyn_cast<DispatchTensorType>()) {
    IREE::Flow::printType(inputType, p);
  } else if (failed(generatedTypePrinter(type, p))) {
    assert(false && "unknown Flow type");
  }
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

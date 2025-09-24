// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/TensorExt/IR/TensorExtTypes.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"

namespace mlir::iree_compiler::IREE::TensorExt {

//===----------------------------------------------------------------------===//
// !iree_tensor_ext.dispatch.tensor
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

bool DispatchTensorType::doesSliceSpanWholeTensor(
    ValueRange dispatchTypeDims, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, ArrayRef<OpFoldResult> strides) const {
  // All offsets must be zero.
  if (!llvm::all_of(offsets, isZeroInteger)) {
    return false;
  }

  // All the sizes must match the entire target size.
  SmallVector<int64_t> staticSizes;
  SmallVector<Value> dynamicSizes;
  dispatchIndexOpFoldResults(sizes, dynamicSizes, staticSizes);
  if (staticSizes != getShape() ||
      llvm::any_of(llvm::zip_equal(dynamicSizes, dispatchTypeDims),
                   [](std::tuple<Value, Value> en) {
                     return std::get<0>(en) != std::get<1>(en);
                   })) {
    return false;
  }

  // All the strides must be 1.
  if (!llvm::all_of(strides, isOneInteger)) {
    return false;
  }
  return true;
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
// Type printing and parsing
//===----------------------------------------------------------------------===//

Type IREETensorExtDialect::parseType(DialectAsmParser &parser) const {
  StringRef mnemonic;
  if (parser.parseKeyword(&mnemonic))
    return {};
  if (mnemonic == "dispatch.tensor")
    return DispatchTensorType::parse(parser);
  parser.emitError(parser.getCurrentLocation())
      << "unknown TensorExt type: " << mnemonic;
  return {};
}

void IREETensorExtDialect::printType(Type type, DialectAsmPrinter &p) const {
  if (auto inputType = llvm::dyn_cast<DispatchTensorType>(type)) {
    IREE::TensorExt::printType(inputType, p);
  } else {
    assert(false && "unknown Flow type");
  }
}

} // namespace mlir::iree_compiler::IREE::TensorExt

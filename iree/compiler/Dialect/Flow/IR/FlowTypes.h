// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_FLOW_IR_FLOWTYPES_H_
#define IREE_COMPILER_DIALECT_FLOW_IR_FLOWTYPES_H_

#include "iree/compiler/Dialect/IREE/IR/IREETypes.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeTypes.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"

// clang-format off: must be included after all LLVM/MLIR headers.
#include "iree/compiler/Dialect/Flow/IR/FlowEnums.h.inc"  // IWYU pragma: export
// clang-format on

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

//===----------------------------------------------------------------------===//
// Object types
//===----------------------------------------------------------------------===//

namespace detail {
struct DispatchTensorTypeStorage;
}  // namespace detail

enum class TensorAccess : uint32_t {
  ReadOnly,
  ReadWrite,
  WriteOnly,
};

// Blatantly ripped from ShapedType, because the closed type system means that
// we can't extend it and reuse all of this.
class DispatchTensorType
    : public Type::TypeBase<DispatchTensorType, Type,
                            detail::DispatchTensorTypeStorage> {
 public:
  using ImplType = detail::DispatchTensorTypeStorage;

  static constexpr int64_t kDynamicSize = -1;

  using Base::Base;

  /// Get or create a new DispatchTensorType of the provided shape and
  /// element type. Assumes the arguments define a well-formed
  /// DispatchTensorType.
  static DispatchTensorType get(TensorAccess access, ArrayRef<int64_t> shape,
                                Type elementType);

  static DispatchTensorType get(TensorAccess access, TensorType tensorType);

  static DispatchTensorType parse(DialectAsmParser &parser);

  /// Returns the allowed operations the tensor.
  TensorAccess getAccess() const;

  /// Return the element type.
  Type getElementType() const;

  /// If an element type is an integer or a float, return its width. Otherwise,
  /// abort.
  unsigned getElementTypeBitWidth() const;

  /// If it has static shape, return the number of elements. Otherwise, abort.
  int64_t getNumElements() const;

  /// If this is a ranked type, return the rank. Otherwise, abort.
  int64_t getRank() const;

  /// Whether or not this is a ranked type. Memrefs, vectors and ranked tensors
  /// have a rank, while unranked tensors do not.
  bool hasRank() const;

  /// If this is a ranked type, return the shape. Otherwise, abort.
  ArrayRef<int64_t> getShape() const;

  /// If this is unranked type or any dimension has unknown size (<0), it
  /// doesn't have static shape. If all dimensions have known size (>= 0), it
  /// has static shape.
  bool hasStaticShape() const;

  /// If this has a static shape and the shape is equal to `shape` return true.
  bool hasStaticShape(ArrayRef<int64_t> shape) const;

  /// If this is a ranked type, return the number of dimensions with dynamic
  /// size. Otherwise, abort.
  int64_t getNumDynamicDims() const;

  /// If this is ranked type, return the size of the specified dimension.
  /// Otherwise, abort.
  int64_t getDimSize(unsigned idx) const;

  /// Returns true if this dimension has a dynamic size (for ranked types);
  /// aborts for unranked types.
  bool isDynamicDim(unsigned idx) const;

  /// Returns the position of the dynamic dimension relative to just the dynamic
  /// dimensions, given its `index` within the shape.
  unsigned getDynamicDimIndex(unsigned index) const;

  /// Whether the given dimension size indicates a dynamic dimension.
  static constexpr bool isDynamic(int64_t dSize) {
    return dSize == kDynamicSize;
  }

  /// Verify the construction of a tensor type.
  static LogicalResult verify(function_ref<InFlightDiagnostic()> emitError,
                              uint32_t access, ArrayRef<int64_t> shape,
                              Type elementType);

  /// Returns true of the given type can be used as an element of a vector type.
  /// In particular, vectors can consist of integer or float primitives.
  static bool isValidElementType(Type t) {
    return TensorType::isValidElementType(t);
  }

  TensorType asTensorType() const {
    return RankedTensorType::get(getShape(), getElementType());
  }

  Shape::RankedShapeType asRankedShapeType() const {
    return Shape::RankedShapeType::get(getShape(), getContext());
  }
};

void printType(DispatchTensorType &type, DialectAsmPrinter &p);

namespace detail {

struct DispatchTensorTypeStorage : public TypeStorage {
  DispatchTensorTypeStorage(uint32_t access, unsigned shapeSize, Type elementTy,
                            const int64_t *shapeElements)
      : access(access),
        shapeElements(shapeElements),
        shapeSize(shapeSize),
        elementType(elementTy) {}

  /// The hash key used for uniquing.
  using KeyTy = std::tuple<uint32_t, ArrayRef<int64_t>, Type>;
  bool operator==(const KeyTy &key) const {
    return key == KeyTy(access, getShape(), elementType);
  }

  /// Construction.
  static DispatchTensorTypeStorage *construct(TypeStorageAllocator &allocator,
                                              const KeyTy &key) {
    // Copy the shape into the bump pointer.
    ArrayRef<int64_t> shape = allocator.copyInto(std::get<1>(key));

    // Initialize the memory using placement new.
    return new (allocator.allocate<DispatchTensorTypeStorage>())
        DispatchTensorTypeStorage(std::get<0>(key), shape.size(),
                                  std::get<2>(key), shape.data());
  }

  ArrayRef<int64_t> getShape() const {
    return ArrayRef<int64_t>(shapeElements, shapeSize);
  }

  uint32_t access;
  const int64_t *shapeElements;
  unsigned shapeSize;
  Type elementType;
};

}  // namespace detail

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_FLOW_IR_FLOWTYPES_H_

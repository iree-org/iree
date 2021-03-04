// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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

// Order matters.
#include "iree/compiler/Dialect/Flow/IR/FlowEnums.h.inc"

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

// Blatantly ripped from ShapedType, because the closed type system means that
// we can't extend it and reuse all of this.
class DispatchTensorType : public Type {
 public:
  using ImplType = detail::DispatchTensorTypeStorage;

  static constexpr int64_t kDynamicSize = -1;

  using Type::Type;

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

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(Type type);

  /// Whether the given dimension size indicates a dynamic dimension.
  static constexpr bool isDynamic(int64_t dSize) {
    return dSize == kDynamicSize;
  }

  /// Verify the construction of a vector type.
  static LogicalResult verify(function_ref<InFlightDiagnostic()> emitError,
                              ArrayRef<int64_t> shape, Type elementType);

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

class DispatchInputType
    : public Type::TypeBase<DispatchInputType, DispatchTensorType,
                            detail::DispatchTensorTypeStorage> {
 public:
  using Base::Base;

  /// Get or create a new DispatchInputType of the provided shape and element
  /// type. Assumes the arguments define a well-formed DispatchInputType.
  static DispatchInputType get(ArrayRef<int64_t> shape, Type elementType);

  /// Get or create a new DispatchInputType of the provided shape and element
  /// type declared at the given, potentially unknown, location.  If the
  /// DispatchInputType defined by the arguments would be ill-formed, emit
  /// errors and return nullptr-wrapping type.
  static DispatchInputType getChecked(ArrayRef<int64_t> shape, Type elementType,
                                      Location location);
  static DispatchInputType getChecked(
      function_ref<InFlightDiagnostic()> emitError, ArrayRef<int64_t> shape,
      Type elementType) {
    return Base::getChecked(emitError, elementType.getContext(), shape,
                            elementType);
  }

  static DispatchInputType get(TensorType tensorType);

  static DispatchInputType parse(DialectAsmParser &parser);
};

void printType(DispatchInputType &type, DialectAsmPrinter &p);

class DispatchOutputType
    : public Type::TypeBase<DispatchOutputType, DispatchTensorType,
                            detail::DispatchTensorTypeStorage> {
 public:
  using Base::Base;

  /// Get or create a new DispatchOutputType of the provided shape and element
  /// type. Assumes the arguments define a well-formed DispatchOutputType.
  static DispatchOutputType get(ArrayRef<int64_t> shape, Type elementType);

  /// Get or create a new DispatchOutputType of the provided shape and element
  /// type declared at the given, potentially unknown, location.  If the
  /// DispatchOutputType defined by the arguments would be ill-formed, emit
  /// errors and return nullptr-wrapping type.
  static DispatchOutputType getChecked(ArrayRef<int64_t> shape,
                                       Type elementType, Location location);
  static DispatchOutputType getChecked(
      function_ref<InFlightDiagnostic()> emitError, ArrayRef<int64_t> shape,
      Type elementType) {
    return Base::getChecked(emitError, elementType.getContext(), shape,
                            elementType);
  }

  static DispatchOutputType get(TensorType tensorType);

  static DispatchOutputType parse(DialectAsmParser &parser);
};

void printType(DispatchOutputType &type, DialectAsmPrinter &p);

inline bool DispatchTensorType::classof(Type type) {
  return type.isa<DispatchInputType, DispatchOutputType>();
}

namespace detail {

struct DispatchTensorTypeStorage : public TypeStorage {
  DispatchTensorTypeStorage(unsigned shapeSize, Type elementTy,
                            const int64_t *shapeElements)
      : shapeElements(shapeElements),
        shapeSize(shapeSize),
        elementType(elementTy) {}

  /// The hash key used for uniquing.
  using KeyTy = std::pair<ArrayRef<int64_t>, Type>;
  bool operator==(const KeyTy &key) const {
    return key == KeyTy(getShape(), elementType);
  }

  /// Construction.
  static DispatchTensorTypeStorage *construct(TypeStorageAllocator &allocator,
                                              const KeyTy &key) {
    // Copy the shape into the bump pointer.
    ArrayRef<int64_t> shape = allocator.copyInto(key.first);

    // Initialize the memory using placement new.
    return new (allocator.allocate<DispatchTensorTypeStorage>())
        DispatchTensorTypeStorage(shape.size(), key.second, shape.data());
  }

  ArrayRef<int64_t> getShape() const {
    return ArrayRef<int64_t>(shapeElements, shapeSize);
  }

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

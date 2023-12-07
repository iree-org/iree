// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_FLOW_IR_FLOWTYPES_H_
#define IREE_COMPILER_DIALECT_FLOW_IR_FLOWTYPES_H_

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"

// clang-format off: must be included after all LLVM/MLIR headers.
#include "iree/compiler/Dialect/Flow/IR/FlowEnums.h.inc" // IWYU pragma: export
// clang-format on

namespace mlir::iree_compiler::IREE::Flow {

#include "iree/compiler/Dialect/Flow/IR/FlowOpInterfaces.h.inc" // IWYU pragma: export
#include "iree/compiler/Dialect/Flow/IR/FlowTypeInterfaces.h.inc" // IWYU pragma: export

//===----------------------------------------------------------------------===//
// Object types
//===----------------------------------------------------------------------===//

namespace detail {
struct DispatchTensorTypeStorage;
} // namespace detail

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

  using Base::Base;

  static constexpr StringLiteral name = "flow.dispatch_tensor";

  /// Get or create a new DispatchTensorType of the provided shape and
  /// element type. Assumes the arguments define a well-formed
  /// DispatchTensorType.
  static DispatchTensorType get(TensorAccess access, ArrayRef<int64_t> shape,
                                Type elementType, Attribute encoding = {});

  static DispatchTensorType get(TensorAccess access, Type type);

  static DispatchTensorType parse(AsmParser &parser);

  /// Returns the allowed operations the tensor.
  TensorAccess getAccess() const;

  /// Returns the bounded type.
  Type getBoundType() const;

  /// Return the element type of the bounded type.
  Type getBoundElementType() const;

  /// If an element type is an integer or a float, return its width. Otherwise,
  /// abort.
  unsigned getBoundElementTypeBitWidth() const;

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

  /// Verify the construction of a tensor type.
  static LogicalResult verify(function_ref<InFlightDiagnostic()> emitError,
                              uint32_t access, Type boundType);

  /// Returns true of the given type can be used as an element of a vector type.
  /// In particular, vectors can consist of integer or float primitives.
  static bool isValidElementType(Type t) {
    return TensorType::isValidElementType(t);
  }

  RankedTensorType asRankedTensorType() const {
    Type boundType = getBoundType();
    if (boundType.isIntOrIndexOrFloat()) {
      return RankedTensorType::get({}, boundType);
    }
    return boundType.cast<RankedTensorType>();
  }
};

void printType(DispatchTensorType &type, DialectAsmPrinter &p);

namespace detail {

struct DispatchTensorTypeStorage : public TypeStorage {
  DispatchTensorTypeStorage(uint32_t access, Type boundType)
      : access(access), boundType(boundType) {}

  /// The hash key used for uniquing.
  using KeyTy = std::tuple<uint32_t, Type>;
  bool operator==(const KeyTy &key) const {
    return key == KeyTy(access, boundType);
  }

  /// Construction.
  static DispatchTensorTypeStorage *construct(TypeStorageAllocator &allocator,
                                              const KeyTy &key) {
    // Initialize the memory using placement new.
    return new (allocator.allocate<DispatchTensorTypeStorage>())
        DispatchTensorTypeStorage(std::get<0>(key), std::get<1>(key));
  }

  uint32_t access;
  Type boundType;
};

} // namespace detail

} // namespace mlir::iree_compiler::IREE::Flow

// clang-format off: must be included after all LLVM/MLIR headers.
#define GET_ATTRDEF_CLASSES
#include "iree/compiler/Dialect/Flow/IR/FlowAttrs.h.inc" // IWYU pragma: keep
#define GET_TYPEDEF_CLASSES
#include "iree/compiler/Dialect/Flow/IR/FlowTypes.h.inc" // IWYU pragma: keep
// clang-format on

#endif // IREE_COMPILER_DIALECT_FLOW_IR_FLOWTYPES_H_

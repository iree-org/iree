// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_IREE_IR_IREETYPES_H_
#define IREE_COMPILER_DIALECT_IREE_IR_IREETYPES_H_

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {

class TiedOpInterface;

namespace detail {

struct ListTypeStorage;
struct PtrTypeStorage;
struct RankedShapeTypeStorage;

}  // namespace detail

// Status code table mapping to iree::StatusCode in the runtime.
enum class StatusCode : int32_t {
  Ok = 0,
  Cancelled = 1,
  Unknown = 2,
  InvalidArgument = 3,
  DeadlineExceeded = 4,
  NotFound = 5,
  AlreadyExists = 6,
  PermissionDenied = 7,
  ResourceExhausted = 8,
  FailedPrecondition = 9,
  Aborted = 10,
  OutOfRange = 11,
  Unimplemented = 12,
  Internal = 13,
  Unavailable = 14,
  DataLoss = 15,
  Unauthenticated = 16,
  DoNotUseReservedForFutureExpansionUseDefaultInSwitchInstead_ = 20
};

/// Placeholder for a variant type (`?`).
class VariantType : public Type::TypeBase<VariantType, Type, TypeStorage> {
 public:
  using Base::Base;
};

/// A list containing an optional element type.
class ListType
    : public Type::TypeBase<ListType, Type, detail::ListTypeStorage> {
 public:
  using Base::Base;

  /// Returns true if the given type can be wrapped in a list.
  static bool isCompatible(Type type);

  /// Returns true if |from| can be implicitly cast to |to| as part of a list
  /// access operation. Example: tensor<*xf32> -> tensor<4xf32>.
  static bool canImplicitlyCast(Type from, Type to);

  /// Gets or creates a ListType with the provided element type.
  static ListType get(Type elementType);

  /// Gets or creates a ListType with the provided element type.
  /// This emits an error at the specified location and returns null if the
  /// element type isn't supported.
  static ListType getChecked(Type elementType, Location location);
  static ListType getChecked(function_ref<InFlightDiagnostic()> emitError,
                             Type elementType);

  /// Verifies construction of a type with the given object.
  static LogicalResult verify(function_ref<InFlightDiagnostic()> emitError,
                              Type elementType) {
    if (!isCompatible(elementType)) {
      return emitError() << "invalid element type for a list: " << elementType;
    }
    return success();
  }

  Type getElementType();
};

/// Base for typed pointer-like references.
class PtrType : public Type::TypeBase<PtrType, Type, detail::PtrTypeStorage> {
 public:
  static PtrType get(Type targetType);
  static PtrType getChecked(Type targetType, Location location);
  static PtrType getChecked(function_ref<InFlightDiagnostic()> emitError,
                            Type targetType);

  using Base::Base;

  Type getTargetType();
};

/// A buffer of constant mapped memory.
class ByteBufferType
    : public Type::TypeBase<ByteBufferType, Type, TypeStorage> {
 public:
  using Base::Base;
};

/// A buffer of read-write memory.
class MutableByteBufferType
    : public Type::TypeBase<MutableByteBufferType, Type, TypeStorage> {
 public:
  using Base::Base;
};

namespace detail {
llvm::Optional<unsigned> getTiedResultOperandIndex(Operation *op,
                                                   unsigned resultIndex);
void setTiedResultOperandIndex(Operation *op, unsigned resultIndex,
                               llvm::Optional<unsigned> operandIndex);
SmallVector<int64_t, 4> getTiedResultOperandIndices(Operation *op);
LogicalResult verifyTiedOp(TiedOpInterface tiedOp);
}  // namespace detail

// Resets or removes the indices in |tiedOperandIndices| based on the given
// exclusion lists.
void excludeTiedOperandAndResultIndices(
    ArrayRef<unsigned> excludedOperandIndices,
    ArrayRef<unsigned> excludedResultIndices,
    SmallVector<int64_t, 4> &tiedOperandIndices);

#include "iree/compiler/Dialect/IREE/IR/IREEOpInterfaces.h.inc"  // IWYU pragma: export

}  // namespace IREE

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_IREE_IR_IREETYPES_H_

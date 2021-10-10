// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_UTIL_IR_UTILTYPES_H_
#define IREE_COMPILER_DIALECT_UTIL_IR_UTILTYPES_H_

#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Endian.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SubElementInterfaces.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Util {

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

struct ValueAccess {
  bool isRead : 1;
  bool isWrite : 1;
  bool isDiscard : 1;
  bool isNone() const { return !isRead && !isWrite && !isDiscard; }
  bool isReadOnly() const { return isRead && !isWrite && !isDiscard; }
  ValueAccess() : isRead(false), isWrite(false), isDiscard(false) {}
  ValueAccess(bool isRead, bool isWrite, bool isDiscard)
      : isRead(isRead), isWrite(isWrite), isDiscard(isDiscard) {}
  static ValueAccess None() { return ValueAccess(false, false, false); }
  static ValueAccess ReadOnly() { return ValueAccess(true, false, false); }
  static ValueAccess ReadWrite() { return ValueAccess(true, true, false); }
  static ValueAccess WriteOnly() { return ValueAccess(false, true, false); }
  static ValueAccess DiscardWrite() { return ValueAccess(false, true, true); }
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
class PtrType : public Type::TypeBase<PtrType, Type, detail::PtrTypeStorage,
                                      mlir::SubElementTypeInterface::Trait> {
 public:
  static PtrType get(Type targetType);
  static PtrType getChecked(Type targetType, Location location);
  static PtrType getChecked(function_ref<InFlightDiagnostic()> emitError,
                            Type targetType);

  using Base::Base;

  Type getTargetType() const;

  void walkImmediateSubElements(
      llvm::function_ref<void(mlir::Attribute)> walkAttrsFn,
      llvm::function_ref<void(mlir::Type)> walkTypesFn) const {
    walkTypesFn(getTargetType());
  }
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
bool isOperandTied(Operation *tiedOp, unsigned operandIndex);
SmallVector<Value> getOperandTiedResults(Operation *op, unsigned operandIndex);
LogicalResult verifyTiedOp(TiedOpInterface tiedOp);

}  // namespace detail

// Resets or removes the indices in |tiedOperandIndices| based on the given
// exclusion lists.
void excludeTiedOperandAndResultIndices(
    ArrayRef<unsigned> excludedOperandIndices,
    ArrayRef<unsigned> excludedResultIndices,
    SmallVector<int64_t, 4> &tiedOperandIndices);

// Walks the SSA use-def chain to find the dynamic dimensions of the value.
// Returns None if the shape cannot be found or if it is defined after
// |forOp|.
Optional<ValueRange> findDynamicDims(Value shapedValue, Operation *forOp);

// Returns the dynamic dimensions for the value at |idx|.
ValueRange findVariadicDynamicDims(unsigned idx, ValueRange values,
                                   ValueRange dynamicDims);

// Aligns |value| to |alignment|, rounding up if needed.
static inline uint64_t align(uint64_t value, uint64_t alignment) {
  return (value + (alignment - 1)) & ~(alignment - 1);
}
static inline uint64_t align(uint64_t value, const APInt &alignment) {
  return align(value, alignment.getZExtValue());
}

// Aligns |value| to |alignment|, rounding up if needed.
Value align(Location loc, Value value, int64_t alignment, OpBuilder &builder);

}  // namespace Util
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#include "iree/compiler/Dialect/Util/IR/UtilAttrInterfaces.h.inc"  // IWYU pragma: export
#include "iree/compiler/Dialect/Util/IR/UtilOpInterfaces.h.inc"  // IWYU pragma: export
#include "iree/compiler/Dialect/Util/IR/UtilTypeInterfaces.h.inc"  // IWYU pragma: export

// clang-format off: must be included after all LLVM/MLIR headers.
#define GET_ATTRDEF_CLASSES
#include "iree/compiler/Dialect/Util/IR/UtilAttrs.h.inc"  // IWYU pragma: keep
// clang-format on

#endif  // IREE_COMPILER_DIALECT_UTIL_IR_UTILTYPES_H_

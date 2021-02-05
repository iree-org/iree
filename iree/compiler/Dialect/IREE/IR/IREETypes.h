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

#ifndef IREE_COMPILER_DIALECT_IREE_IR_IREETYPES_H_
#define IREE_COMPILER_DIALECT_IREE_IR_IREETYPES_H_

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {

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

/// A list containing an optional element type.
class ListType
    : public Type::TypeBase<ListType, Type, detail::ListTypeStorage> {
 public:
  using Base::Base;

  /// Returns true if the given type can be wrapped in a list.
  static bool isCompatible(Type type);

  /// Gets or creates a ListType with the provided element type.
  static ListType get(Type elementType);

  /// Gets or creates a ListType with the provided element type.
  /// This emits an error at the specified location and returns null if the
  /// element type isn't supported.
  static ListType getChecked(Type elementType, Location location);

  /// Verifies construction of a type with the given object.
  static LogicalResult verifyConstructionInvariants(Location loc,
                                                    Type elementType) {
    if (!isCompatible(elementType)) {
      return emitError(loc)
             << "invalid element type for a list: " << elementType;
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

}  // namespace IREE

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_IREE_IR_IREETYPES_H_

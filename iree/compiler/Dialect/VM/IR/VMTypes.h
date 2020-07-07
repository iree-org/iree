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

#ifndef IREE_COMPILER_DIALECT_VM_IR_VMTYPES_H_
#define IREE_COMPILER_DIALECT_VM_IR_VMTYPES_H_

#include "iree/compiler/Dialect/IREE/IR/IREETypes.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"

// Order matters.
#include "iree/compiler/Dialect/VM/IR/VMEnums.h.inc"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace VM {

namespace detail {
struct ListTypeStorage;
struct RefTypeStorage;
}  // namespace detail

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

  static bool kindof(unsigned kind) { return kind == TypeKind::List; }
};

/// An opaque ref object that comes from an external source.
class OpaqueType : public Type::TypeBase<OpaqueType, Type, TypeStorage> {
 public:
  using Base::Base;

  static bool kindof(unsigned kind) { return kind == TypeKind::Opaque; }

  static OpaqueType get(MLIRContext *context) {
    return Base::get(context, TypeKind::Opaque);
  }
};

/// A ref_ptr containing a reference to a ref-object-compatible type.
class RefType : public Type::TypeBase<RefType, Type, detail::RefTypeStorage> {
 public:
  using Base::Base;

  /// Returns true if the given type can be wrapped in a ref ptr.
  static bool isCompatible(Type type);

  /// Gets or creates a RefType with the provided target object type.
  static RefType get(Type objectType);

  /// Gets or creates a RefType with the provided target object type.
  /// This emits an error at the specified location and returns null if the
  /// object type isn't supported.
  static RefType getChecked(Type objectType, Location location);

  /// Verifies construction of a type with the given object.
  static LogicalResult verifyConstructionInvariants(Location loc,
                                                    Type objectType) {
    if (!isCompatible(objectType)) {
      return emitError(loc) << "invalid object type for a ref: " << objectType;
    }
    return success();
  }

  Type getObjectType();

  static bool kindof(unsigned kind) { return kind == TypeKind::Ref; }
};

}  // namespace VM
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_VM_IR_VMTYPES_H_

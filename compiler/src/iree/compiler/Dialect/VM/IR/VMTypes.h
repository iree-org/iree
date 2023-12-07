// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_VM_IR_VMTYPES_H_
#define IREE_COMPILER_DIALECT_VM_IR_VMTYPES_H_

#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"

// clang-format off: must be included after all LLVM/MLIR headers.
#define GET_ATTRDEF_CLASSES
#include "iree/compiler/Dialect/VM/IR/VMAttrs.h.inc" // IWYU pragma: export
#include "iree/compiler/Dialect/VM/IR/VMEnums.h.inc" // IWYU pragma: keep
// clang-format on

namespace mlir::iree_compiler::IREE::VM {

namespace detail {
struct ListTypeStorage;
struct RefTypeStorage;
} // namespace detail

/// A byte buffer.
class BufferType : public Type::TypeBase<BufferType, Type, TypeStorage> {
public:
  using Base::Base;

  static constexpr StringLiteral name = "vm.buffer";
};

/// A list containing an optional element type.
class ListType
    : public Type::TypeBase<ListType, Type, detail::ListTypeStorage> {
public:
  using Base::Base;

  static constexpr StringLiteral name = "vm.list";

  /// Returns true if the given type can be wrapped in a list.
  static bool isCompatible(Type type);

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

/// An opaque ref object that comes from an external source.
class OpaqueType : public Type::TypeBase<OpaqueType, Type, TypeStorage> {
public:
  using Base::Base;

  static constexpr StringLiteral name = "vm.opaque";
};

/// A ref<T> containing a reference to a ref-object-compatible type.
/// This models an iree_vm_ref_t intrusive reference counted object.
class RefType : public Type::TypeBase<RefType, Type, detail::RefTypeStorage> {
public:
  using Base::Base;

  static constexpr StringLiteral name = "vm.ref";

  /// Returns true if the given type can be wrapped in a ref ptr.
  static bool isCompatible(Type type);

  /// Gets or creates a RefType with the provided target object type.
  static RefType get(Type objectType);

  /// Gets or creates a RefType with the provided target object type.
  /// This emits an error at the specified location and returns null if the
  /// object type isn't supported.
  static RefType getChecked(Type objectType, Location location);
  static RefType getChecked(function_ref<InFlightDiagnostic()> emitError,
                            Type objectType);

  /// Verifies construction of a type with the given object.
  static LogicalResult verify(function_ref<InFlightDiagnostic()> emitError,
                              Type objectType) {
    if (!isCompatible(objectType)) {
      return emitError() << "invalid object type for a ref: " << objectType;
    }
    return success();
  }

  Type getObjectType();
};

} // namespace mlir::iree_compiler::IREE::VM

#endif // IREE_COMPILER_DIALECT_VM_IR_VMTYPES_H_

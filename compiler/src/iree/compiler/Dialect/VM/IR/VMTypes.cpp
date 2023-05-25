// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VM/IR/VMTypes.h"

#include "iree/compiler/Dialect/VM/IR/VMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/TypeSupport.h"

// clang-format off: must be included after all LLVM/MLIR headers.
#define GET_ATTRDEF_CLASSES
#include "iree/compiler/Dialect/VM/IR/VMAttrs.cpp.inc"  // IWYU pragma: keep
#include "iree/compiler/Dialect/VM/IR/VMEnums.cpp.inc"  // IWYU pragma: keep
// clang-format on

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace VM {

//===----------------------------------------------------------------------===//
// ListType
//===----------------------------------------------------------------------===//

namespace detail {

struct ListTypeStorage : public TypeStorage {
  ListTypeStorage(Type elementType) : elementType(elementType) {}

  /// The hash key used for uniquing.
  using KeyTy = Type;
  bool operator==(const KeyTy &key) const { return key == elementType; }

  static ListTypeStorage *construct(TypeStorageAllocator &allocator,
                                    const KeyTy &key) {
    // Initialize the memory using placement new.
    return new (allocator.allocate<ListTypeStorage>()) ListTypeStorage(key);
  }

  Type elementType;
};

}  // namespace detail

// static
bool ListType::isCompatible(Type type) {
  if (llvm::isa<OpaqueType>(type)) {
    // Allow all types (variant).
    return true;
  } else if (llvm::isa<RefType>(type)) {
    // Allow all ref types.
    return true;
  } else if (type.isIntOrFloat()) {
    // Allow all byte-aligned types.
    return (type.getIntOrFloatBitWidth() % 8) == 0;
  }
  // Disallow undefined types.
  return false;
}

ListType ListType::get(Type elementType) {
  return Base::get(elementType.getContext(), elementType);
}

ListType ListType::getChecked(Type elementType, Location location) {
  return Base::getChecked(location, elementType);
}

ListType ListType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                              Type elementType) {
  return Base::getChecked(emitError, elementType.getContext(), elementType);
}

Type ListType::getElementType() { return getImpl()->elementType; }

//===----------------------------------------------------------------------===//
// RefType
//===----------------------------------------------------------------------===//

namespace detail {

struct RefTypeStorage : public TypeStorage {
  RefTypeStorage(Type objectType) : objectType(llvm::cast<Type>(objectType)) {}

  /// The hash key used for uniquing.
  using KeyTy = Type;
  bool operator==(const KeyTy &key) const { return key == objectType; }

  static RefTypeStorage *construct(TypeStorageAllocator &allocator,
                                   const KeyTy &key) {
    // Initialize the memory using placement new.
    return new (allocator.allocate<RefTypeStorage>()) RefTypeStorage(key);
  }

  Type objectType;
};

}  // namespace detail

// static
bool RefType::isCompatible(Type type) {
  if (llvm::isa<RefType>(type)) {
    // Already a ref - don't double-wrap.
    return false;
  } else if (type.isSignlessIntOrIndexOrFloat() ||
             llvm::isa<ComplexType>(type)) {
    // Ignore known primitive types.
    return false;
  }
  // Assume all other types (user types, buffers, etc) can be wrapped.
  return true;
}

RefType RefType::get(Type objectType) {
  return Base::get(objectType.getContext(), objectType);
}

RefType RefType::getChecked(Type objectType, Location location) {
  return Base::getChecked(location, objectType);
}

RefType RefType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                            Type objectType) {
  return Base::getChecked(emitError, objectType.getContext(), objectType);
}

Type RefType::getObjectType() { return getImpl()->objectType; }

//===----------------------------------------------------------------------===//
// VMDialect
//===----------------------------------------------------------------------===//

void VMDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "iree/compiler/Dialect/VM/IR/VMAttrs.cpp.inc"  // IWYU pragma: keep
      >();
}
void VMDialect::registerTypes() {
  addTypes<IREE::VM::BufferType, IREE::VM::ListType, IREE::VM::OpaqueType,
           IREE::VM::RefType>();
}

//===----------------------------------------------------------------------===//
// Attribute printing and parsing
//===----------------------------------------------------------------------===//

Attribute VMDialect::parseAttribute(DialectAsmParser &parser, Type type) const {
  StringRef mnemonic;
  Attribute genAttr;
  OptionalParseResult parseResult =
      generatedAttributeParser(parser, &mnemonic, type, genAttr);
  if (parseResult.has_value()) return genAttr;
  parser.emitError(parser.getNameLoc())
      << "unknown HAL attribute: " << mnemonic;
  return {};
}

void VMDialect::printAttribute(Attribute attr, DialectAsmPrinter &p) const {
  TypeSwitch<Attribute>(attr).Default([&](Attribute) {
    if (failed(generatedAttributePrinter(attr, p))) {
      assert(false && "unhandled HAL attribute kind");
    }
  });
}

}  // namespace VM
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

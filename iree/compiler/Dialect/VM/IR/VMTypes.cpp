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
#include "iree/compiler/Dialect/VM/IR/VMEnums.cpp.inc"    // IWYU pragma: keep
#include "iree/compiler/Dialect/VM/IR/VMStructs.cpp.inc"  // IWYU pragma: keep
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
  if (type.isa<OpaqueType>()) {
    // Allow all types (variant).
    return true;
  } else if (type.isa<RefType>()) {
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
  RefTypeStorage(Type objectType) : objectType(objectType.cast<Type>()) {}

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
  if (type.isa<RefType>()) {
    // Already a ref - don't double-wrap.
    return false;
  } else if (type.isSignlessIntOrIndexOrFloat()) {
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
// Attribute printing and parsing
//===----------------------------------------------------------------------===//

Attribute OrdinalCountsAttr::parse(DialectAsmParser &p) {
  Type i32 = p.getBuilder().getIntegerType(32);
  IntegerAttr importFuncsAttr;
  IntegerAttr exportFuncsAttr;
  IntegerAttr internalFuncsAttr;
  IntegerAttr globalBytesAttr;
  IntegerAttr globalRefsAttr;
  IntegerAttr rodatasAttr;
  IntegerAttr rwdatasAttr;
  if (failed(p.parseLess()) || failed(p.parseKeyword("import_funcs")) ||
      failed(p.parseEqual()) ||
      failed(p.parseAttribute(importFuncsAttr, i32)) ||
      failed(p.parseComma()) || failed(p.parseKeyword("export_funcs")) ||
      failed(p.parseEqual()) ||
      failed(p.parseAttribute(exportFuncsAttr, i32)) ||
      failed(p.parseComma()) || failed(p.parseKeyword("internal_funcs")) ||
      failed(p.parseEqual()) ||
      failed(p.parseAttribute(internalFuncsAttr, i32)) ||
      failed(p.parseComma()) || failed(p.parseKeyword("global_bytes")) ||
      failed(p.parseEqual()) ||
      failed(p.parseAttribute(globalBytesAttr, i32)) ||
      failed(p.parseComma()) || failed(p.parseKeyword("global_refs")) ||
      failed(p.parseEqual()) || failed(p.parseAttribute(globalRefsAttr, i32)) ||
      failed(p.parseComma()) || failed(p.parseKeyword("rodatas")) ||
      failed(p.parseEqual()) || failed(p.parseAttribute(rodatasAttr, i32)) ||
      failed(p.parseComma()) || failed(p.parseKeyword("rwdatas")) ||
      failed(p.parseEqual()) || failed(p.parseAttribute(rwdatasAttr, i32)) ||
      failed(p.parseGreater())) {
    return {};
  }
  return get(importFuncsAttr, exportFuncsAttr, internalFuncsAttr,
             globalBytesAttr, globalRefsAttr, rodatasAttr, rwdatasAttr);
}

void OrdinalCountsAttr::print(DialectAsmPrinter &p) const {
  auto &os = p.getStream();
  os << getKindName() << "<";
  os << "import_funcs = " << import_funcs() << ", ";
  os << "export_funcs = " << export_funcs() << ", ";
  os << "internal_funcs = " << internal_funcs() << ", ";
  os << "global_bytes = " << global_bytes() << ", ";
  os << "global_refs = " << global_refs() << ", ";
  os << "rodatas = " << rodatas() << ", ";
  os << "rwdatas = " << rwdatas();
  os << ">";
}

//===----------------------------------------------------------------------===//
// VMDialect
//===----------------------------------------------------------------------===//

void VMDialect::registerAttributes() {
  addAttributes<IREE::VM::OrdinalCountsAttr>();
}
void VMDialect::registerTypes() {
  addTypes<IREE::VM::BufferType, IREE::VM::ListType, IREE::VM::OpaqueType,
           IREE::VM::RefType>();
}

}  // namespace VM
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

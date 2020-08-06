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

#include "iree/compiler/Dialect/VM/IR/VMTypes.h"

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/TypeSupport.h"

// Order matters:
#include "iree/compiler/Dialect/VM/IR/VMEnums.cpp.inc"

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
  if (type.isa<RefType>()) {
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
  return Base::get(elementType.getContext(), TypeKind::List, elementType);
}

ListType ListType::getChecked(Type elementType, Location location) {
  return Base::getChecked(location, TypeKind::List, elementType);
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
  return Base::get(objectType.getContext(), TypeKind::Ref, objectType);
}

RefType RefType::getChecked(Type objectType, Location location) {
  return Base::getChecked(location, TypeKind::Ref, objectType);
}

Type RefType::getObjectType() { return getImpl()->objectType; }

}  // namespace VM
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

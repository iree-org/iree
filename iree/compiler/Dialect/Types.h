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

#ifndef IREE_COMPILER_DIALECT_TYPES_H_
#define IREE_COMPILER_DIALECT_TYPES_H_

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {

namespace TypeKind {
enum Kind {
  // TODO(b/143787186): move back down to +0 when old dialects are removed.
  RefPtr = Type::FIRST_IREE_TYPE + 60,
  OpaqueRefObject,
  ByteBuffer,
  MutableByteBuffer,

  FIRST_HAL_TYPE = Type::FIRST_IREE_TYPE + 20,
  FIRST_SEQ_TYPE = Type::FIRST_IREE_TYPE + 40,
};
}  // namespace TypeKind

namespace HAL {
namespace TypeKind {
enum Kind {
  Allocator = IREE::TypeKind::FIRST_HAL_TYPE,
  Buffer,
  BufferView,
  CommandBuffer,
  DescriptorSet,
  DescriptorSetLayout,
  Device,
  Event,
  Executable,
  ExecutableCache,
  Fence,
  RingBuffer,
  Semaphore,
};
}  // namespace TypeKind
}  // namespace HAL

namespace SEQ {
namespace TypeKind {
enum Kind {
  Device = IREE::TypeKind::FIRST_SEQ_TYPE,
  Policy,
  Resource,
  Timeline,
};
}  // namespace TypeKind
}  // namespace SEQ

/// Base type for RefObject-derived types.
/// These can be wrapped in RefPtrType.
class RefObjectType : public Type {
 public:
  using ImplType = TypeStorage;
  using Type::Type;

  static bool classof(Type type) {
    // TODO(benvanik): figure out how to do a semi-open type system.
    return true;
  }
};

// TODO(benvanik): checked version with supported type kinds.
/// An opaque ref object that comes from an external source.
class OpaqueRefObjectType
    : public Type::TypeBase<OpaqueRefObjectType, RefObjectType> {
 public:
  using Base::Base;

  static bool kindof(unsigned kind) {
    return kind == TypeKind::OpaqueRefObject;
  }

  static OpaqueRefObjectType get(MLIRContext *context) {
    return Base::get(context, TypeKind::OpaqueRefObject);
  }
};

/// A buffer of constant mapped memory.
class ByteBufferType : public Type::TypeBase<ByteBufferType, RefObjectType> {
 public:
  using Base::Base;

  static bool kindof(unsigned kind) { return kind == TypeKind::ByteBuffer; }

  static ByteBufferType get(MLIRContext *context) {
    return Base::get(context, TypeKind::ByteBuffer);
  }
};

/// A buffer of read-write memory.
class MutableByteBufferType
    : public Type::TypeBase<MutableByteBufferType, RefObjectType> {
 public:
  using Base::Base;

  static bool kindof(unsigned kind) {
    return kind == TypeKind::MutableByteBuffer;
  }

  static MutableByteBufferType get(MLIRContext *context) {
    return Base::get(context, TypeKind::MutableByteBuffer);
  }
};

namespace detail {

struct RefPtrTypeStorage : public TypeStorage {
  RefPtrTypeStorage(Type objectType, unsigned subclassData = 0)
      : TypeStorage(subclassData),
        objectType(objectType.cast<RefObjectType>()) {}

  /// The hash key used for uniquing.
  using KeyTy = Type;
  bool operator==(const KeyTy &key) const { return key == objectType; }

  static RefPtrTypeStorage *construct(TypeStorageAllocator &allocator,
                                      const KeyTy &key) {
    // Initialize the memory using placement new.
    return new (allocator.allocate<RefPtrTypeStorage>()) RefPtrTypeStorage(key);
  }

  RefObjectType objectType;
};

}  // namespace detail

/// A ref_ptr containing a reference to a RefObjectType.
class RefPtrType
    : public Type::TypeBase<RefPtrType, Type, detail::RefPtrTypeStorage> {
 public:
  using Base::Base;

  /// Gets or creates a RefPtrType with the provided target object type.
  static RefPtrType get(RefObjectType objectType) {
    return Base::get(objectType.getContext(), TypeKind::RefPtr, objectType);
  }

  /// Gets or creates a RefPtrType with the provided target object type.
  /// This emits an error at the specified location and returns null if the
  /// object type isn't supported.
  static RefPtrType getChecked(Type objectType, Location location) {
    return Base::getChecked(location, objectType.getContext(), TypeKind::RefPtr,
                            objectType);
  }

  /// Verifies construction of a type with the given object.
  static LogicalResult verifyConstructionInvariants(
      llvm::Optional<Location> loc, MLIRContext *context, Type objectType) {
    if (!RefObjectType::classof(objectType)) {
      if (loc) {
        emitError(*loc) << "invalid object type for a ref_ptr: " << objectType;
      }
      return failure();
    }
    return success();
  }

  RefObjectType getObjectType() { return getImpl()->objectType; }

  static bool kindof(unsigned kind) { return kind == TypeKind::RefPtr; }
};

}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_TYPES_H_

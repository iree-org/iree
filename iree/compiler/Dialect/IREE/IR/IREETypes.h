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

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {

namespace detail {
struct PtrTypeStorage;
struct RefPtrTypeStorage;
struct RankedShapeTypeStorage;
}  // namespace detail

namespace TypeKind {
enum Kind {
  Ptr = Type::FIRST_IREE_TYPE,
  RefPtr,
  OpaqueRefObject,
  ByteBuffer,
  MutableByteBuffer,

  FIRST_HAL_TYPE = Type::FIRST_IREE_TYPE + 20,
  FIRST_SEQ_TYPE = Type::FIRST_IREE_TYPE + 40,
  FIRST_SHAPE_TYPE = Type::FIRST_IREE_TYPE + 60,
  FIRST_STRING_TYPE = Type::FIRST_IREE_TYPE + 80,
  FIRST_VMLA_TYPE = Type::FIRST_IREE_TYPE + 100,
  FIRST_TENSORLIST_TYPE = Type::FIRST_IREE_TYPE + 120,
  FIRST_TF_TENSORLIST_TYPE = Type::FIRST_IREE_TYPE + 140,
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
  ExecutableLayout,
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

namespace Shape {
namespace TypeKind {
enum Kind {
  RankedShape = IREE::TypeKind::FIRST_SHAPE_TYPE,
};
}  // namespace TypeKind
}  // namespace Shape

namespace Strings {
namespace TypeKind {
enum Kind {
  String = IREE::TypeKind::FIRST_STRING_TYPE,
};
}  // namespace TypeKind
}  // namespace Strings

namespace VMLA {
namespace TypeKind {
enum Kind {
  Buffer = IREE::TypeKind::FIRST_VMLA_TYPE,
};
}  // namespace TypeKind
}  // namespace VMLA

/// Base for typed pointer-like references.
class PtrType : public Type::TypeBase<PtrType, Type, detail::PtrTypeStorage> {
 public:
  static PtrType get(Type targetType);
  static PtrType getChecked(Type targetType, Location location);
  static bool kindof(unsigned kind) { return kind == TypeKind::Ptr; }

  using Base::Base;

  Type getTargetType();
};

/// An opaque ref object that comes from an external source.
class OpaqueType : public Type::TypeBase<OpaqueType, Type> {
 public:
  using Base::Base;

  static bool kindof(unsigned kind) {
    return kind == TypeKind::OpaqueRefObject;
  }

  static OpaqueType get(MLIRContext *context) {
    return Base::get(context, TypeKind::OpaqueRefObject);
  }
};

/// A buffer of constant mapped memory.
class ByteBufferType : public Type::TypeBase<ByteBufferType, Type> {
 public:
  using Base::Base;

  static bool kindof(unsigned kind) { return kind == TypeKind::ByteBuffer; }

  static ByteBufferType get(MLIRContext *context) {
    return Base::get(context, TypeKind::ByteBuffer);
  }
};

/// A buffer of read-write memory.
class MutableByteBufferType
    : public Type::TypeBase<MutableByteBufferType, Type> {
 public:
  using Base::Base;

  static bool kindof(unsigned kind) {
    return kind == TypeKind::MutableByteBuffer;
  }

  static MutableByteBufferType get(MLIRContext *context) {
    return Base::get(context, TypeKind::MutableByteBuffer);
  }
};

/// A ref_ptr containing a reference to a ref-object-compatible type.
class RefPtrType
    : public Type::TypeBase<RefPtrType, Type, detail::RefPtrTypeStorage> {
 public:
  using Base::Base;

  /// Returns true if the given type can be wrapped in a ref ptr.
  static bool isCompatible(Type type);

  /// Gets or creates a RefPtrType with the provided target object type.
  static RefPtrType get(Type objectType);

  /// Gets or creates a RefPtrType with the provided target object type.
  /// This emits an error at the specified location and returns null if the
  /// object type isn't supported.
  static RefPtrType getChecked(Type objectType, Location location);

  /// Verifies construction of a type with the given object.
  static LogicalResult verifyConstructionInvariants(
      llvm::Optional<Location> loc, MLIRContext *context, Type objectType) {
    if (!isCompatible(objectType)) {
      if (loc) {
        emitError(*loc) << "invalid object type for a ref_ptr: " << objectType;
      }
      return failure();
    }
    return success();
  }

  Type getObjectType();

  static bool kindof(unsigned kind) { return kind == TypeKind::RefPtr; }
};

}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_IREE_IR_IREETYPES_H_

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

namespace TypeKind {
enum Kind {
  // NOTE: The range is [FIRST_IREE_TYPE..FIRST_IREE_TYPE+255].
  Ptr = Type::FIRST_IREE_TYPE,
  ByteBuffer,
  MutableByteBuffer,

  FIRST_HAL_TYPE = Type::FIRST_IREE_TYPE + 20,
  FIRST_SEQ_TYPE = Type::FIRST_IREE_TYPE + 40,
  FIRST_SHAPE_TYPE = Type::FIRST_IREE_TYPE + 60,
  FIRST_STRING_TYPE = Type::FIRST_IREE_TYPE + 80,
  FIRST_VM_TYPE = Type::FIRST_IREE_TYPE + 100,
  FIRST_VMLA_TYPE = Type::FIRST_IREE_TYPE + 120,
  FIRST_TENSORLIST_TYPE = Type::FIRST_IREE_TYPE + 140,
  FIRST_TF_TENSORLIST_TYPE = Type::FIRST_IREE_TYPE + 160,
  FIRST_SEQUENCE_TYPE = Type::FIRST_IREE_TYPE + 180,
};
}  // namespace TypeKind

namespace Sequence {
namespace TypeKind {
enum Kind {
  Sequence = IREE::TypeKind::FIRST_SEQUENCE_TYPE,
};
}  // namespace TypeKind
}  // namespace Sequence

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

namespace Strings {
namespace TypeKind {
enum Kind {
  String = IREE::TypeKind::FIRST_STRING_TYPE,
  StringTensor,
};
}  // namespace TypeKind
}  // namespace Strings

namespace VM {
namespace TypeKind {
enum Kind {
  Ref = IREE::TypeKind::FIRST_VM_TYPE,
  Opaque,
  List,
};
}  // namespace TypeKind
}  // namespace VM

namespace VMLA {
namespace TypeKind {
enum Kind {
  Buffer = IREE::TypeKind::FIRST_VMLA_TYPE,
  Interface,
};
}  // namespace TypeKind
}  // namespace VMLA

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

  static ByteBufferType get(MLIRContext *context) {
    return Base::get(context, TypeKind::ByteBuffer);
  }
};

/// A buffer of read-write memory.
class MutableByteBufferType
    : public Type::TypeBase<MutableByteBufferType, Type, TypeStorage> {
 public:
  using Base::Base;

  static MutableByteBufferType get(MLIRContext *context) {
    return Base::get(context, TypeKind::MutableByteBuffer);
  }
};

}  // namespace IREE

namespace Shape {
namespace TypeKind {
enum Kind {
  RankedShape = IREE::TypeKind::FIRST_SHAPE_TYPE,
};
}  // namespace TypeKind
}  // namespace Shape

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_IREE_IR_IREETYPES_H_

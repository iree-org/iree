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

#ifndef IREE_COMPILER_DIALECT_HAL_IR_HALTYPES_H_
#define IREE_COMPILER_DIALECT_HAL_IR_HALTYPES_H_

#include <cstdint>

#include "iree/compiler/Dialect/IREE/IR/IREEAttributes.h"
#include "iree/compiler/Dialect/IREE/IR/IREETypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"

// Order matters.
#include "iree/compiler/Dialect/HAL/IR/HALEnums.h.inc"
#include "iree/compiler/Dialect/HAL/IR/HALStructs.h.inc"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

#include "iree/compiler/Dialect/HAL/IR/HALOpInterface.h.inc"

//===----------------------------------------------------------------------===//
// Enum utilities
//===----------------------------------------------------------------------===//

// Returns a stable identifier for the MLIR element type or nullopt if the
// type is unsupported in the ABI.
llvm::Optional<int32_t> getElementTypeValue(Type type);

// Returns an attribute with the MLIR element type or {}.
IntegerAttr getElementTypeAttr(Type type);

//===----------------------------------------------------------------------===//
// RefObject types
//===----------------------------------------------------------------------===//

class AllocatorType : public Type::TypeBase<AllocatorType, Type, TypeStorage> {
 public:
  using Base::Base;
  static AllocatorType get(MLIRContext *context) {
    return Base::get(context, TypeKind::Allocator);
  }
};

class BufferType : public Type::TypeBase<BufferType, Type, TypeStorage> {
 public:
  using Base::Base;
  static BufferType get(MLIRContext *context) {
    return Base::get(context, TypeKind::Buffer);
  }
};

class BufferViewType
    : public Type::TypeBase<BufferViewType, Type, TypeStorage> {
 public:
  using Base::Base;
  static BufferViewType get(MLIRContext *context) {
    return Base::get(context, TypeKind::BufferView);
  }
};

class CommandBufferType
    : public Type::TypeBase<CommandBufferType, Type, TypeStorage> {
 public:
  using Base::Base;
  static CommandBufferType get(MLIRContext *context) {
    return Base::get(context, TypeKind::CommandBuffer);
  }
};

class DescriptorSetType
    : public Type::TypeBase<DescriptorSetType, Type, TypeStorage> {
 public:
  using Base::Base;
  static DescriptorSetType get(MLIRContext *context) {
    return Base::get(context, TypeKind::DescriptorSet);
  }
};

class DescriptorSetLayoutType
    : public Type::TypeBase<DescriptorSetLayoutType, Type, TypeStorage> {
 public:
  using Base::Base;
  static DescriptorSetLayoutType get(MLIRContext *context) {
    return Base::get(context, TypeKind::DescriptorSetLayout);
  }
};

class DeviceType : public Type::TypeBase<DeviceType, Type, TypeStorage> {
 public:
  using Base::Base;
  static DeviceType get(MLIRContext *context) {
    return Base::get(context, TypeKind::Device);
  }
};

class EventType : public Type::TypeBase<EventType, Type, TypeStorage> {
 public:
  using Base::Base;
  static EventType get(MLIRContext *context) {
    return Base::get(context, TypeKind::Event);
  }
};

class ExecutableType
    : public Type::TypeBase<ExecutableType, Type, TypeStorage> {
 public:
  using Base::Base;
  static ExecutableType get(MLIRContext *context) {
    return Base::get(context, TypeKind::Executable);
  }
};

class ExecutableCacheType
    : public Type::TypeBase<ExecutableCacheType, Type, TypeStorage> {
 public:
  using Base::Base;
  static ExecutableCacheType get(MLIRContext *context) {
    return Base::get(context, TypeKind::ExecutableCache);
  }
};

class ExecutableLayoutType
    : public Type::TypeBase<ExecutableLayoutType, Type, TypeStorage> {
 public:
  using Base::Base;
  static ExecutableLayoutType get(MLIRContext *context) {
    return Base::get(context, TypeKind::ExecutableLayout);
  }
};

class RingBufferType
    : public Type::TypeBase<RingBufferType, Type, TypeStorage> {
 public:
  using Base::Base;
  static RingBufferType get(MLIRContext *context) {
    return Base::get(context, TypeKind::RingBuffer);
  }
};

class SemaphoreType : public Type::TypeBase<SemaphoreType, Type, TypeStorage> {
 public:
  using Base::Base;
  static SemaphoreType get(MLIRContext *context) {
    return Base::get(context, TypeKind::Semaphore);
  }
};

//===----------------------------------------------------------------------===//
// Struct types
//===----------------------------------------------------------------------===//

class BufferBarrierType {
 public:
  static TupleType get(MLIRContext *context) {
    return TupleType::get(
        {
            IntegerType::get(32, context),
            IntegerType::get(32, context),
            BufferType::get(context),
            IndexType::get(context),
            IndexType::get(context),
        },
        context);
  }
};

class MemoryBarrierType {
 public:
  static TupleType get(MLIRContext *context) {
    return TupleType::get(
        {
            IntegerType::get(32, context),
            IntegerType::get(32, context),
        },
        context);
  }
};

// A tuple containing runtime values for a descriptor set binding:
// <binding ordinal, hal.buffer, buffer byte offset, buffer byte length>
using DescriptorSetBindingValue = std::tuple<uint32_t, Value, Value, Value>;

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_HAL_IR_HALTYPES_H_

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

#ifndef IREE_COMPILER_IR_TYPES_H_
#define IREE_COMPILER_IR_TYPES_H_

#include <cstdint>

#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "third_party/llvm/llvm/include/llvm/ADT/DenseMapInfo.h"
#include "third_party/llvm/llvm/include/llvm/ADT/StringSwitch.h"

// Order matters.
#include "iree/compiler/IR/Enums.h.inc"

namespace mlir {
namespace iree_compiler {

namespace TypeKind {
enum Kind {
  Device = Type::FIRST_IREE_TYPE,
  DeviceGroup,
  CommandBuffer,
  Event,
  Semaphore,
  Fence,
};
}  // namespace TypeKind

// clang-format off
#define IREE_TYPE_TABLE(map)                                                   \
  map("device", TypeKind::Device, DeviceType)                                  \
  map("device_group", TypeKind::DeviceGroup, DeviceGroupType)                  \
  map("command_buffer", TypeKind::CommandBuffer, CommandBufferType)            \
  map("event", TypeKind::Event, EventType)                                     \
  map("semaphore", TypeKind::Semaphore, SemaphoreType)                         \
  map("fence", TypeKind::Fence, FenceType)
// clang-format on

// iree.device mapping to a runtime-resolved device type.
class DeviceType : public Type::TypeBase<DeviceType, Type> {
 public:
  using Base::Base;

  static bool kindof(unsigned kind) { return kind == TypeKind::Device; }

  static DeviceType get(MLIRContext *context);
};

// iree.device_group relating multiple iree.device requirements with each other.
class DeviceGroupType : public Type::TypeBase<DeviceGroupType, Type> {
 public:
  using Base::Base;

  static bool kindof(unsigned kind) { return kind == TypeKind::DeviceGroup; }

  static DeviceGroupType get(MLIRContext *context);
};

// iree.command_buffer mapping to an iree::hal::CommandBuffer.
class CommandBufferType : public Type::TypeBase<CommandBufferType, Type> {
 public:
  using Base::Base;

  static bool kindof(unsigned kind) { return kind == TypeKind::CommandBuffer; }

  static CommandBufferType get(MLIRContext *context);
};

// iree.event mapping to an iree::hal::Event.
class EventType : public Type::TypeBase<EventType, Type> {
 public:
  using Base::Base;

  static bool kindof(unsigned kind) { return kind == TypeKind::Event; }

  static EventType get(MLIRContext *context);
};

// iree.semaphore mapping to an iree::hal::Semaphore.
class SemaphoreType : public Type::TypeBase<SemaphoreType, Type> {
 public:
  using Base::Base;

  static bool kindof(unsigned kind) { return kind == TypeKind::Semaphore; }

  static SemaphoreType get(MLIRContext *context);
};

// iree.fence mapping to an iree::hal::Fence.
class FenceType : public Type::TypeBase<FenceType, Type> {
 public:
  using Base::Base;

  static bool kindof(unsigned kind) { return kind == TypeKind::Fence; }

  static FenceType get(MLIRContext *context);
};

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_IR_TYPES_H_

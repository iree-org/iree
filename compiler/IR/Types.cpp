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

#include "compiler/IR/Types.h"

#include "compiler/IR/Enums.cpp.inc"

namespace mlir {
namespace iree_compiler {

// static
DeviceType DeviceType::get(MLIRContext *context) {
  return Base::get(context, TypeKind::Device);
}

// static
DeviceGroupType DeviceGroupType::get(MLIRContext *context) {
  return Base::get(context, TypeKind::DeviceGroup);
}

// static
CommandBufferType CommandBufferType::get(MLIRContext *context) {
  return Base::get(context, TypeKind::CommandBuffer);
}

// static
EventType EventType::get(MLIRContext *context) {
  return Base::get(context, TypeKind::Event);
}

// static
SemaphoreType SemaphoreType::get(MLIRContext *context) {
  return Base::get(context, TypeKind::Semaphore);
}

// static
FenceType FenceType::get(MLIRContext *context) {
  return Base::get(context, TypeKind::Fence);
}

}  // namespace iree_compiler
}  // namespace mlir

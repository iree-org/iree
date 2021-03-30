// Copyright 2020 Google LLC
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

// This files defines a helper to trigger the registration of dialects to
// the system.
//
// Based on MLIR's InitAllDialects but for IREE dialects.

#ifndef IREE_TOOLS_INIT_IREE_DIALECTS_H_
#define IREE_TOOLS_INIT_IREE_DIALECTS_H_

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/IREE/IR/IREEDialect.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeDialect.h"
#include "iree/compiler/Dialect/VM/IR/VMDialect.h"
#include "iree/compiler/Dialect/VMLA/IR/VMLADialect.h"
#include "iree/compiler/Dialect/Vulkan/IR/VulkanDialect.h"
#include "mlir/IR/Dialect.h"

namespace mlir {
namespace iree_compiler {

// Add all the IREE dialects to the provided registry.
inline void registerIreeDialects(DialectRegistry &registry) {
  // clang-format off
  registry.insert<IREE::Flow::FlowDialect,
                  IREE::HAL::HALDialect,
                  ShapeDialect,
                  IREEDialect,
                  IREE::VM::VMDialect,
                  IREE::VMLA::VMLADialect,
                  IREE::Vulkan::VulkanDialect>();
  // clang-format on
}

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_TOOLS_INIT_IREE_DIALECTS_H_

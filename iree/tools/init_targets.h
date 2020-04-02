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

#ifndef IREE_TOOLS_INIT_TARGETS_H_
#define IREE_TOOLS_INIT_TARGETS_H_

#include "iree/compiler/Dialect/HAL/Target/LLVM/LLVMTarget.h"
#include "iree/compiler/Dialect/HAL/Target/VMLA/VMLATarget.h"
#include "iree/compiler/Dialect/HAL/Target/VulkanSPIRV/VulkanSPIRVTarget.h"

namespace mlir {
namespace iree_compiler {

// This function should be called before creating any MLIRContext if one
// expects all the possible target backends to be available. Custom tools can
// select which targets they want to support by only registering those they
// need.
inline void registerHALTargetBackends() {
  static bool init_once = []() {
    IREE::HAL::registerLLVMTargetBackends(
        []() { return IREE::HAL::getLLVMTargetOptionsFromFlags(); });
    IREE::HAL::registerVMLATargetBackends(
        []() { return IREE::HAL::getVMLATargetOptionsFromFlags(); });
    IREE::HAL::registerVulkanSPIRVTargetBackends(
        []() { return IREE::HAL::getVulkanSPIRVTargetOptionsFromFlags(); });
    return true;
  }();
  (void)init_once;
}

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_TOOLS_INIT_TARGETS_H_

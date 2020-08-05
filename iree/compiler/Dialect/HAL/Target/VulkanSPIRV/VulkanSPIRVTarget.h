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

#ifndef IREE_COMPILER_DIALECT_HAL_TARGET_VULKANSPIRV_VULKANSPIRVTARGET_H_
#define IREE_COMPILER_DIALECT_HAL_TARGET_VULKANSPIRV_VULKANSPIRVTARGET_H_

#include <string>

#include "iree/compiler/Conversion/LinalgToSPIRV/Passes.h"
#include "iree/compiler/Dialect/HAL/Target/TargetBackend.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

// Options controlling the SPIR-V translation.
struct VulkanSPIRVTargetOptions {
  // SPIR-V codegeneration options
  SPIRVCodegenOptions codegenOptions;
  // Vulkan target environment as #vk.target_env attribute assembly.
  std::string vulkanTargetEnv;
};

// Returns a VulkanSPIRVTargetOptions struct initialized with Vulkan/SPIR-V
// related command-line flags.
VulkanSPIRVTargetOptions getVulkanSPIRVTargetOptionsFromFlags();

// Registers the Vulkan/SPIR-V backends.
void registerVulkanSPIRVTargetBackends(
    std::function<VulkanSPIRVTargetOptions()> queryOptions);

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_HAL_TARGET_VULKANSPIRV_VULKANSPIRVTARGET_H_

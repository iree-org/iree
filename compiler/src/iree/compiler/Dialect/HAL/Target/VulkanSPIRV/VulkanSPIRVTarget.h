// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_HAL_TARGET_VULKANSPIRV_VULKANSPIRVTARGET_H_
#define IREE_COMPILER_DIALECT_HAL_TARGET_VULKANSPIRV_VULKANSPIRVTARGET_H_

#include <functional>
#include <string>

namespace mlir::iree_compiler::IREE::HAL {

// Options controlling the SPIR-V translation.
struct VulkanSPIRVTargetOptions {
  // Vulkan target environment, either as #vk.target_env attribute assembly
  // or as a Vulkan target triple.
  std::string targetTripleOrEnv;
  // Whether to use indirect bindings for all generated dispatches.
  bool indirectBindings = false;
};

// Returns a VulkanSPIRVTargetOptions struct initialized with Vulkan/SPIR-V
// related command-line flags.
VulkanSPIRVTargetOptions getVulkanSPIRVTargetOptionsFromFlags();

// Registers the Vulkan/SPIR-V backends.
void registerVulkanSPIRVTargetBackends(
    std::function<VulkanSPIRVTargetOptions()> queryOptions);

} // namespace mlir::iree_compiler::IREE::HAL

#endif // IREE_COMPILER_DIALECT_HAL_TARGET_VULKANSPIRV_VULKANSPIRVTARGET_H_

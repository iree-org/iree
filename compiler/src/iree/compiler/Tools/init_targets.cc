// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Tools/init_targets.h"

#include <functional>

#ifdef IREE_HAVE_LLVM_CPU_TARGET
#include "iree/compiler/Dialect/HAL/Target/LLVMCPU/LLVMCPUTarget.h"
#endif // IREE_HAVE_LLVM_CPU_TARGET
#ifdef IREE_HAVE_VULKANSPIRV_TARGET
#include "iree/compiler/Dialect/HAL/Target/VulkanSPIRV/VulkanSPIRVTarget.h"
#endif // IREE_HAVE_VULKANSPIRV_TARGET

namespace mlir::iree_compiler {

// This function should be called before creating any MLIRContext if one
// expects all the possible target backends to be available. Custom tools can
// select which targets they want to support by only registering those they
// need.
void registerHALTargetBackends() {
  static bool init_once = []() {
#ifdef IREE_HAVE_LLVM_CPU_TARGET
    IREE::HAL::registerLLVMCPUTargetBackends(
        []() { return IREE::HAL::LLVMTargetOptions::getFromFlags(); });
#endif // IREE_HAVE_LLVM_CPU_TARGET
#ifdef IREE_HAVE_VULKANSPIRV_TARGET
    IREE::HAL::registerVulkanSPIRVTargetBackends(
        []() { return IREE::HAL::getVulkanSPIRVTargetOptionsFromFlags(); });
#endif // IREE_HAVE_VULKANSPIRV_TARGET
    return true;
  }();
  (void)init_once;
}

} // namespace mlir::iree_compiler

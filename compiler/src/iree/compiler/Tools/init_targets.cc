// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Tools/init_targets.h"

#include <functional>

#ifdef IREE_HAVE_CUDA_TARGET
#include "iree/compiler/Dialect/HAL/Target/CUDA/CUDATarget.h"
#endif  // IREE_HAVE_CUDA_TARGET
#ifdef IREE_HAVE_LLVM_CPU_TARGET
#include "iree/compiler/Dialect/HAL/Target/LLVMCPU/LLVMCPUTarget.h"
#endif  // IREE_HAVE_LLVM_CPU_TARGET
#ifdef IREE_HAVE_METALSPIRV_TARGET
#include "iree/compiler/Dialect/HAL/Target/MetalSPIRV/MetalSPIRVTarget.h"
#endif  // IREE_HAVE_METALSPIRV_TARGET
#ifdef IREE_HAVE_ROCM_TARGET
#include "iree/compiler/Dialect/HAL/Target/ROCM/ROCMTarget.h"
#endif  // IREE_HAVE_ROCM_TARGET
#ifdef IREE_HAVE_VMVX_TARGET
#include "iree/compiler/Dialect/HAL/Target/VMVX/VMVXTarget.h"
#endif  // IREE_HAVE_VMVX_TARGET
#ifdef IREE_HAVE_VULKANSPIRV_TARGET
#include "iree/compiler/Dialect/HAL/Target/VulkanSPIRV/VulkanSPIRVTarget.h"
#endif  // IREE_HAVE_VULKANSPIRV_TARGET
#ifdef IREE_HAVE_WEBGPU_TARGET
#include "iree/compiler/Dialect/HAL/Target/WebGPU/WebGPUTarget.h"
#endif  // IREE_HAVE_WEBGPU_TARGET

namespace mlir {
namespace iree_compiler {

// This function should be called before creating any MLIRContext if one
// expects all the possible target backends to be available. Custom tools can
// select which targets they want to support by only registering those they
// need.
void registerHALTargetBackends() {
  static bool init_once = []() {

#ifdef IREE_HAVE_CUDA_TARGET
    IREE::HAL::registerCUDATargetBackends();
#endif  // IREE_HAVE_CUDA_TARGET
#ifdef IREE_HAVE_LLVM_CPU_TARGET
    IREE::HAL::registerLLVMCPUTargetBackends(
        []() { return IREE::HAL::getLLVMTargetOptionsFromFlags(); });
#endif  // IREE_HAVE_LLVM_CPU_TARGET
#ifdef IREE_HAVE_METALSPIRV_TARGET
    IREE::HAL::registerMetalSPIRVTargetBackends();
#endif  // IREE_HAVE_METALSPIRV_TARGET
#ifdef IREE_HAVE_ROCM_TARGET
    IREE::HAL::registerROCMTargetBackends();
#endif  // IREE_HAVE_ROCM_TARGET
#ifdef IREE_HAVE_VMVX_TARGET
    IREE::HAL::registerVMVXTargetBackends();
#endif  // IREE_HAVE_VMVX_TARGET
#ifdef IREE_HAVE_VULKANSPIRV_TARGET
    IREE::HAL::registerVulkanSPIRVTargetBackends(
        []() { return IREE::HAL::getVulkanSPIRVTargetOptionsFromFlags(); });
#endif  // IREE_HAVE_VULKANSPIRV_TARGET
#ifdef IREE_HAVE_WEBGPU_TARGET
    IREE::HAL::registerWebGPUTargetBackends(
        []() { return IREE::HAL::getWebGPUTargetOptionsFromFlags(); });
#endif  // IREE_HAVE_WEBGPU_TARGET
    return true;
  }();
  (void)init_once;
}

}  // namespace iree_compiler
}  // namespace mlir

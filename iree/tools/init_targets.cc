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

#include "iree/tools/init_targets.h"

#ifdef IREE_HAVE_CUDA_TARGET
#include "iree/compiler/Dialect/HAL/Target/CUDA/CUDATarget.h"
#endif  // IREE_HAVE_CUDA_TARGET
#ifdef IREE_HAVE_LLVMAOT_TARGET
#include "iree/compiler/Dialect/HAL/Target/LLVM/LLVMAOTTarget.h"
#endif  // IREE_HAVE_LLVMAOT_TARGET
#ifdef IREE_HAVE_METALSPIRV_TARGET
#include "iree/compiler/Dialect/HAL/Target/MetalSPIRV/MetalSPIRVTarget.h"
#endif  // IREE_HAVE_METALSPIRV_TARGET
#ifdef IREE_HAVE_VMLA_TARGET
#include "iree/compiler/Dialect/HAL/Target/VMLA/VMLATarget.h"
#endif  // IREE_HAVE_VMLA_TARGET
#ifdef IREE_HAVE_VMVX_TARGET
#include "iree/compiler/Dialect/HAL/Target/VMVX/VMVXTarget.h"
#endif  // IREE_HAVE_VMVX_TARGET
#ifdef IREE_HAVE_VULKANSPIRV_TARGET
#include "iree/compiler/Dialect/HAL/Target/VulkanSPIRV/VulkanSPIRVTarget.h"
#endif  // IREE_HAVE_VULKANSPIRV_TARGET

namespace mlir {
namespace iree_compiler {

// This function should be called before creating any MLIRContext if one
// expects all the possible target backends to be available. Custom tools can
// select which targets they want to support by only registering those they
// need.
void registerHALTargetBackends() {
  static bool init_once = []() {

#ifdef IREE_HAVE_CUDA_TARGET
    IREE::HAL::registerCUDATargetBackends(
        []() { return IREE::HAL::getCUDATargetOptionsFromFlags(); });
#endif  // IREE_HAVE_CUDA_TARGET
#ifdef IREE_HAVE_LLVMAOT_TARGET
    IREE::HAL::registerLLVMAOTTargetBackends(
        []() { return IREE::HAL::getLLVMTargetOptionsFromFlags(); });
#endif  // IREE_HAVE_LLVMAOT_TARGET
#ifdef IREE_HAVE_METALSPIRV_TARGET
    IREE::HAL::registerMetalSPIRVTargetBackends(
        []() { return IREE::HAL::getMetalSPIRVTargetOptionsFromFlags(); });
#endif  // IREE_HAVE_METALSPIRV_TARGET
#ifdef IREE_HAVE_VMLA_TARGET
    IREE::HAL::registerVMLATargetBackends(
        []() { return IREE::HAL::getVMLATargetOptionsFromFlags(); });
#endif  // IREE_HAVE_VMLA_TARGET
#ifdef IREE_HAVE_VMVX_TARGET
    IREE::HAL::registerVMVXTargetBackends(
        []() { return IREE::HAL::getVMVXTargetOptionsFromFlags(); });
#endif  // IREE_HAVE_VMVX_TARGET
#ifdef IREE_HAVE_VULKANSPIRV_TARGET
    IREE::HAL::registerVulkanSPIRVTargetBackends(
        []() { return IREE::HAL::getVulkanSPIRVTargetOptionsFromFlags(); });
#endif  // IREE_HAVE_VULKANSPIRV_TARGET
    return true;
  }();
  (void)init_once;
}

}  // namespace iree_compiler
}  // namespace mlir

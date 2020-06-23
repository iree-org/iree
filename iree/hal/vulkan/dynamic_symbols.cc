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

#include "iree/hal/vulkan/dynamic_symbols.h"

#include <cstddef>

#include "absl/base/attributes.h"
#include "absl/base/macros.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "iree/base/platform_headers.h"
#include "iree/base/source_location.h"
#include "iree/base/status.h"
#include "iree/base/target_platform.h"
#include "iree/base/tracing.h"
#include "iree/hal/vulkan/dynamic_symbol_tables.h"

namespace iree {
namespace hal {
namespace vulkan {

// Read-only table of function pointer information designed to be in .rdata.
// To reduce binary size this structure is packed (knowing that we won't have
// gigabytes of function pointers :).
struct FunctionPtrInfo {
  // Name of the function (like 'vkSomeFunction').
  const char* function_name;
  // 1 if the function pointer can be resolved via vkGetDeviceProcAddr.
  uint32_t is_device : 1;
  // 1 if the function is required and the loader should bail if not found.
  uint32_t is_required : 1;
  // TODO(benvanik): remove from table by manually walking sizeof(uintptr_t).
  // An offset in bytes from the base of &syms to where the PFN_vkSomeFunction
  // member is located.
  uint32_t member_offset : 30;
} ABSL_ATTRIBUTE_PACKED;

namespace {

#define REQUIRED_PFN_FUNCTION_PTR(function_name, is_device) \
  {#function_name, is_device, 1, offsetof(DynamicSymbols, function_name)},
#define OPTIONAL_PFN_FUNCTION_PTR(function_name, is_device) \
  {#function_name, is_device, 0, offsetof(DynamicSymbols, function_name)},
#define EXCLUDED_PFN_FUNCTION_PTR(function_name, is_device)
#define INS_PFN_FUNCTION_PTR(requirement, function_name) \
  requirement##_PFN_FUNCTION_PTR(function_name, 0)
#define DEV_PFN_FUNCTION_PTR(requirement, function_name) \
  requirement##_PFN_FUNCTION_PTR(function_name, 1)

// Defines the table of mandatory FunctionPtrInfos resolved prior to instance
// creation. These are safe to call with no instance parameter and should be
// exported by all loaders/ICDs.
static constexpr const FunctionPtrInfo kInstancelessFunctionPtrInfos[] = {
    REQUIRED_PFN_FUNCTION_PTR(vkCreateInstance, false)                        //
    REQUIRED_PFN_FUNCTION_PTR(vkEnumerateInstanceLayerProperties, false)      //
    REQUIRED_PFN_FUNCTION_PTR(vkEnumerateInstanceExtensionProperties, false)  //
};

// Defines the table of FunctionPtrInfos for dynamic loading that must wait
// until an instance has been created to be resolved.
static constexpr const FunctionPtrInfo kDynamicFunctionPtrInfos[] = {
    IREE_VULKAN_DYNAMIC_SYMBOL_TABLES(INS_PFN_FUNCTION_PTR,
                                      DEV_PFN_FUNCTION_PTR)};

static const char* kVulkanLoaderSearchNames[] = {
#if defined(IREE_PLATFORM_ANDROID)
    "libvulkan.so",
#elif defined(IREE_PLATFORM_WINDOWS)
    "vulkan-1.dll",
#else
    "libvulkan.so.1",
#endif  // IREE_PLATFORM_WINDOWS
};

Status ResolveFunctions(DynamicSymbols* syms,
                        const DynamicSymbols::GetProcAddrFn& get_proc_addr) {
  // Resolve the method the shared object uses to resolve other functions.
  // Some libraries will export all symbols while others will only export this
  // single function.
  syms->vkGetInstanceProcAddr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
      get_proc_addr("vkGetInstanceProcAddr"));
  if (!syms->vkGetInstanceProcAddr) {
    return UnavailableErrorBuilder(IREE_LOC)
           << "Required method vkGetInstanceProcAddr not "
              "found in provided Vulkan library (did you pick the wrong file?)";
  }

  // Resolve the mandatory functions that we need to create instances.
  // If the provided |get_proc_addr| cannot resolve these then it's not a loader
  // or ICD we want to use, anyway.
  for (int i = 0; i < ABSL_ARRAYSIZE(kInstancelessFunctionPtrInfos); ++i) {
    const auto& function_ptr = kInstancelessFunctionPtrInfos[i];
    auto* member_ptr = reinterpret_cast<PFN_vkVoidFunction*>(
        reinterpret_cast<uint8_t*>(syms) + function_ptr.member_offset);
    *member_ptr =
        syms->vkGetInstanceProcAddr(VK_NULL_HANDLE, function_ptr.function_name);
    if (*member_ptr == nullptr) {
      return UnavailableErrorBuilder(IREE_LOC)
             << "Mandatory Vulkan function " << function_ptr.function_name
             << " not available; invalid loader/ICD?";
    }
  }

  return OkStatus();
}

}  // namespace

// static
StatusOr<ref_ptr<DynamicSymbols>> DynamicSymbols::Create(
    const GetProcAddrFn& get_proc_addr) {
  IREE_TRACE_SCOPE0("DynamicSymbols::Create");

  auto syms = make_ref<DynamicSymbols>();
  RETURN_IF_ERROR(ResolveFunctions(syms.get(), get_proc_addr));
  syms->FixupExtensionFunctions();
  return syms;
}

// static
StatusOr<ref_ptr<DynamicSymbols>> DynamicSymbols::CreateFromSystemLoader() {
  IREE_TRACE_SCOPE0("DynamicSymbols::CreateFromSystemLoader");

  ASSIGN_OR_RETURN(
      auto loader_library,
      DynamicLibrary::Load(absl::MakeSpan(kVulkanLoaderSearchNames)));
  auto syms = make_ref<DynamicSymbols>();
  syms->loader_library_ = std::move(loader_library);

  auto* loader_library_ptr = syms->loader_library_.get();
  RETURN_IF_ERROR(ResolveFunctions(
      syms.get(), [loader_library_ptr](const char* function_name) {
        return loader_library_ptr->GetSymbol<PFN_vkVoidFunction>(function_name);
      }));
  syms->FixupExtensionFunctions();
  return syms;
}

Status DynamicSymbols::LoadFromInstance(VkInstance instance) {
  IREE_TRACE_SCOPE0("DynamicSymbols::LoadFromInstance");
  return LoadFromDevice(instance, VK_NULL_HANDLE);
}

Status DynamicSymbols::LoadFromDevice(VkInstance instance, VkDevice device) {
  IREE_TRACE_SCOPE0("DynamicSymbols::LoadFromDevice");

  if (!instance) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Instance must have been created and a default instance proc "
              "lookup function is required";
  }

  // Setup the lookup methods first. The rest of the syms uses these to
  // resolve function pointers.
  this->vkGetDeviceProcAddr = reinterpret_cast<PFN_vkGetDeviceProcAddr>(
      this->vkGetInstanceProcAddr(instance, "vkGetDeviceProcAddr"));
  if (!this->vkGetDeviceProcAddr) {
    return UnavailableErrorBuilder(IREE_LOC)
           << "Required Vulkan function vkGetDeviceProcAddr not available; "
              "invalid driver handle?";
  }

  // Load the rest of the functions.
  for (int i = 0; i < ABSL_ARRAYSIZE(kDynamicFunctionPtrInfos); ++i) {
    const auto& function_ptr = kDynamicFunctionPtrInfos[i];
    auto* member_ptr = reinterpret_cast<PFN_vkVoidFunction*>(
        reinterpret_cast<uint8_t*>(this) + function_ptr.member_offset);
    if (function_ptr.is_device && device) {
      *member_ptr =
          this->vkGetDeviceProcAddr(device, function_ptr.function_name);
    } else {
      *member_ptr =
          this->vkGetInstanceProcAddr(instance, function_ptr.function_name);
    }
    if (*member_ptr == nullptr && function_ptr.is_required) {
      return UnavailableErrorBuilder(IREE_LOC)
             << "Required Vulkan function " << function_ptr.function_name
             << " not available";
    }
  }

  FixupExtensionFunctions();

  return OkStatus();
}

DynamicSymbols::DynamicSymbols() = default;

DynamicSymbols::~DynamicSymbols() = default;

void DynamicSymbols::FixupExtensionFunctions() {
  this->vkGetSemaphoreCounterValue = this->vkGetSemaphoreCounterValue
                                         ? this->vkGetSemaphoreCounterValue
                                         : this->vkGetSemaphoreCounterValueKHR;
  this->vkWaitSemaphores = this->vkWaitSemaphores ? this->vkWaitSemaphores
                                                  : this->vkWaitSemaphoresKHR;
  this->vkSignalSemaphore = this->vkSignalSemaphore
                                ? this->vkSignalSemaphore
                                : this->vkSignalSemaphoreKHR;
}

}  // namespace vulkan
}  // namespace hal
}  // namespace iree

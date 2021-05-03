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

#include "iree/hal/vulkan/api.h"

#include "iree/base/api.h"
#include "iree/base/tracing.h"
#include "iree/hal/vulkan/dynamic_symbols.h"
#include "iree/hal/vulkan/extensibility_util.h"
#include "iree/hal/vulkan/vulkan_device.h"
#include "iree/hal/vulkan/vulkan_driver.h"

using namespace iree::hal::vulkan;

// TODO(benvanik): move these into the appropriate files and delete this .cc.

//===----------------------------------------------------------------------===//
// iree::hal::vulkan::DynamicSymbols
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t iree_hal_vulkan_syms_create(
    void* vkGetInstanceProcAddr_fn, iree_allocator_t host_allocator,
    iree_hal_vulkan_syms_t** out_syms) {
  IREE_TRACE_SCOPE0("iree_hal_vulkan_syms_create");
  IREE_ASSERT_ARGUMENT(out_syms);
  *out_syms = nullptr;

  iree::ref_ptr<iree::hal::vulkan::DynamicSymbols> syms;
  IREE_RETURN_IF_ERROR(DynamicSymbols::Create(
      [&vkGetInstanceProcAddr_fn](const char* function_name) {
        // Only resolve vkGetInstanceProcAddr, rely on syms->LoadFromInstance()
        // and/or syms->LoadFromDevice() for further loading.
        std::string fn = "vkGetInstanceProcAddr";
        if (strncmp(function_name, fn.data(), fn.size()) == 0) {
          return reinterpret_cast<PFN_vkVoidFunction>(vkGetInstanceProcAddr_fn);
        }
        return reinterpret_cast<PFN_vkVoidFunction>(NULL);
      },
      &syms));

  *out_syms = reinterpret_cast<iree_hal_vulkan_syms_t*>(syms.release());
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_hal_vulkan_syms_create_from_system_loader(
    iree_allocator_t host_allocator, iree_hal_vulkan_syms_t** out_syms) {
  IREE_TRACE_SCOPE0("iree_hal_vulkan_syms_create_from_system_loader");
  IREE_ASSERT_ARGUMENT(out_syms);
  *out_syms = nullptr;

  iree::ref_ptr<iree::hal::vulkan::DynamicSymbols> syms;
  IREE_RETURN_IF_ERROR(DynamicSymbols::CreateFromSystemLoader(&syms));
  *out_syms = reinterpret_cast<iree_hal_vulkan_syms_t*>(syms.release());
  return iree_ok_status();
}

IREE_API_EXPORT void iree_hal_vulkan_syms_retain(iree_hal_vulkan_syms_t* syms) {
  IREE_ASSERT_ARGUMENT(syms);
  auto* handle = reinterpret_cast<DynamicSymbols*>(syms);
  if (handle) {
    handle->AddReference();
  }
}

IREE_API_EXPORT void iree_hal_vulkan_syms_release(
    iree_hal_vulkan_syms_t* syms) {
  IREE_ASSERT_ARGUMENT(syms);
  auto* handle = reinterpret_cast<DynamicSymbols*>(syms);
  if (handle) {
    handle->ReleaseReference();
  }
}

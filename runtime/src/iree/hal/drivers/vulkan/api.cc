// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/api.h"

#include <cstring>
#include <functional>
#include <string>

#include "iree/base/api.h"
#include "iree/base/tracing.h"
#include "iree/hal/drivers/vulkan/dynamic_symbols.h"
#include "iree/hal/drivers/vulkan/util/ref_ptr.h"

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
  IREE_RETURN_IF_ERROR(iree::hal::vulkan::DynamicSymbols::Create(
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
  IREE_RETURN_IF_ERROR(
      iree::hal::vulkan::DynamicSymbols::CreateFromSystemLoader(&syms));
  *out_syms = reinterpret_cast<iree_hal_vulkan_syms_t*>(syms.release());
  return iree_ok_status();
}

IREE_API_EXPORT void iree_hal_vulkan_syms_retain(iree_hal_vulkan_syms_t* syms) {
  IREE_ASSERT_ARGUMENT(syms);
  auto* handle = reinterpret_cast<iree::hal::vulkan::DynamicSymbols*>(syms);
  if (handle) {
    handle->AddReference();
  }
}

IREE_API_EXPORT void iree_hal_vulkan_syms_release(
    iree_hal_vulkan_syms_t* syms) {
  IREE_ASSERT_ARGUMENT(syms);
  auto* handle = reinterpret_cast<iree::hal::vulkan::DynamicSymbols*>(syms);
  if (handle) {
    handle->ReleaseReference();
  }
}

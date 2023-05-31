// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/registration/driver_module.h"

#include <cinttypes>
#include <cstddef>

#include "iree/base/api.h"
#include "iree/base/internal/flags.h"
#include "iree/base/tracing.h"
#include "iree/hal/drivers/vulkan/api.h"

#ifndef NDEBUG
#define IREE_HAL_VULKAN_DEBUG_FLAG_DEFAULT true
#else
#define IREE_HAL_VULKAN_DEBUG_FLAG_DEFAULT false
#endif  // !NDEBUG

IREE_FLAG(bool, vulkan_validation_layers, IREE_HAL_VULKAN_DEBUG_FLAG_DEFAULT,
          "Enables standard Vulkan validation layers.");
IREE_FLAG(bool, vulkan_debug_utils, IREE_HAL_VULKAN_DEBUG_FLAG_DEFAULT,
          "Enables VK_EXT_debug_utils, records markers, and logs errors.");
IREE_FLAG(int32_t, vulkan_debug_verbosity, 2,
          "Cutoff for debug output; "
          "0=none, 1=errors, 2=warnings, 3=info, 4=debug.");

IREE_FLAG(bool, vulkan_tracing, true,
          "Enables Vulkan tracing (if IREE tracing is enabled).");

IREE_FLAG(bool, vulkan_robust_buffer_access, false,
          "Enables the Vulkan 'robustBufferAccess' feature.");

IREE_FLAG(
    bool, vulkan_dedicated_compute_queue, false,
    "Use a dedicated queue with VK_QUEUE_COMPUTE_BIT for dispatch workloads.");
IREE_FLAG(
    int64_t, vulkan_large_heap_block_size, 0,
    "Preferred allocator block size for large allocations in bytes. Sets the\n"
    "minimum bound on memory consumption.");

static iree_status_t iree_hal_vulkan_create_driver_with_flags(
    iree_string_view_t identifier, iree_allocator_t host_allocator,
    iree_hal_driver_t** out_driver) {
  IREE_TRACE_SCOPE();

  // Setup driver options from flags. We do this here as we want to enable other
  // consumers that may not be using modules/command line flags to be able to
  // set their options however they want.
  iree_hal_vulkan_driver_options_t driver_options;
  iree_hal_vulkan_driver_options_initialize(&driver_options);

// TODO(benvanik): make this a flag - it's useful for testing the same binary
// against multiple versions of Vulkan.
#if defined(IREE_PLATFORM_ANDROID)
  // TODO(#4494): always enable 1.2
  driver_options.api_version = VK_API_VERSION_1_1;
#else
  driver_options.api_version = VK_API_VERSION_1_2;
#endif  // IREE_PLATFORM_ANDROID

  if (FLAG_vulkan_validation_layers) {
    driver_options.requested_features |=
        IREE_HAL_VULKAN_FEATURE_ENABLE_VALIDATION_LAYERS;
  }
  if (FLAG_vulkan_debug_utils) {
    driver_options.requested_features |=
        IREE_HAL_VULKAN_FEATURE_ENABLE_DEBUG_UTILS;
    driver_options.debug_verbosity = FLAG_vulkan_debug_verbosity;
  }
  if (FLAG_vulkan_tracing) {
    driver_options.requested_features |= IREE_HAL_VULKAN_FEATURE_ENABLE_TRACING;
  }
  if (FLAG_vulkan_robust_buffer_access) {
    driver_options.requested_features |=
        IREE_HAL_VULKAN_FEATURE_ENABLE_ROBUST_BUFFER_ACCESS;
  }

  if (FLAG_vulkan_dedicated_compute_queue) {
    driver_options.device_options.flags |=
        IREE_HAL_VULKAN_DEVICE_FLAG_DEDICATED_COMPUTE_QUEUE;
  }
  if (FLAG_vulkan_large_heap_block_size) {
    driver_options.device_options.large_heap_block_size =
        FLAG_vulkan_large_heap_block_size;
  }

  // Load the Vulkan library. This will fail if the library cannot be found or
  // does not have the expected functions.
  iree_hal_vulkan_syms_t* syms = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_syms_create_from_system_loader(host_allocator, &syms));

  iree_status_t status = iree_hal_vulkan_driver_create(
      identifier, &driver_options, syms, host_allocator, out_driver);

  iree_hal_vulkan_syms_release(syms);
  return status;
}

static iree_status_t iree_hal_vulkan_driver_factory_enumerate(
    void* self, iree_host_size_t* out_driver_info_count,
    const iree_hal_driver_info_t** out_driver_infos) {
  // NOTE: we could query supported vulkan versions or featuresets here.
  static const iree_hal_driver_info_t driver_infos[1] = {{
      /*driver_name=*/iree_make_cstring_view("vulkan"),
      /*full_name=*/iree_make_cstring_view("Vulkan 1.x (dynamic)"),
  }};
  *out_driver_info_count = IREE_ARRAYSIZE(driver_infos);
  *out_driver_infos = driver_infos;
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_driver_factory_try_create(
    void* self, iree_string_view_t driver_name, iree_allocator_t host_allocator,
    iree_hal_driver_t** out_driver) {
  if (!iree_string_view_equal(driver_name, IREE_SV("vulkan"))) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "no driver '%.*s' is provided by this factory",
                            (int)driver_name.size, driver_name.data);
  }
  return iree_hal_vulkan_create_driver_with_flags(driver_name, host_allocator,
                                                  out_driver);
}

IREE_API_EXPORT iree_status_t
iree_hal_vulkan_driver_module_register(iree_hal_driver_registry_t* registry) {
  static const iree_hal_driver_factory_t factory = {
      /*self=*/NULL,
      iree_hal_vulkan_driver_factory_enumerate,
      iree_hal_vulkan_driver_factory_try_create,
  };
  return iree_hal_driver_registry_register_factory(registry, &factory);
}

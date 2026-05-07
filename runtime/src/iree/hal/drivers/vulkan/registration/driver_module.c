// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/registration/driver_module.h"

#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/drivers/vulkan/api.h"

IREE_FLAG_LIST(string, vulkan_libvulkan_search_path,
               "Search path (directory or file) for the Vulkan loader library "
               "(`libvulkan.so`, `vulkan-1.dll`, etc).");

#ifndef NDEBUG
#define IREE_HAL_VULKAN_DEBUG_FLAG_DEFAULT true
#else
#define IREE_HAL_VULKAN_DEBUG_FLAG_DEFAULT false
#endif  // !NDEBUG

IREE_FLAG(bool, vulkan_validation_layers, IREE_HAL_VULKAN_DEBUG_FLAG_DEFAULT,
          "Requests Vulkan validation layers.");
IREE_FLAG(bool, vulkan_debug_utils, IREE_HAL_VULKAN_DEBUG_FLAG_DEFAULT,
          "Requests Vulkan debug utils.");
IREE_FLAG(int32_t, vulkan_debug_verbosity, 2,
          "Cutoff for debug output; 0=none, 1=errors, 2=warnings, 3=info, "
          "4=debug.");
IREE_FLAG(bool, vulkan_tracing, true,
          "Requests Vulkan events in HAL profiling streams.");
IREE_FLAG(bool, vulkan_robust_buffer_access, false,
          "Requests robust buffer access.");
IREE_FLAG(bool, vulkan_sparse_binding, true,
          "Requests sparse binding for large virtual buffers.");
IREE_FLAG(bool, vulkan_sparse_residency, true,
          "Requests sparse residency and aliased sparse buffer mappings.");
IREE_FLAG(bool, vulkan_buffer_device_addresses, true,
          "Requests buffer device addresses for pointer-first executables.");
IREE_FLAG(string, vulkan_dispatch_abi, "all",
          "Executable dispatch ABI policy: descriptor, bda, or all.");
IREE_FLAG(
    int32_t, vulkan_cached_bda_replay_instances, 16,
    "Maximum cached native BDA command-buffer replay instances per queue lane; "
    "0 disables replay caching.");
IREE_FLAG(int64_t, vulkan_cached_bda_replay_publication_bytes,
          64ll * 1024ll * 1024ll,
          "Maximum cached native BDA replay publication bytes per queue lane.");
IREE_FLAG(
    int32_t, vulkan_retained_cached_bda_replay_instances, 1,
    "Idle cached native BDA replay instances retained per command buffer after "
    "device trim.");
IREE_FLAG(
    bool, vulkan_dedicated_compute_queue, true,
    "Requests a dedicated queue with VK_QUEUE_COMPUTE_BIT for dispatch work.");

static iree_status_t iree_hal_vulkan_driver_factory_enumerate(
    void* self, iree_host_size_t* out_driver_info_count,
    const iree_hal_driver_info_t** out_driver_infos) {
  static const iree_hal_driver_info_t default_driver_info = {
      .driver_name = IREE_SVL("vulkan"),
      .full_name = IREE_SVL("Vulkan HAL Rewrite Scaffold"),
  };
  *out_driver_info_count = 1;
  *out_driver_infos = &default_driver_info;
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

  iree_hal_vulkan_driver_options_t options;
  iree_hal_vulkan_driver_options_initialize(&options);

  options.libvulkan_search_paths = FLAG_vulkan_libvulkan_search_path_list();
  options.debug_verbosity = FLAG_vulkan_debug_verbosity;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_dispatch_abis_parse(
      iree_make_cstring_view(FLAG_vulkan_dispatch_abi),
      &options.device_options.dispatch_abis));
  if (FLAG_vulkan_cached_bda_replay_instances < 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--vulkan_cached_bda_replay_instances must be non-negative");
  }
  if (FLAG_vulkan_cached_bda_replay_publication_bytes < 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--vulkan_cached_bda_replay_publication_bytes must be non-negative");
  }
  if (FLAG_vulkan_retained_cached_bda_replay_instances < 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--vulkan_retained_cached_bda_replay_instances must be non-negative");
  }
  options.device_options.max_cached_bda_replay_instances =
      (uint32_t)FLAG_vulkan_cached_bda_replay_instances;
  options.device_options.max_cached_bda_replay_publication_bytes =
      (uint64_t)FLAG_vulkan_cached_bda_replay_publication_bytes;
  options.device_options.retained_cached_bda_replay_instances =
      (uint32_t)FLAG_vulkan_retained_cached_bda_replay_instances;
  if (options.device_options.max_cached_bda_replay_instances != 0 &&
      options.device_options.retained_cached_bda_replay_instances >
          options.device_options.max_cached_bda_replay_instances) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--vulkan_retained_cached_bda_replay_instances must be <= "
        "--vulkan_cached_bda_replay_instances");
  }
  if (FLAG_vulkan_validation_layers) {
    options.requested_features |=
        IREE_HAL_VULKAN_FEATURE_ENABLE_VALIDATION_LAYERS;
  }
  if (FLAG_vulkan_debug_utils) {
    options.requested_features |= IREE_HAL_VULKAN_FEATURE_ENABLE_DEBUG_UTILS;
  }
  if (FLAG_vulkan_tracing) {
    options.requested_features |= IREE_HAL_VULKAN_FEATURE_ENABLE_TRACING;
  }
  if (FLAG_vulkan_robust_buffer_access) {
    options.requested_features |=
        IREE_HAL_VULKAN_FEATURE_ENABLE_ROBUST_BUFFER_ACCESS;
  }
  if (FLAG_vulkan_sparse_binding) {
    options.requested_features |= IREE_HAL_VULKAN_FEATURE_ENABLE_SPARSE_BINDING;
  }
  if (FLAG_vulkan_sparse_residency) {
    options.requested_features |=
        IREE_HAL_VULKAN_FEATURE_ENABLE_SPARSE_RESIDENCY_ALIASED;
  }
  if (FLAG_vulkan_buffer_device_addresses) {
    options.requested_features |=
        IREE_HAL_VULKAN_FEATURE_ENABLE_BUFFER_DEVICE_ADDRESSES;
  }
  if (FLAG_vulkan_dedicated_compute_queue) {
    options.device_options.flags |=
        IREE_HAL_VULKAN_DEVICE_FLAG_DEDICATED_COMPUTE_QUEUE;
  } else {
    options.device_options.flags &=
        ~IREE_HAL_VULKAN_DEVICE_FLAG_DEDICATED_COMPUTE_QUEUE;
  }

  return iree_hal_vulkan_driver_create(driver_name, &options, /*syms=*/NULL,
                                       host_allocator, out_driver);
}

IREE_API_EXPORT iree_status_t
iree_hal_vulkan_driver_module_register(iree_hal_driver_registry_t* registry) {
  static const iree_hal_driver_factory_t factory = {
      .self = NULL,
      .enumerate = iree_hal_vulkan_driver_factory_enumerate,
      .try_create = iree_hal_vulkan_driver_factory_try_create,
  };
  return iree_hal_driver_registry_register_factory(registry, &factory);
}

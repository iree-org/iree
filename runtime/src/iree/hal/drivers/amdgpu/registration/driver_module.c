// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/registration/driver_module.h"

#include "iree/base/api.h"
#include "iree/base/internal/flags.h"
#include "iree/hal/drivers/amdgpu/api.h"

IREE_FLAG_LIST(string, amdgpu_libhsa_search_path,
               "Search path (directory or file) for the ROCR-Runtime library "
               "(`libhsa-runtime64.so`, etc).");

IREE_FLAG(int64_t, amdgpu_host_block_pool_small_size, 0,
          "Size in bytes of a small host block in the pool. Must be a power of "
          "two or 0 for the default.");
IREE_FLAG(int64_t, amdgpu_host_block_pool_large_size, 0,
          "Size in bytes of a large host block in the pool. Must be a power of "
          "two or 0 for the default.");

IREE_FLAG(int64_t, amdgpu_device_block_pool_small_size, 0,
          "Size in bytes of a small device block in the pool. Must be a power "
          "of two or 0 for the default.");
IREE_FLAG(int64_t, amdgpu_device_block_pool_small_capacity, 0,
          "Initial small block pool block allocation count in blocks or 0 for "
          "the default.");
IREE_FLAG(int64_t, amdgpu_device_block_pool_large_size, 0,
          "Size in bytes of a large device block in the pool. Must be a power "
          "of two or 0 for the default.");
IREE_FLAG(int64_t, amdgpu_device_block_pool_large_capacity, 0,
          "Initial large block pool block allocation count in blocks or 0 for "
          "the default.");

IREE_FLAG(bool, amdgpu_preallocate_pools, true,
          "Preallocates a reasonable number of resources in pools to reduce "
          "initial execution latency.");

IREE_FLAG(bool, amdgpu_trace_execution, false,
          "Enables dispatch-level tracing (if device instrumentation is "
          "compiled in).");

IREE_FLAG(bool, amdgpu_exclusive_execution, false,
          "Forces queues to run one entry at a time instead of overlapping or "
          "aggressively scheduling queue entries out-of-order.");

IREE_FLAG(int64_t, amdgpu_wait_active_for_ns, 0,
          "Uses HSA_WAIT_STATE_ACTIVE for up to duration before switching to "
          "HSA_WAIT_STATE_BLOCKED. >0 will increase CPU usage in cases where "
          "the waits are long and decrease latency in cases where "
          "the waits are short.");

static iree_status_t iree_hal_amdgpu_driver_factory_enumerate(
    void* self, iree_host_size_t* out_driver_info_count,
    const iree_hal_driver_info_t** out_driver_infos) {
  static const iree_hal_driver_info_t default_driver_info = {
      .driver_name = IREE_SVL("amdgpu"),
      .full_name = IREE_SVL("AMD GPU Driver (HSA/ROCR)"),
  };
  *out_driver_info_count = 1;
  *out_driver_infos = &default_driver_info;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_driver_factory_try_create(
    void* self, iree_string_view_t driver_name, iree_allocator_t host_allocator,
    iree_hal_driver_t** out_driver) {
  if (!iree_string_view_equal(driver_name, IREE_SV("amdgpu"))) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "no driver '%.*s' is provided by this factory",
                            (int)driver_name.size, driver_name.data);
  }

  iree_hal_amdgpu_driver_options_t options;
  iree_hal_amdgpu_driver_options_initialize(&options);
  iree_hal_amdgpu_logical_device_options_t* device_options =
      &options.default_device_options;

  options.libhsa_search_paths = FLAG_amdgpu_libhsa_search_path_list();

  if (FLAG_amdgpu_host_block_pool_small_size) {
    device_options->host_block_pools.small.block_size =
        FLAG_amdgpu_host_block_pool_small_size;
  }
  if (FLAG_amdgpu_host_block_pool_large_size) {
    device_options->host_block_pools.large.block_size =
        FLAG_amdgpu_host_block_pool_large_size;
  }

  if (FLAG_amdgpu_device_block_pool_small_size) {
    device_options->device_block_pools.small.block_size =
        FLAG_amdgpu_device_block_pool_small_size;
  }
  if (FLAG_amdgpu_device_block_pool_small_capacity) {
    device_options->device_block_pools.small.initial_capacity =
        FLAG_amdgpu_device_block_pool_small_capacity;
  }
  if (FLAG_amdgpu_device_block_pool_large_size) {
    device_options->device_block_pools.large.block_size =
        FLAG_amdgpu_device_block_pool_large_size;
  }
  if (FLAG_amdgpu_device_block_pool_large_capacity) {
    device_options->device_block_pools.large.initial_capacity =
        FLAG_amdgpu_device_block_pool_large_capacity;
  }

  device_options->preallocate_pools = FLAG_amdgpu_preallocate_pools;

  device_options->trace_execution = FLAG_amdgpu_trace_execution;

  device_options->exclusive_execution = FLAG_amdgpu_exclusive_execution;

  device_options->wait_active_for_ns = FLAG_amdgpu_wait_active_for_ns;

  iree_status_t status = iree_hal_amdgpu_driver_create(
      driver_name, &options, host_allocator, out_driver);

  return status;
}

IREE_API_EXPORT iree_status_t
iree_hal_amdgpu_driver_module_register(iree_hal_driver_registry_t* registry) {
  static const iree_hal_driver_factory_t factory = {
      .self = NULL,
      .enumerate = iree_hal_amdgpu_driver_factory_enumerate,
      .try_create = iree_hal_amdgpu_driver_factory_try_create,
  };
  return iree_hal_driver_registry_register_factory(registry, &factory);
}

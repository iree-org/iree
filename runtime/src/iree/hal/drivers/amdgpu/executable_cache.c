// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/executable_cache.h"

#include "iree/hal/drivers/amdgpu/executable.h"
#include "iree/hal/drivers/amdgpu/util/topology.h"

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_executable_cache_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_amdgpu_executable_cache_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  const iree_hal_amdgpu_libhsa_t* libhsa;
  const iree_hal_amdgpu_topology_t* topology;
} iree_hal_amdgpu_executable_cache_t;

static const iree_hal_executable_cache_vtable_t
    iree_hal_amdgpu_executable_cache_vtable;

static iree_hal_amdgpu_executable_cache_t*
iree_hal_amdgpu_executable_cache_cast(iree_hal_executable_cache_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_amdgpu_executable_cache_vtable);
  return (iree_hal_amdgpu_executable_cache_t*)base_value;
}

iree_status_t iree_hal_amdgpu_executable_cache_create(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    const iree_hal_amdgpu_topology_t* topology, iree_string_view_t identifier,
    iree_allocator_t host_allocator,
    iree_hal_executable_cache_t** out_executable_cache) {
  IREE_ASSERT_ARGUMENT(out_executable_cache);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_executable_cache = NULL;

  // This should have been checked earlier but we do it here in case the user is
  // bypassing that code.
  IREE_ASSERT_GE(topology->gpu_agent_count, 1);
  if (IREE_UNLIKELY(topology->gpu_agent_count == 0)) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                             "topology must have at least one GPU device"));
  }

  iree_hal_amdgpu_executable_cache_t* executable_cache = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*executable_cache),
                                (void**)&executable_cache));
  iree_hal_resource_initialize(&iree_hal_amdgpu_executable_cache_vtable,
                               &executable_cache->resource);
  executable_cache->host_allocator = host_allocator;
  executable_cache->libhsa = libhsa;
  executable_cache->topology = topology;

  *out_executable_cache = (iree_hal_executable_cache_t*)executable_cache;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_amdgpu_executable_cache_destroy(
    iree_hal_executable_cache_t* base_executable_cache) {
  iree_hal_amdgpu_executable_cache_t* executable_cache =
      iree_hal_amdgpu_executable_cache_cast(base_executable_cache);
  iree_allocator_t host_allocator = executable_cache->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_free(host_allocator, executable_cache);

  IREE_TRACE_ZONE_END(z0);
}

static bool iree_hal_amdgpu_executable_cache_can_prepare_format(
    iree_hal_executable_cache_t* base_executable_cache,
    iree_hal_executable_caching_mode_t caching_mode,
    iree_string_view_t executable_format) {
  iree_hal_amdgpu_executable_cache_t* executable_cache =
      iree_hal_amdgpu_executable_cache_cast(base_executable_cache);
  bool is_supported = false;
  IREE_IGNORE_ERROR(iree_hal_amdgpu_executable_format_supported(
      executable_cache->libhsa, executable_cache->topology->gpu_agents[0],
      executable_format, &is_supported, /*out_isa=*/NULL));
  return is_supported;
}

static iree_status_t iree_hal_amdgpu_executable_cache_prepare_executable(
    iree_hal_executable_cache_t* base_executable_cache,
    const iree_hal_executable_params_t* executable_params,
    iree_hal_executable_t** out_executable) {
  iree_hal_amdgpu_executable_cache_t* executable_cache =
      iree_hal_amdgpu_executable_cache_cast(base_executable_cache);
  return iree_hal_amdgpu_executable_create(
      executable_cache->libhsa, executable_cache->topology, executable_params,
      executable_cache->host_allocator, out_executable);
}

static const iree_hal_executable_cache_vtable_t
    iree_hal_amdgpu_executable_cache_vtable = {
        .destroy = iree_hal_amdgpu_executable_cache_destroy,
        .can_prepare_format =
            iree_hal_amdgpu_executable_cache_can_prepare_format,
        .prepare_executable =
            iree_hal_amdgpu_executable_cache_prepare_executable,
};

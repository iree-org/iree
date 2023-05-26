// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/webgpu/nop_executable_cache.h"

#include <stdbool.h>
#include <stddef.h>

#include "experimental/webgpu/executable.h"
#include "iree/base/api.h"
#include "iree/base/tracing.h"

typedef struct iree_hal_webgpu_nop_executable_cache_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  WGPUDevice device;
} iree_hal_webgpu_nop_executable_cache_t;

extern const iree_hal_executable_cache_vtable_t
    iree_hal_webgpu_nop_executable_cache_vtable;

static iree_hal_webgpu_nop_executable_cache_t*
iree_hal_webgpu_nop_executable_cache_cast(
    iree_hal_executable_cache_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value,
                       &iree_hal_webgpu_nop_executable_cache_vtable);
  return (iree_hal_webgpu_nop_executable_cache_t*)base_value;
}

iree_status_t iree_hal_webgpu_nop_executable_cache_create(
    WGPUDevice device, iree_string_view_t identifier, iree_loop_t loop,
    iree_allocator_t host_allocator,
    iree_hal_executable_cache_t** out_executable_cache) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_executable_cache);
  *out_executable_cache = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_webgpu_nop_executable_cache_t* executable_cache = NULL;
  iree_status_t status = iree_allocator_malloc(
      host_allocator, sizeof(*executable_cache), (void**)&executable_cache);
  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&iree_hal_webgpu_nop_executable_cache_vtable,
                                 &executable_cache->resource);
    executable_cache->host_allocator = host_allocator;
    executable_cache->device = device;
    *out_executable_cache = (iree_hal_executable_cache_t*)executable_cache;
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_webgpu_nop_executable_cache_destroy(
    iree_hal_executable_cache_t* base_executable_cache) {
  iree_hal_webgpu_nop_executable_cache_t* executable_cache =
      iree_hal_webgpu_nop_executable_cache_cast(base_executable_cache);
  iree_allocator_t host_allocator = executable_cache->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_free(host_allocator, executable_cache);

  IREE_TRACE_ZONE_END(z0);
}

static bool iree_hal_webgpu_nop_executable_cache_can_prepare_format(
    iree_hal_executable_cache_t* base_executable_cache,
    iree_hal_executable_caching_mode_t caching_mode,
    iree_string_view_t executable_format) {
  // TODO(benvanik): allow SPIR-V `webgpu-spirv-fb` etc based on device support.
  return iree_string_view_equal(executable_format,
                                iree_make_cstring_view("webgpu-wgsl-fb"));
}

static iree_status_t iree_hal_webgpu_nop_executable_cache_prepare_executable(
    iree_hal_executable_cache_t* base_executable_cache,
    const iree_hal_executable_params_t* executable_params,
    iree_hal_executable_t** out_executable) {
  iree_hal_webgpu_nop_executable_cache_t* executable_cache =
      iree_hal_webgpu_nop_executable_cache_cast(base_executable_cache);
  return iree_hal_webgpu_executable_create(
      executable_cache->device, executable_params,
      executable_cache->host_allocator, out_executable);
}

const iree_hal_executable_cache_vtable_t
    iree_hal_webgpu_nop_executable_cache_vtable = {
        .destroy = iree_hal_webgpu_nop_executable_cache_destroy,
        .can_prepare_format =
            iree_hal_webgpu_nop_executable_cache_can_prepare_format,
        .prepare_executable =
            iree_hal_webgpu_nop_executable_cache_prepare_executable,
};

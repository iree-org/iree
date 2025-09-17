// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/cuda/nop_executable_cache.h"

#include <stdbool.h>
#include <stddef.h>

#include "iree/base/api.h"
#include "iree/base/tracing.h"
#include "iree/hal/drivers/cuda/native_executable.h"

typedef struct iree_hal_cuda_nop_executable_cache_t {
  // Abstract resource used for injecting reference counting and vtable;
  // must be at offset 0.
  iree_hal_resource_t resource;

  iree_allocator_t host_allocator;

  const iree_hal_cuda_dynamic_symbols_t* symbols;

  CUdevice device;
} iree_hal_cuda_nop_executable_cache_t;

static const iree_hal_executable_cache_vtable_t
    iree_hal_cuda_nop_executable_cache_vtable;

static iree_hal_cuda_nop_executable_cache_t*
iree_hal_cuda_nop_executable_cache_cast(
    iree_hal_executable_cache_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_cuda_nop_executable_cache_vtable);
  return (iree_hal_cuda_nop_executable_cache_t*)base_value;
}

iree_status_t iree_hal_cuda_nop_executable_cache_create(
    iree_string_view_t identifier,
    const iree_hal_cuda_dynamic_symbols_t* symbols, CUdevice device,
    iree_allocator_t host_allocator,
    iree_hal_executable_cache_t** out_executable_cache) {
  IREE_ASSERT_ARGUMENT(out_executable_cache);
  IREE_TRACE_ZONE_BEGIN(z0);

  *out_executable_cache = NULL;
  iree_hal_cuda_nop_executable_cache_t* executable_cache = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*executable_cache),
                                (void**)&executable_cache));

  iree_hal_resource_initialize(&iree_hal_cuda_nop_executable_cache_vtable,
                               &executable_cache->resource);
  executable_cache->host_allocator = host_allocator;
  executable_cache->symbols = symbols;
  executable_cache->device = device;

  *out_executable_cache = (iree_hal_executable_cache_t*)executable_cache;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_cuda_nop_executable_cache_destroy(
    iree_hal_executable_cache_t* base_executable_cache) {
  iree_hal_cuda_nop_executable_cache_t* executable_cache =
      iree_hal_cuda_nop_executable_cache_cast(base_executable_cache);
  iree_allocator_t host_allocator = executable_cache->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_free(host_allocator, executable_cache);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_cuda_nop_executable_cache_infer_format(
    iree_hal_executable_cache_t* base_executable_cache,
    iree_hal_executable_caching_mode_t caching_mode,
    iree_const_byte_span_t executable_data,
    iree_host_size_t executable_format_capacity, char* executable_format,
    iree_host_size_t* out_inferred_size) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = iree_hal_cuda_native_executable_infer_format(
      executable_data, executable_format_capacity, executable_format,
      out_inferred_size);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static bool iree_hal_cuda_nop_executable_cache_can_prepare_format(
    iree_hal_executable_cache_t* base_executable_cache,
    iree_hal_executable_caching_mode_t caching_mode,
    iree_string_view_t executable_format) {
  return iree_string_view_equal(executable_format,
                                iree_make_cstring_view("PTXE"));
}

static iree_status_t iree_hal_cuda_nop_executable_cache_prepare_executable(
    iree_hal_executable_cache_t* base_executable_cache,
    const iree_hal_executable_params_t* executable_params,
    iree_hal_executable_t** out_executable) {
  iree_hal_cuda_nop_executable_cache_t* executable_cache =
      iree_hal_cuda_nop_executable_cache_cast(base_executable_cache);
  return iree_hal_cuda_native_executable_create(
      executable_cache->symbols, executable_cache->device, executable_params,
      executable_cache->host_allocator, out_executable);
}

static const iree_hal_executable_cache_vtable_t
    iree_hal_cuda_nop_executable_cache_vtable = {
        .destroy = iree_hal_cuda_nop_executable_cache_destroy,
        .infer_format = iree_hal_cuda_nop_executable_cache_infer_format,
        .can_prepare_format =
            iree_hal_cuda_nop_executable_cache_can_prepare_format,
        .prepare_executable =
            iree_hal_cuda_nop_executable_cache_prepare_executable,
};

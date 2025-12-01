// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/hsa/nop_executable_cache.h"

#include <stddef.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/tracing.h"

typedef struct iree_hal_hsa_nop_executable_cache_t {
  iree_hal_resource_t resource;
  const iree_hal_hsa_dynamic_symbols_t* symbols;
  iree_allocator_t host_allocator;
  iree_string_view_t identifier;
} iree_hal_hsa_nop_executable_cache_t;

static const iree_hal_executable_cache_vtable_t
    iree_hal_hsa_nop_executable_cache_vtable;

static iree_hal_hsa_nop_executable_cache_t*
iree_hal_hsa_nop_executable_cache_cast(iree_hal_executable_cache_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_hsa_nop_executable_cache_vtable);
  return (iree_hal_hsa_nop_executable_cache_t*)base_value;
}

iree_status_t iree_hal_hsa_nop_executable_cache_create(
    iree_string_view_t identifier,
    const iree_hal_hsa_dynamic_symbols_t* symbols,
    iree_allocator_t host_allocator,
    iree_hal_executable_cache_t** out_executable_cache) {
  IREE_ASSERT_ARGUMENT(out_executable_cache);
  *out_executable_cache = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_hsa_nop_executable_cache_t* executable_cache = NULL;
  iree_host_size_t total_size =
      iree_sizeof_struct(*executable_cache) + identifier.size;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, total_size,
                                (void**)&executable_cache));

  iree_hal_resource_initialize(&iree_hal_hsa_nop_executable_cache_vtable,
                               &executable_cache->resource);
  executable_cache->symbols = symbols;
  executable_cache->host_allocator = host_allocator;
  iree_string_view_append_to_buffer(
      identifier, &executable_cache->identifier,
      (char*)executable_cache + iree_sizeof_struct(*executable_cache));

  *out_executable_cache = (iree_hal_executable_cache_t*)executable_cache;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_hsa_nop_executable_cache_destroy(
    iree_hal_executable_cache_t* base_executable_cache) {
  iree_hal_hsa_nop_executable_cache_t* executable_cache =
      iree_hal_hsa_nop_executable_cache_cast(base_executable_cache);
  iree_allocator_t host_allocator = executable_cache->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_free(host_allocator, executable_cache);

  IREE_TRACE_ZONE_END(z0);
}

static bool iree_hal_hsa_nop_executable_cache_can_prepare_format(
    iree_hal_executable_cache_t* base_executable_cache,
    iree_hal_executable_caching_mode_t caching_mode,
    iree_string_view_t executable_format) {
  // We support HSACO format (same as HIP).
  return iree_string_view_equal(executable_format, IREE_SV("rocm-hsaco-fb"));
}

static iree_status_t iree_hal_hsa_nop_executable_cache_prepare_executable(
    iree_hal_executable_cache_t* base_executable_cache,
    const iree_hal_executable_params_t* executable_params,
    iree_hal_executable_t** out_executable) {
  // No-op cache doesn't cache anything.
  // The actual loading is done by the native_executable_create.
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "nop executable cache does not support direct "
                          "executable preparation; use native_executable_create");
}

static const iree_hal_executable_cache_vtable_t
    iree_hal_hsa_nop_executable_cache_vtable = {
        .destroy = iree_hal_hsa_nop_executable_cache_destroy,
        .can_prepare_format = iree_hal_hsa_nop_executable_cache_can_prepare_format,
        .prepare_executable = iree_hal_hsa_nop_executable_cache_prepare_executable,
};


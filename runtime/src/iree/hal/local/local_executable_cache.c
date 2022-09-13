// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/local/local_executable_cache.h"

#include <stdbool.h>
#include <stddef.h>

#include "iree/base/tracing.h"

typedef struct iree_hal_local_executable_cache_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  iree_string_view_t identifier;
  iree_host_size_t worker_capacity;
  iree_host_size_t loader_count;
  iree_hal_executable_loader_t* loaders[];
} iree_hal_local_executable_cache_t;

static const iree_hal_executable_cache_vtable_t
    iree_hal_local_executable_cache_vtable;

static iree_hal_local_executable_cache_t* iree_hal_local_executable_cache_cast(
    iree_hal_executable_cache_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_local_executable_cache_vtable);
  return (iree_hal_local_executable_cache_t*)base_value;
}

iree_status_t iree_hal_local_executable_cache_create(
    iree_string_view_t identifier, iree_host_size_t worker_capacity,
    iree_host_size_t loader_count, iree_hal_executable_loader_t** loaders,
    iree_allocator_t host_allocator,
    iree_hal_executable_cache_t** out_executable_cache) {
  IREE_ASSERT_ARGUMENT(!loader_count || loaders);
  IREE_ASSERT_ARGUMENT(out_executable_cache);
  *out_executable_cache = NULL;

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_local_executable_cache_t* executable_cache = NULL;
  iree_host_size_t total_size =
      sizeof(*executable_cache) +
      loader_count * sizeof(*executable_cache->loaders) + identifier.size;
  iree_status_t status = iree_allocator_malloc(host_allocator, total_size,
                                               (void**)&executable_cache);
  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&iree_hal_local_executable_cache_vtable,
                                 &executable_cache->resource);
    executable_cache->host_allocator = host_allocator;
    iree_string_view_append_to_buffer(
        identifier, &executable_cache->identifier,
        (char*)executable_cache + total_size - identifier.size);
    executable_cache->worker_capacity = worker_capacity;

    executable_cache->loader_count = loader_count;
    for (iree_host_size_t i = 0; i < executable_cache->loader_count; ++i) {
      executable_cache->loaders[i] = loaders[i];
      iree_hal_executable_loader_retain(executable_cache->loaders[i]);
    }

    *out_executable_cache = (iree_hal_executable_cache_t*)executable_cache;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_local_executable_cache_destroy(
    iree_hal_executable_cache_t* base_executable_cache) {
  iree_hal_local_executable_cache_t* executable_cache =
      iree_hal_local_executable_cache_cast(base_executable_cache);
  iree_allocator_t host_allocator = executable_cache->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  for (iree_host_size_t i = 0; i < executable_cache->loader_count; ++i) {
    iree_hal_executable_loader_release(executable_cache->loaders[i]);
  }
  iree_allocator_free(host_allocator, executable_cache);

  IREE_TRACE_ZONE_END(z0);
}

static bool iree_hal_local_executable_cache_can_prepare_format(
    iree_hal_executable_cache_t* base_executable_cache,
    iree_hal_executable_caching_mode_t caching_mode,
    iree_string_view_t executable_format) {
  iree_hal_local_executable_cache_t* executable_cache =
      iree_hal_local_executable_cache_cast(base_executable_cache);
  for (iree_host_size_t i = 0; i < executable_cache->loader_count; ++i) {
    if (iree_hal_executable_loader_query_support(
            executable_cache->loaders[i], caching_mode, executable_format)) {
      return true;
    }
  }
  return false;
}

static iree_status_t iree_hal_local_executable_cache_prepare_executable(
    iree_hal_executable_cache_t* base_executable_cache,
    const iree_hal_executable_params_t* executable_params,
    iree_hal_executable_t** out_executable) {
  iree_hal_local_executable_cache_t* executable_cache =
      iree_hal_local_executable_cache_cast(base_executable_cache);
  for (iree_host_size_t i = 0; i < executable_cache->loader_count; ++i) {
    if (!iree_hal_executable_loader_query_support(
            executable_cache->loaders[i], executable_params->caching_mode,
            executable_params->executable_format)) {
      // Loader definitely can't handle the executable; no use trying so skip.
      continue;
    }
    // The loader _may_ handle the executable; if the specific executable is not
    // supported then the try will fail with IREE_STATUS_CANCELLED and we should
    // continue trying other loaders.
    iree_status_t status = iree_hal_executable_loader_try_load(
        executable_cache->loaders[i], executable_params,
        executable_cache->worker_capacity, out_executable);
    if (iree_status_is_ok(status)) {
      // Executable was successfully loaded.
      return status;
    } else if (!iree_status_is_cancelled(status)) {
      // Error beyond just the try failing due to unsupported formats.
      return status;
    }
    iree_status_ignore(status);
  }
  return iree_make_status(
      IREE_STATUS_NOT_FOUND,
      "no executable loader registered for the given executable format '%.*s'",
      (int)executable_params->executable_format.size,
      executable_params->executable_format.data);
}

static const iree_hal_executable_cache_vtable_t
    iree_hal_local_executable_cache_vtable = {
        .destroy = iree_hal_local_executable_cache_destroy,
        .can_prepare_format =
            iree_hal_local_executable_cache_can_prepare_format,
        .prepare_executable =
            iree_hal_local_executable_cache_prepare_executable,
};

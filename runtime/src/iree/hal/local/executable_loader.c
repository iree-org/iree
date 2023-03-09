// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/local/executable_loader.h"

#include "iree/base/tracing.h"

iree_status_t iree_hal_executable_import_provider_try_resolve(
    const iree_hal_executable_import_provider_t import_provider,
    iree_host_size_t count, const char* const* symbol_names, void** out_fn_ptrs,
    void** out_fn_contexts,
    iree_hal_executable_import_resolution_t* out_resolution) {
  if (!count) return iree_ok_status();
  IREE_ASSERT_ARGUMENT(symbol_names);
  IREE_ASSERT_ARGUMENT(out_fn_ptrs);
  IREE_ASSERT_ARGUMENT(out_fn_contexts);
  if (out_resolution) *out_resolution = 0;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE(z0, count);

  // It's fine for there to be no registered provider if all symbols are
  // optional. This is a special case for NULL import providers.
  if (import_provider.resolve == NULL) {
    bool any_required = false;
    for (iree_host_size_t i = 0; i < count; ++i) {
      if (!iree_hal_executable_import_is_optional(symbol_names[i])) {
        any_required = true;
        break;
      }
    }
    if (any_required) {
      return iree_make_status(IREE_STATUS_UNAVAILABLE,
                              "no import provider registered for resolving "
                              "required executable imports");
    } else {
      // No required imports so a NULL provider is fine.
      return iree_ok_status();
    }
  }

  iree_status_t status =
      import_provider.resolve(import_provider.self, count, symbol_names,
                              out_fn_ptrs, out_fn_contexts, out_resolution);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_executable_loader_initialize(
    const void* vtable, iree_hal_executable_import_provider_t import_provider,
    iree_hal_executable_loader_t* out_base_loader) {
  iree_atomic_ref_count_init(&out_base_loader->ref_count);
  out_base_loader->vtable = vtable;
  out_base_loader->import_provider = import_provider;
}

void iree_hal_executable_loader_retain(
    iree_hal_executable_loader_t* executable_loader) {
  if (IREE_LIKELY(executable_loader)) {
    iree_atomic_ref_count_inc(&executable_loader->ref_count);
  }
}

void iree_hal_executable_loader_release(
    iree_hal_executable_loader_t* executable_loader) {
  if (IREE_LIKELY(executable_loader) &&
      iree_atomic_ref_count_dec(&executable_loader->ref_count) == 1) {
    executable_loader->vtable->destroy(executable_loader);
  }
}

bool iree_hal_executable_loader_query_support(
    iree_hal_executable_loader_t* executable_loader,
    iree_hal_executable_caching_mode_t caching_mode,
    iree_string_view_t executable_format) {
  IREE_ASSERT_ARGUMENT(executable_loader);
  return executable_loader->vtable->query_support(
      executable_loader, caching_mode, executable_format);
}

bool iree_hal_query_any_executable_loader_support(
    iree_host_size_t loader_count, iree_hal_executable_loader_t** loaders,
    iree_hal_executable_caching_mode_t caching_mode,
    iree_string_view_t executable_format) {
  IREE_ASSERT_ARGUMENT(loaders);
  for (iree_host_size_t i = 0; i < loader_count; ++i) {
    if (iree_hal_executable_loader_query_support(loaders[i], caching_mode,
                                                 executable_format)) {
      return true;
    }
  }
  return false;
}

iree_status_t iree_hal_executable_loader_try_load(
    iree_hal_executable_loader_t* executable_loader,
    const iree_hal_executable_params_t* executable_params,
    iree_host_size_t worker_capacity, iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(executable_loader);
  IREE_ASSERT_ARGUMENT(executable_params);
  IREE_ASSERT_ARGUMENT(!executable_params->pipeline_layout_count ||
                       executable_params->pipeline_layouts);
  IREE_ASSERT_ARGUMENT(!executable_params->executable_data.data_length ||
                       executable_params->executable_data.data);
  IREE_ASSERT_ARGUMENT(out_executable);
  return executable_loader->vtable->try_load(
      executable_loader, executable_params, worker_capacity, out_executable);
}

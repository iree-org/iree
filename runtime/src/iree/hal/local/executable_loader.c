// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/local/executable_loader.h"

#if defined(IREE_HAL_EXECUTABLE_IMPORT_PROVIDER_DEFAULT_FN)

// Defined by the user and linked in to the binary:
extern iree_hal_executable_import_provider_t
IREE_HAL_EXECUTABLE_IMPORT_PROVIDER_DEFAULT_FN(void);

iree_hal_executable_import_provider_t
iree_hal_executable_import_provider_default(void) {
  return IREE_HAL_EXECUTABLE_IMPORT_PROVIDER_DEFAULT_FN();
}

#else
iree_hal_executable_import_provider_t
iree_hal_executable_import_provider_default(void) {
  return iree_hal_executable_import_provider_null();
}
#endif  // IREE_HAL_EXECUTABLE_IMPORT_PROVIDER_DEFAULT_FN

iree_status_t iree_hal_executable_import_provider_resolve(
    const iree_hal_executable_import_provider_t import_provider,
    iree_string_view_t symbol_name, void** out_fn_ptr, void** out_fn_context) {
  IREE_ASSERT_ARGUMENT(out_fn_ptr);
  IREE_ASSERT_ARGUMENT(out_fn_context);
  *out_fn_ptr = NULL;
  *out_fn_context = NULL;

  // A `?` suffix indicates the symbol is weakly linked and can be NULL.
  bool is_weak = false;
  if (iree_string_view_ends_with(symbol_name, iree_make_cstring_view("?"))) {
    is_weak = true;
    symbol_name = iree_string_view_substr(symbol_name, 0, symbol_name.size - 1);
  }

  // Note that it's fine for there to be no registered provider if all symbols
  // are weak.
  if (import_provider.resolve == NULL) {
    if (is_weak) return iree_ok_status();
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "no import provider registered for resolving "
                            "executable imports (while try to resolve %.*s)",
                            (int)symbol_name.size, symbol_name.data);
  }

  iree_status_t status = import_provider.resolve(
      import_provider.self, symbol_name, out_fn_ptr, out_fn_context);
  if (!iree_status_is_ok(status) && is_weak) {
    status = iree_status_ignore(status);  // ok to fail on weak symbols
  }

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

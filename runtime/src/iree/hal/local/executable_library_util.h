// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_LOCAL_EXECUTABLE_LIBRARY_UTIL_H_
#define IREE_HAL_LOCAL_EXECUTABLE_LIBRARY_UTIL_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/local/executable_library.h"
#include "iree/hal/local/executable_loader.h"

// Verifies the |library| matches the |executable_params|.
iree_status_t iree_hal_executable_library_verify(
    const iree_hal_executable_params_t* executable_params,
    const iree_hal_executable_library_v0_t* library);

// Allocates and resolves import function and context storage on |environment|
// using |import_provider|. All imports will be called through |import_thunk|.
iree_status_t iree_hal_executable_library_initialize_imports(
    iree_hal_executable_environment_v0_t* environment,
    const iree_hal_executable_import_provider_t import_provider,
    const iree_hal_executable_import_table_v0_t* import_table,
    iree_hal_executable_import_thunk_v0_t import_thunk,
    iree_allocator_t host_allocator);

// Frees environment imports previously allocated with
// iree_hal_executable_library_allocate_imports. Must only be called after all
// existing references to the environment have been dropped.
void iree_hal_executable_library_deinitialize_imports(
    iree_hal_executable_environment_v0_t* environment,
    iree_allocator_t host_allocator);

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION
iree_zone_id_t iree_hal_executable_library_call_zone_begin(
    iree_string_view_t executable_identifier,
    const iree_hal_executable_library_v0_t* library, iree_host_size_t ordinal);
#define IREE_HAL_EXECUTABLE_LIBRARY_CALL_TRACE_ZONE_BEGIN(              \
    zone_id, executable_identifier, library, ordinal)                   \
  iree_zone_id_t zone_id = iree_hal_executable_library_call_zone_begin( \
      executable_identifier, library, ordinal)
#else
#define IREE_HAL_EXECUTABLE_LIBRARY_CALL_TRACE_ZONE_BEGIN( \
    zone_id, executable_identifier, library, ordinal)      \
  iree_zone_id_t zone_id = 0;                              \
  (void)zone_id;
#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION
iree_status_t iree_hal_executable_library_setup_tracing(
    const iree_hal_executable_library_v0_t* library,
    iree_allocator_t host_allocator, tracy_file_mapping** out_file_mapping);
#define IREE_HAL_EXECUTABLE_LIBRARY_SETUP_TRACING(library, host_allocator, \
                                                  out_file_mapping)        \
  iree_hal_executable_library_setup_tracing(library, host_allocator,       \
                                            out_file_mapping)
#else
#define IREE_HAL_EXECUTABLE_LIBRARY_SETUP_TRACING(library, host_allocator, \
                                                  out_file_mapping)        \
  iree_ok_status()
#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION

#endif  // IREE_HAL_LOCAL_EXECUTABLE_LIBRARY_UTIL_H_

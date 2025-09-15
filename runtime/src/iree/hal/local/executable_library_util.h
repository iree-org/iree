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

// Returns the number of exports in the library.
iree_host_size_t iree_hal_executable_library_export_count(
    const iree_hal_executable_library_v0_t* library);

// Returns information about an export at the given ordinal.
iree_status_t iree_hal_executable_library_export_info(
    const iree_hal_executable_library_v0_t* library,
    iree_hal_executable_export_ordinal_t export_ordinal,
    iree_hal_executable_export_info_t* out_info);

// Populates parameter information for an export.
iree_status_t iree_hal_executable_library_export_parameters(
    const iree_hal_executable_library_v0_t* library,
    iree_hal_executable_export_ordinal_t export_ordinal,
    iree_host_size_t capacity,
    iree_hal_executable_export_parameter_t* out_parameters);

// Looks up an export ordinal by name.
iree_status_t iree_hal_executable_library_lookup_export_by_name(
    const iree_hal_executable_library_v0_t* library, iree_string_view_t name,
    iree_hal_executable_export_ordinal_t* out_export_ordinal);

#if defined(IREE_HAL_EXECUTABLE_LIBRARY_CALL_HOOK)
#if !IREE_HAVE_ATTRIBUTE_WEAK
#error IREE_HAL_EXECUTABLE_LIBRARY_CALL_HOOK requires toolchain support for weak symbols.
#endif
IREE_ATTRIBUTE_WEAK void iree_hal_executable_library_call_hook_begin(
    iree_string_view_t executable_identifier,
    const iree_hal_executable_library_v0_t* library, iree_host_size_t ordinal);
IREE_ATTRIBUTE_WEAK void iree_hal_executable_library_call_hook_end(
    iree_string_view_t executable_identifier,
    const iree_hal_executable_library_v0_t* library, iree_host_size_t ordinal);
#define IREE_HAL_EXECUTABLE_LIBRARY_CALL_HOOK_BEGIN(executable_identifier, \
                                                    library, ordinal)      \
  do {                                                                     \
    if (iree_hal_executable_library_call_hook_begin /* weak symbol */) {   \
      iree_hal_executable_library_call_hook_begin(executable_identifier,   \
                                                  library, ordinal);       \
    }                                                                      \
  } while (0)
#define IREE_HAL_EXECUTABLE_LIBRARY_CALL_HOOK_END(executable_identifier, \
                                                  library, ordinal)      \
  do {                                                                   \
    if (iree_hal_executable_library_call_hook_end /* weak symbol */) {   \
      iree_hal_executable_library_call_hook_end(executable_identifier,   \
                                                library, ordinal);       \
    }                                                                    \
  } while (0)
#else  // defined(IREE_HAL_EXECUTABLE_LIBRARY_CALL_HOOK)
#define IREE_HAL_EXECUTABLE_LIBRARY_CALL_HOOK_BEGIN(executable_identifier, \
                                                    library, ordinal)
#define IREE_HAL_EXECUTABLE_LIBRARY_CALL_HOOK_END(executable_identifier, \
                                                  library, ordinal)
#endif  // defined(IREE_HAL_EXECUTABLE_LIBRARY_CALL_HOOK)

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION

// Publishes all source files in the library to the tracing provider, if any.
void iree_hal_executable_library_publish_source_files(
    const iree_hal_executable_library_v0_t* library);

iree_zone_id_t iree_hal_executable_library_call_zone_begin(
    iree_string_view_t executable_identifier,
    const iree_hal_executable_library_v0_t* library, iree_host_size_t ordinal);
#define IREE_HAL_EXECUTABLE_LIBRARY_CALL_TRACE_ZONE_BEGIN(              \
    zone_id, executable_identifier, library, ordinal)                   \
  iree_zone_id_t zone_id = iree_hal_executable_library_call_zone_begin( \
      executable_identifier, library, ordinal)

#else

static inline void iree_hal_executable_library_publish_source_files(
    const iree_hal_executable_library_v0_t* library) {}

#define IREE_HAL_EXECUTABLE_LIBRARY_CALL_TRACE_ZONE_BEGIN( \
    zone_id, executable_identifier, library, ordinal)      \
  iree_zone_id_t zone_id = 0;                              \
  (void)zone_id;
#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION

#endif  // IREE_HAL_LOCAL_EXECUTABLE_LIBRARY_UTIL_H_

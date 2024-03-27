// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/hip/dynamic_symbols.h"

#include <string.h>

#include "iree/base/api.h"
#include "iree/base/internal/dynamic_library.h"
#include "iree/base/internal/path.h"
#include "iree/base/target_platform.h"
#include "iree/hal/drivers/hip/status_util.h"

//===----------------------------------------------------------------------===//
// HIP dynamic symbols
//===----------------------------------------------------------------------===//

static const char* iree_hal_hip_dylib_names[] = {
#if defined(IREE_PLATFORM_WINDOWS)
    "amdhip64.dll",
#else
    "libamdhip64.so",
#endif  // IREE_PLATFORM_WINDOWS
};

// Resolves all HIP dynamic symbols in `dynamic_symbol_tables.h`
static iree_status_t iree_hal_hip_dynamic_symbols_resolve_all(
    iree_hal_hip_dynamic_symbols_t* syms) {
#define IREE_HAL_HIP_REQUIRED_PFN_DECL(hip_symbol_name, ...) \
  {                                                          \
    static const char* name = #hip_symbol_name;              \
    IREE_RETURN_IF_ERROR(iree_dynamic_library_lookup_symbol( \
        syms->dylib, name, (void**)&syms->hip_symbol_name)); \
  }
#define IREE_HAL_HIP_REQUIRED_PFN_STR_DECL(hip_symbol_name, ...) \
  IREE_HAL_HIP_REQUIRED_PFN_DECL(hip_symbol_name, ...)
#define IREE_HAL_HIP_OPTIONAL_PFN_DECL(hip_symbol_name, ...) \
  {                                                          \
    static const char* name = #hip_symbol_name;              \
    IREE_IGNORE_ERROR(iree_dynamic_library_lookup_symbol(    \
        syms->dylib, name, (void**)&syms->hip_symbol_name)); \
  }
#include "iree/hal/drivers/hip/dynamic_symbol_tables.h"  // IWYU pragma: keep
#undef IREE_HAL_HIP_REQUIRED_PFN_DECL
#undef IREE_HAL_HIP_REQUIRED_PFN_STR_DECL
#undef IREE_HAL_HIP_OPTIONAL_PFN_DECL
  return iree_ok_status();
}

static bool iree_hal_hip_try_load_dylib(const char* file_path,
                                        iree_dynamic_library_flags_t flags,
                                        iree_allocator_t allocator,
                                        iree_string_builder_t* error_builder,
                                        iree_dynamic_library_t** out_library) {
  iree_status_t status = iree_dynamic_library_load_from_file(
      file_path, flags, allocator, out_library);
  if (iree_status_is_ok(status)) {
    return true;
  }

  char* buffer = NULL;
  iree_host_size_t length = 0;
  if (iree_status_to_string(status, &allocator, &buffer, &length)) {
    iree_status_ignore(iree_string_builder_append_format(
        error_builder, "\n  Tried: %s\n    %.*s", file_path, (int)length,
        buffer));
    iree_allocator_free(allocator, buffer);
  }

  iree_status_ignore(status);
  return false;
}

iree_status_t iree_hal_hip_dynamic_symbols_initialize(
    iree_allocator_t host_allocator, iree_host_size_t hip_lib_search_path_count,
    const iree_string_view_t* hip_lib_search_paths,
    iree_hal_hip_dynamic_symbols_t* out_syms) {
  IREE_ASSERT_ARGUMENT(out_syms);
  IREE_TRACE_ZONE_BEGIN(z0);

  memset(out_syms, 0, sizeof(*out_syms));

  // Load the library.
  bool loaded_one = false;
  iree_status_t status = iree_ok_status();
  iree_string_builder_t error_builder;
  iree_string_builder_initialize(iree_allocator_system(), &error_builder);

  if (hip_lib_search_path_count == 0) {
    // If no explicit search path, then have the system try to find the library
    // by filename alone.
    for (iree_host_size_t i = 0;
         i < IREE_ARRAYSIZE(iree_hal_hip_dylib_names) && !loaded_one; ++i) {
      if (iree_hal_hip_try_load_dylib(
              iree_hal_hip_dylib_names[i], IREE_DYNAMIC_LIBRARY_FLAG_NONE,
              host_allocator, &error_builder, &out_syms->dylib)) {
        loaded_one = true;
      }
    }
  } else {
    // With an explicit path, try each entry individually.
    iree_string_builder_t path_builder;
    iree_string_builder_initialize(host_allocator, &path_builder);
    for (iree_host_size_t i = 0; i < hip_lib_search_path_count && !loaded_one;
         ++i) {
      iree_string_view_t path_entry = hip_lib_search_paths[i];
      iree_string_view_t file_prefix = iree_string_view_literal("file:");
      iree_string_builder_reset(&path_builder);
      if (iree_string_view_consume_prefix(&path_entry, file_prefix)) {
        // Load verbatim.
        status = iree_string_builder_append_string(&path_builder, path_entry);
        if (!iree_status_is_ok(status)) break;
        if (iree_hal_hip_try_load_dylib(
                path_builder.buffer, IREE_DYNAMIC_LIBRARY_FLAG_NONE,
                host_allocator, &error_builder, &out_syms->dylib)) {
          loaded_one = true;
        }
      } else {
        // Try each variant of a platform specific library name.
        for (iree_host_size_t j = 0;
             j < IREE_ARRAYSIZE(iree_hal_hip_dylib_names) && !loaded_one; ++j) {
          // Join the directory with a system specific library name.
          iree_string_view_t sep = iree_string_view_literal("/");
          status = iree_string_builder_append_string(&path_builder, path_entry);
          if (!iree_status_is_ok(status)) break;
          status = iree_string_builder_append_string(&path_builder, sep);
          if (!iree_status_is_ok(status)) break;
          status = iree_string_builder_append_string(
              &path_builder,
              iree_make_cstring_view(iree_hal_hip_dylib_names[j]));
          if (!iree_status_is_ok(status)) break;
          path_builder.size = iree_file_path_canonicalize(path_builder.buffer,
                                                          path_builder.size);
          if (iree_hal_hip_try_load_dylib(
                  path_builder.buffer, IREE_DYNAMIC_LIBRARY_FLAG_NONE,
                  host_allocator, &error_builder, &out_syms->dylib)) {
            loaded_one = true;
          }
        }
      }
      if (!iree_status_is_ok(status)) break;
    }
    iree_string_builder_deinitialize(&path_builder);
  }

  if (iree_status_is_ok(status)) {
    if (loaded_one) {
      status = iree_hal_hip_dynamic_symbols_resolve_all(out_syms);
    } else {
      iree_string_view_t error_detail =
          iree_string_builder_view(&error_builder);
      status = iree_make_status(
          IREE_STATUS_UNAVAILABLE,
          "HIP runtime library 'amdhip64.dll'/'libamdhip64.so' not available: "
          "please ensure installed and in dynamic library search path:%*.s",
          (int)error_detail.size, error_detail.data);
    }
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_hip_dynamic_symbols_deinitialize(out_syms);
  }
  iree_string_builder_deinitialize(&error_builder);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_hip_dynamic_symbols_deinitialize(
    iree_hal_hip_dynamic_symbols_t* syms) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_dynamic_library_release(syms->dylib);
  memset(syms, 0, sizeof(*syms));

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_hip_dynamic_symbols_append_path_to_builder(
    iree_hal_hip_dynamic_symbols_t* syms, iree_string_builder_t* out_path) {
  if (!syms->dylib) {
    return iree_make_status(IREE_STATUS_NOT_FOUND);
  }
  // Specific choice of symbol is not important.
  return iree_dynamic_library_get_symbol_path(syms->hipDeviceGet, out_path);
}

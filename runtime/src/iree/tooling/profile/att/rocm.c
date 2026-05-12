// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/profile/att/rocm.h"

#include <stdlib.h>

#include "iree/base/internal/path.h"
#include "iree/tooling/profile/att/util.h"

iree_string_view_t iree_profile_att_rocm_library_path_or_env(
    iree_string_view_t rocm_library_path) {
  if (!iree_string_view_is_empty(rocm_library_path)) {
    return rocm_library_path;
  }

  // Match the AMDGPU runtime capture path knobs so users can configure ROCm
  // discovery once and use the same environment for capture and decode.
  static const char* const env_names[] = {
      "IREE_HAL_AMDGPU_LIBAQLPROFILE_PATH",
      "IREE_HAL_AMDGPU_LIBHSA_PATH",
  };
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(env_names); ++i) {
    iree_string_view_t env_path = iree_make_cstring_view(getenv(env_names[i]));
    if (!iree_string_view_is_empty(env_path)) {
      return env_path;
    }
  }

  return iree_string_view_empty();
}

iree_status_t iree_profile_att_rocm_load_dynamic_library(
    iree_string_view_t rocm_library_path, const char* library_name,
    iree_allocator_t host_allocator, iree_dynamic_library_t** out_library) {
  *out_library = NULL;

  char* loaded_path = NULL;
  iree_status_t status = iree_ok_status();
  if (!iree_string_view_is_empty(rocm_library_path)) {
    if (iree_file_path_is_dynamic_library(rocm_library_path)) {
      iree_string_view_t basename = iree_file_path_basename(rocm_library_path);
      if (iree_string_view_equal(basename,
                                 iree_make_cstring_view(library_name))) {
        status = iree_profile_att_copy_cstring(rocm_library_path,
                                               host_allocator, &loaded_path);
      } else {
        status = iree_file_path_join(iree_file_path_dirname(rocm_library_path),
                                     iree_make_cstring_view(library_name),
                                     host_allocator, &loaded_path);
      }
    } else {
      status = iree_file_path_join(rocm_library_path,
                                   iree_make_cstring_view(library_name),
                                   host_allocator, &loaded_path);
    }
  } else {
    status = iree_profile_att_copy_cstring(iree_make_cstring_view(library_name),
                                           host_allocator, &loaded_path);
  }

  if (iree_status_is_ok(status)) {
    status = iree_dynamic_library_load_from_file(loaded_path,
                                                 IREE_DYNAMIC_LIBRARY_FLAG_NONE,
                                                 host_allocator, out_library);
  }

  iree_allocator_free(host_allocator, loaded_path);
  return status;
}

iree_status_t iree_profile_att_rocm_lookup_symbol(
    iree_dynamic_library_t* library, const char* symbol_name, void** out_fn) {
  return iree_dynamic_library_lookup_symbol(library, symbol_name, out_fn);
}

iree_status_t iree_profile_att_rocm_resolve_library_dir(
    void* symbol, iree_string_view_t rocm_library_path,
    iree_allocator_t host_allocator, char** out_directory) {
  *out_directory = NULL;
  if (!iree_string_view_is_empty(rocm_library_path) &&
      !iree_file_path_is_dynamic_library(rocm_library_path)) {
    return iree_profile_att_copy_cstring(rocm_library_path, host_allocator,
                                         out_directory);
  }
  if (!iree_string_view_is_empty(rocm_library_path)) {
    iree_string_view_t dirname = iree_file_path_dirname(rocm_library_path);
    if (!iree_string_view_is_empty(dirname)) {
      return iree_profile_att_copy_cstring(dirname, host_allocator,
                                           out_directory);
    }
  }

  iree_string_builder_t builder;
  iree_string_builder_initialize(host_allocator, &builder);
  iree_status_t status =
      iree_dynamic_library_append_symbol_path_to_builder(symbol, &builder);
  if (iree_status_is_ok(status)) {
    iree_string_view_t library_path = iree_string_builder_view(&builder);
    iree_string_view_t dirname = iree_file_path_dirname(library_path);
    status =
        iree_profile_att_copy_cstring(dirname, host_allocator, out_directory);
  }
  iree_string_builder_deinitialize(&builder);
  return status;
}

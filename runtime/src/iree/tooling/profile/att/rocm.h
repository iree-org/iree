// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOOLING_PROFILE_ATT_ROCM_H_
#define IREE_TOOLING_PROFILE_ATT_ROCM_H_

#include "iree/base/api.h"
#include "iree/base/internal/dynamic_library.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Returns |rocm_library_path| when set, otherwise the first ATT-compatible ROCm
// path environment variable. The returned view is borrowed from the caller or
// process environment and must not be stored across environment mutation.
iree_string_view_t iree_profile_att_rocm_library_path_or_env(
    iree_string_view_t rocm_library_path);

// Loads |library_name| from |rocm_library_path|, or from the platform loader
// search path when |rocm_library_path| is empty.
//
// If |rocm_library_path| names a dynamic library with a different basename,
// the target library is resolved beside it.
iree_status_t iree_profile_att_rocm_load_dynamic_library(
    iree_string_view_t rocm_library_path, const char* library_name,
    iree_allocator_t host_allocator, iree_dynamic_library_t** out_library);

// Looks up |symbol_name| in |library|.
iree_status_t iree_profile_att_rocm_lookup_symbol(
    iree_dynamic_library_t* library, const char* symbol_name, void** out_fn);

// Resolves the directory containing |symbol| or the explicit ROCm path.
// |out_directory| receives an allocator-owned NUL-terminated path. |symbol|
// must be non-NULL when |rocm_library_path| is empty.
iree_status_t iree_profile_att_rocm_resolve_library_dir(
    void* symbol, iree_string_view_t rocm_library_path,
    iree_allocator_t host_allocator, char** out_directory);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TOOLING_PROFILE_ATT_ROCM_H_

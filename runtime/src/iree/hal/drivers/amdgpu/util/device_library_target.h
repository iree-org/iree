// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_UTIL_DEVICE_LIBRARY_TARGET_H_
#define IREE_HAL_DRIVERS_AMDGPU_UTIL_DEVICE_LIBRARY_TARGET_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// AMDGPU Device-Library Targets
//===----------------------------------------------------------------------===//

// Candidate embedded device-library target string.
typedef struct iree_hal_amdgpu_device_library_target_candidate_t {
  // NUL-terminated target string storage.
  char storage[64];
  // Candidate target view pointing into |storage|.
  iree_string_view_t value;
} iree_hal_amdgpu_device_library_target_candidate_t;

// Device-library target candidates in descending specificity order.
typedef struct iree_hal_amdgpu_device_library_target_candidate_list_t {
  // Number of populated candidate entries.
  iree_host_size_t count;
  // Candidate entries from most-specific to least-specific.
  iree_hal_amdgpu_device_library_target_candidate_t values[4];
} iree_hal_amdgpu_device_library_target_candidate_list_t;

// Returns true when an embedded file architecture suffix matches |target| as a
// complete segment before any dot-separated binary suffix.
bool iree_hal_amdgpu_device_library_target_matches_file_arch(
    iree_string_view_t file_arch, iree_string_view_t target);

// Builds ordered device-library target candidates for an HSA ISA name.
iree_status_t iree_hal_amdgpu_device_library_target_candidates_from_isa(
    iree_string_view_t isa_name,
    iree_hal_amdgpu_device_library_target_candidate_list_t* out_candidates);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_UTIL_DEVICE_LIBRARY_TARGET_H_

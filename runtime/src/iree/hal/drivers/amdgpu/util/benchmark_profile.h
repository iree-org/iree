// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_UTIL_BENCHMARK_PROFILE_H_
#define IREE_HAL_DRIVERS_AMDGPU_UTIL_BENCHMARK_PROFILE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a profile sink that counts profile chunks without writing them.
iree_status_t iree_hal_amdgpu_benchmark_discard_profile_sink_create(
    iree_allocator_t host_allocator, iree_hal_profile_sink_t** out_sink);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_UTIL_BENCHMARK_PROFILE_H_

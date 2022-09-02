// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_LOCAL_LOCAL_EXECUTABLE_CACHE_H_
#define IREE_HAL_LOCAL_LOCAL_EXECUTABLE_CACHE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/local/executable_loader.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// TODO(benvanik): when we refactor executable caches this can become something
// more specialized; like nop_executable_cache (does nothing but pass through)
// or inproc_lru_executable_cache (simple in-memory LRU of recent executables).
//
// We can also set this up so they share storage. Ideally a JIT'ed executable in
// one device is the same JIT'ed executable in another, and in multi-tenant
// situations we're likely to want that isolation _and_ sharing.

iree_status_t iree_hal_local_executable_cache_create(
    iree_string_view_t identifier, iree_host_size_t worker_capacity,
    iree_host_size_t loader_count, iree_hal_executable_loader_t** loaders,
    iree_allocator_t host_allocator,
    iree_hal_executable_cache_t** out_executable_cache);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_LOCAL_LOCAL_EXECUTABLE_CACHE_H_

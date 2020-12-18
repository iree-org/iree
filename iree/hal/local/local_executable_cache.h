// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
    iree_string_view_t identifier, iree_host_size_t loader_count,
    iree_hal_executable_loader_t** loaders, iree_allocator_t host_allocator,
    iree_hal_executable_cache_t** out_executable_cache);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_LOCAL_LOCAL_EXECUTABLE_CACHE_H_

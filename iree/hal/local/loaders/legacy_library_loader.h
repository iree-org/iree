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

#ifndef IREE_HAL_LOCAL_LOADERS_LEGACY_LIBRARY_LOADER_H_
#define IREE_HAL_LOCAL_LOADERS_LEGACY_LIBRARY_LOADER_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/local/executable_loader.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates an executable loader that can load files from platform-supported
// dynamic libraries (such as .dylib on darwin, .so on linux, .dll on windows).
//
// This uses the legacy "dylib"-style format that will be deleted soon and is
// only a placeholder until the compiler can be switched to output
// iree_hal_executable_library_t-compatible files.
iree_status_t iree_hal_legacy_library_loader_create(
    iree_allocator_t host_allocator,
    iree_hal_executable_loader_t** out_executable_loader);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_LOCAL_LOADERS_LEGACY_LIBRARY_LOADER_H_

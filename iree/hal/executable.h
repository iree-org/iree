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

#ifndef IREE_HAL_EXECUTABLE_H_
#define IREE_HAL_EXECUTABLE_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/resource.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_device_s iree_hal_device_t;

//===----------------------------------------------------------------------===//
// iree_hal_executable_t
//===----------------------------------------------------------------------===//

// Handle to a loaded executable.
// Loading of executables routes through an executable cache, allowing for
// context-aware scoped caches. HAL implementations can use this to preserve
// JIT'ed executables across processes or reuse executables across device
// instances.
//
// Executables provide one or more entry points that can be dispatched via
// iree_hal_command_buffer_dispatch. Some entry points may represent the same
// computation but specialized in different ways such that the runtime can
// switch strategies and choose between them per-dispatch.
//
//
// Maps (roughly) to vkShaderModule + VkPipeline[].
typedef struct iree_hal_executable_s iree_hal_executable_t;

// Retains the given |executable| for the caller.
IREE_API_EXPORT void iree_hal_executable_retain(
    iree_hal_executable_t* executable);

// Releases the given |executable| from the caller.
IREE_API_EXPORT void iree_hal_executable_release(
    iree_hal_executable_t* executable);

//===----------------------------------------------------------------------===//
// iree_hal_executable_t implementation details
//===----------------------------------------------------------------------===//

typedef struct {
  // << HAL C porting in progress >>
  IREE_API_UNSTABLE

  void(IREE_API_PTR* destroy)(iree_hal_executable_t* executable);
} iree_hal_executable_vtable_t;

IREE_API_EXPORT void iree_hal_executable_destroy(
    iree_hal_executable_t* executable);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_EXECUTABLE_H_

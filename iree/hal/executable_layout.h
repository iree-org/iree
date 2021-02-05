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

#ifndef IREE_HAL_EXECUTABLE_LAYOUT_H_
#define IREE_HAL_EXECUTABLE_LAYOUT_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/descriptor_set_layout.h"
#include "iree/hal/resource.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_device_s iree_hal_device_t;

//===----------------------------------------------------------------------===//
// iree_hal_executable_layout_t
//===----------------------------------------------------------------------===//

// Defines the resource binding layout used by an executable.
// A "descriptor" is effectively a bound memory range and each dispatch can use
// one or more "descriptor sets" to access their I/O memory. A "descriptor set
// layout" defines the types and usage semantics of the descriptors that make up
// one set. An "executable layout" defines all of the set layouts that will be
// used when dispatching. Implementations can use this to verify program
// correctness and accelerate reservation/allocatation/computation of
// descriptor-related operations.
//
// Executables can share the same layout even if they do not use all of the
// resources referenced by descriptor sets referenced by the layout. Doing so
// allows for more efficient binding as bound descriptor sets can be reused when
// command buffer executable bindings change.
//
// Maps to VkPipelineLayout:
// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkPipelineLayout.html
typedef struct iree_hal_executable_layout_s iree_hal_executable_layout_t;

// Creates an executable layout composed of the given descriptor set layouts.
// The returned executable layout can be used by multiple executables with the
// same compatible resource binding layouts.
IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_executable_layout_create(
    iree_hal_device_t* device, iree_host_size_t push_constants,
    iree_host_size_t set_layout_count,
    iree_hal_descriptor_set_layout_t** set_layouts,
    iree_hal_executable_layout_t** out_executable_layout);

// Retains the given |executable_layout| for the caller.
IREE_API_EXPORT void IREE_API_CALL iree_hal_executable_layout_retain(
    iree_hal_executable_layout_t* executable_layout);

// Releases the given |executable_layout| from the caller.
IREE_API_EXPORT void IREE_API_CALL iree_hal_executable_layout_release(
    iree_hal_executable_layout_t* executable_layout);

//===----------------------------------------------------------------------===//
// iree_hal_executable_layout_t implementation details
//===----------------------------------------------------------------------===//

typedef struct {
  // << HAL C porting in progress >>
  IREE_API_UNSTABLE

  void(IREE_API_PTR* destroy)(iree_hal_executable_layout_t* executable_layout);
} iree_hal_executable_layout_vtable_t;

IREE_API_EXPORT void IREE_API_CALL iree_hal_executable_layout_destroy(
    iree_hal_executable_layout_t* executable_layout);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_EXECUTABLE_LAYOUT_H_

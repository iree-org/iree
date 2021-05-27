// Copyright 2021 Google LLC
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

#ifndef IREE_HAL_ROCM_ALLOCATOR_H_
#define IREE_HAL_ROCM_ALLOCATOR_H_

#include "experimental/rocm/context_wrapper.h"
#include "experimental/rocm/status_util.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Create a ROCM allocator.
iree_status_t iree_hal_rocm_allocator_create(
    iree_hal_rocm_context_wrapper_t *context,
    iree_hal_allocator_t **out_allocator);

// Free an allocation represent by the given device or host pointer.
void iree_hal_rocm_allocator_free(iree_hal_allocator_t *allocator,
                                  hipDeviceptr_t device_ptr, void *host_ptr,
                                  iree_hal_memory_type_t memory_type);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_ROCM_ALLOCATOR_H_

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

#ifndef IREE_HAL_CUDA_BUFFER_H_
#define IREE_HAL_CUDA_BUFFER_H_

#include "iree/hal/api.h"
#include "iree/hal/cuda/cuda_headers.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Wraps a cuda allocation in an iree_hal_buffer_t.
iree_status_t iree_hal_cuda_buffer_wrap(
    iree_hal_allocator_t* allocator, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_device_size_t allocation_size,
    iree_device_size_t byte_offset, iree_device_size_t byte_length,
    CUdeviceptr device_ptr, void* host_ptr, iree_hal_buffer_t** out_buffer);

// Returns the cuda base pointer for the given |buffer|.
// This is the entire allocated_buffer and must be offset by the buffer
// byte_offset and byte_length when used.
CUdeviceptr iree_hal_cuda_buffer_device_pointer(iree_hal_buffer_t* buffer);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_CUDA_BUFFER_H_

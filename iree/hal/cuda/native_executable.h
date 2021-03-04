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

#ifndef IREE_HAL_CUDA_NATIVE_EXECUTABLE_H_
#define IREE_HAL_CUDA_NATIVE_EXECUTABLE_H_

#include "iree/hal/api.h"
#include "iree/hal/cuda/context_wrapper.h"
#include "iree/hal/cuda/cuda_headers.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates an executable from a PTX module. The module may contain several
// kernels that can be extracted along with the associated block size.
iree_status_t iree_hal_cuda_native_executable_create(
    iree_hal_cuda_context_wrapper_t* context,
    const iree_hal_executable_spec_t* executable_spec,
    iree_hal_executable_t** out_executable);

CUfunction iree_hal_cuda_native_executable_for_entry_point(
    iree_hal_executable_t* executable, int32_t entry_point);

// Return the block size of the given |entry_point| within the executable.
iree_status_t iree_hal_cuda_native_executable_block_size(
    iree_hal_executable_t* executable, int32_t entry_point, uint32_t* x,
    uint32_t* y, uint32_t* z);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_CUDA_NATIVE_EXECUTABLE_H_

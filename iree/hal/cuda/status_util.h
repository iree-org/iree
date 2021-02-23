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

#ifndef IREE_HAL_CUDA_STATUS_UTIL_H_
#define IREE_HAL_CUDA_STATUS_UTIL_H_

#include "iree/base/api.h"
#include "iree/hal/cuda/dynamic_symbols.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Converts a CUresult to an iree_status_t.
//
// Usage:
//   iree_status_t status = CU_RESULT_TO_STATUS(cuDoThing(...));
#define CU_RESULT_TO_STATUS(syms, expr, ...) \
  iree_hal_cuda_result_to_status((syms), ((syms)->expr), __FILE__, __LINE__)

// IREE_RETURN_IF_ERROR but implicitly converts the CUresult return value to
// a Status.
//
// Usage:
//   CUDA_RETURN_IF_ERROR(cuDoThing(...), "message");
#define CUDA_RETURN_IF_ERROR(syms, expr, ...)                                 \
  IREE_RETURN_IF_ERROR(iree_hal_cuda_result_to_status((syms), ((syms)->expr), \
                                                      __FILE__, __LINE__),    \
                       __VA_ARGS__)

// IREE_IGNORE_ERROR but implicitly converts the CUresult return value to a
// ::util::Status and checks that it is OkStatus.
//
// Usage:
//   CUDA_IGNORE_ERROR(cuDoThing(...));
#define CUDA_IGNORE_ERROR(syms, expr)                                      \
  IREE_IGNORE_ERROR(iree_hal_cuda_result_to_status((syms), ((syms)->expr), \
                                                   __FILE__, __LINE__))

// Converts a CUresult to a Status object.
iree_status_t iree_hal_cuda_result_to_status(
    iree_hal_cuda_dynamic_symbols_t* syms, CUresult result, const char* file,
    uint32_t line);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_CUDA_STATUS_UTIL_H_

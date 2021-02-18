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

#include "iree/hal/cuda/status_util.h"

#include "iree/hal/cuda/dynamic_symbols.h"

iree_status_t iree_hal_cuda_result_to_status(
    iree_hal_cuda_dynamic_symbols_t* syms, CUresult result, const char* file,
    uint32_t line) {
  if (IREE_LIKELY(result == CUDA_SUCCESS)) {
    return iree_ok_status();
  }

  const char* error_name = NULL;
  if (syms->cuGetErrorName(result, &error_name) != CUDA_SUCCESS) {
    error_name = "UNKNOWN";
  }

  const char* error_string = NULL;
  if (syms->cuGetErrorString(result, &error_string) != CUDA_SUCCESS) {
    error_string = "Unknown error.";
  }
  return iree_make_status(IREE_STATUS_INTERNAL,
                          "CUDA driver error '%s' (%d): %s", error_name, result,
                          error_string);
}

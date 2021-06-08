// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/cuda/status_util.h"

#include <stddef.h>

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

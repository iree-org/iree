// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/rocm/status_util.h"

#include "experimental/rocm/dynamic_symbols.h"

iree_status_t iree_hal_rocm_result_to_status(
    iree_hal_rocm_dynamic_symbols_t *syms, hipError_t result, const char *file,
    uint32_t line) {
  if (IREE_LIKELY(result == hipSuccess)) {
    return iree_ok_status();
  }

  const char *error_name = syms->hipGetErrorName(result);
  if (result == hipErrorUnknown) {
    error_name = "UNKNOWN";
  }

  const char *error_string = syms->hipGetErrorString(result);
  if (result == hipErrorUnknown) {
    error_string = "Unknown error.";
  }
  return iree_make_status(IREE_STATUS_INTERNAL,
                          "rocm driver error '%s' (%d): %s", error_name, result,
                          error_string);
}

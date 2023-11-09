// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/hip/status_util.h"

#include <stddef.h>

#include "experimental/hip/dynamic_symbols.h"
#include "iree/base/status.h"

// TODO: Map HIP error strings with their corresponding IREE error state
// classification.

// Converts HIP |error_name| to the corresponding IREE status code.
static iree_status_code_t iree_hal_hip_error_name_to_status_code(
    const char* error_name) {
  return IREE_STATUS_UNKNOWN;
}

iree_status_t iree_hal_hip_result_to_status(
    const iree_hal_hip_dynamic_symbols_t* syms, hipError_t result,
    const char* file, uint32_t line) {
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

  return iree_make_status_with_location(
    file, line, iree_hal_hip_error_name_to_status_code(error_name),
    "HIP driver error '%s' (%d): %s", error_name, result, error_string);
}

// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/hip/status_util.h"

#include <stddef.h>

#include "experimental/hip/dynamic_symbols.h"
#include "iree/base/status.h"

// The list of HIP error strings with their corresponding IREE error state
// classification.
//
// Note that the list of errors is taken from `hipError_t` enum in
// hip_runtime_api.h. This is not an exhaustive list; we are just listing
// common ones here.
#define IREE_HIP_ERROR_LIST(IREE_HIP_MAP_ERROR)                                \
  IREE_HIP_MAP_ERROR("hipErrorInvalidValue", IREE_STATUS_INVALID_ARGUMENT)     \
  IREE_HIP_MAP_ERROR("hipErrorOutOfMemory", IREE_STATUS_RESOURCE_EXHAUSTED)    \
  IREE_HIP_MAP_ERROR("hipErrorInitializationError", IREE_STATUS_INTERNAL)      \
  IREE_HIP_MAP_ERROR("hipErrorDeinitialized", IREE_STATUS_INTERNAL)            \
  IREE_HIP_MAP_ERROR("hipErrorNoDevice", IREE_STATUS_FAILED_PRECONDITION)      \
  IREE_HIP_MAP_ERROR("hipErrorInvalidDevice", IREE_STATUS_FAILED_PRECONDITION) \
  IREE_HIP_MAP_ERROR("hipErrorInvalidImage", IREE_STATUS_FAILED_PRECONDITION)  \
  IREE_HIP_MAP_ERROR("hipErrorInvalidContext",                                 \
                     IREE_STATUS_FAILED_PRECONDITION)                          \
  IREE_HIP_MAP_ERROR("hipErrorNotMapped", IREE_STATUS_INTERNAL)                \
  IREE_HIP_MAP_ERROR("hipErrorNotMappedAsArray", IREE_STATUS_INTERNAL)         \
  IREE_HIP_MAP_ERROR("hipErrorNotMappedAsPointer", IREE_STATUS_INTERNAL)       \
  IREE_HIP_MAP_ERROR("hipErrorInvalidSource", IREE_STATUS_FAILED_PRECONDITION) \
  IREE_HIP_MAP_ERROR("hipErrorNotFound", IREE_STATUS_NOT_FOUND)                \
  IREE_HIP_MAP_ERROR("hipErrorNotReady", IREE_STATUS_UNAVAILABLE)              \
  IREE_HIP_MAP_ERROR("hipErrorUnknown", IREE_STATUS_UNKNOWN)

// Converts HIP |error_name| to the corresponding IREE status code.
static iree_status_code_t iree_hal_hip_error_name_to_status_code(
    const char* error_name) {
#define IREE_HIP_ERROR_TO_IREE_STATUS(hip_error, iree_status)   \
  if (strncmp(error_name, hip_error, strlen(hip_error)) == 0) { \
    return iree_status;                                         \
  }
  IREE_HIP_ERROR_LIST(IREE_HIP_ERROR_TO_IREE_STATUS)
#undef IREE_HIP_ERROR_TO_IREE_STATUS
  return IREE_STATUS_UNKNOWN;
}

iree_status_t iree_hal_hip_result_to_status(
    const iree_hal_hip_dynamic_symbols_t* syms, hipError_t result,
    const char* file, uint32_t line) {
  if (IREE_LIKELY(result == hipSuccess)) {
    return iree_ok_status();
  }

  const char* error_name = syms->hipGetErrorName(result);
  if (result == hipErrorUnknown) {
    error_name = "HIP_ERROR_UNKNOWN";
  }

  const char* error_string = syms->hipGetErrorString(result);
  if (result == hipErrorUnknown) {
    error_string = "unknown error";
  }

  return iree_make_status_with_location(
      file, line, iree_hal_hip_error_name_to_status_code(error_name),
      "HIP driver error '%s' (%d): %s", error_name, result, error_string);
}

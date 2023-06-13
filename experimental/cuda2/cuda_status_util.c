// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/cuda2/cuda_status_util.h"

#include <stddef.h>

#include "experimental/cuda2/cuda_dynamic_symbols.h"
#include "iree/base/status.h"

// The list of CUDA error strings with their corresponding IREE error state
// classification.
//
// Note that the list of errors is taken from `cudaError_enum` in cuda.h.
// This is not an exhaustive list; we are just listing common ones here.
#define IREE_CUDA_ERROR_LIST(IREE_CUDA_MAP_ERROR)                              \
  IREE_CUDA_MAP_ERROR("CUDA_ERROR_INVALID_VALUE",                              \
                      IREE_STATUS_INVALID_ARGUMENT)                            \
  IREE_CUDA_MAP_ERROR("CUDA_ERROR_OUT_OF_MEMORY",                              \
                      IREE_STATUS_RESOURCE_EXHAUSTED)                          \
  IREE_CUDA_MAP_ERROR("CUDA_ERROR_NOT_INITIALIZED", IREE_STATUS_INTERNAL)      \
  IREE_CUDA_MAP_ERROR("CUDA_ERROR_DEINITIALIZED", IREE_STATUS_INTERNAL)        \
  IREE_CUDA_MAP_ERROR("CUDA_ERROR_STUB_LIBRARY",                               \
                      IREE_STATUS_FAILED_PRECONDITION)                         \
  IREE_CUDA_MAP_ERROR("CUDA_ERROR_NO_DEVICE", IREE_STATUS_FAILED_PRECONDITION) \
  IREE_CUDA_MAP_ERROR("CUDA_ERROR_INVALID_DEVICE",                             \
                      IREE_STATUS_FAILED_PRECONDITION)                         \
  IREE_CUDA_MAP_ERROR("CUDA_ERROR_DEVICE_NOT_LICENSED",                        \
                      IREE_STATUS_FAILED_PRECONDITION)                         \
  IREE_CUDA_MAP_ERROR("CUDA_ERROR_INVALID_IMAGE",                              \
                      IREE_STATUS_FAILED_PRECONDITION)                         \
  IREE_CUDA_MAP_ERROR("CUDA_ERROR_INVALID_CONTEXT", IREE_STATUS_INTERNAL)      \
  IREE_CUDA_MAP_ERROR("CUDA_ERROR_MAP_FAILED", IREE_STATUS_INTERNAL)           \
  IREE_CUDA_MAP_ERROR("CUDA_ERROR_UNMAP_FAILED", IREE_STATUS_INTERNAL)         \
  IREE_CUDA_MAP_ERROR("CUDA_ERROR_ALREADY_MAPPED", IREE_STATUS_ALREADY_EXISTS) \
  IREE_CUDA_MAP_ERROR("CUDA_ERROR_NO_BINARY_FOR_GPU",                          \
                      IREE_STATUS_FAILED_PRECONDITION)                         \
  IREE_CUDA_MAP_ERROR("CUDA_ERROR_ALREADY_ACQUIRED",                           \
                      IREE_STATUS_ALREADY_EXISTS)                              \
  IREE_CUDA_MAP_ERROR("CUDA_ERROR_NOT_MAPPED", IREE_STATUS_INTERNAL)           \
  IREE_CUDA_MAP_ERROR("CUDA_ERROR_NOT_MAPPED_AS_ARRAY", IREE_STATUS_INTERNAL)  \
  IREE_CUDA_MAP_ERROR("CUDA_ERROR_NOT_MAPPED_AS_POINTER",                      \
                      IREE_STATUS_INTERNAL)                                    \
  IREE_CUDA_MAP_ERROR("CUDA_ERROR_UNSUPPORTED_LIMIT",                          \
                      IREE_STATUS_OUT_OF_RANGE)                                \
  IREE_CUDA_MAP_ERROR("CUDA_ERROR_CONTEXT_ALREADY_IN_USE",                     \
                      IREE_STATUS_INTERNAL)                                    \
  IREE_CUDA_MAP_ERROR("CUDA_ERROR_INVALID_PTX",                                \
                      IREE_STATUS_FAILED_PRECONDITION)                         \
  IREE_CUDA_MAP_ERROR("CUDA_ERROR_NVLINK_UNCORRECTABLE",                       \
                      IREE_STATUS_DATA_LOSS)                                   \
  IREE_CUDA_MAP_ERROR("CUDA_ERROR_JIT_COMPILER_NOT_FOUND",                     \
                      IREE_STATUS_FAILED_PRECONDITION)                         \
  IREE_CUDA_MAP_ERROR("CUDA_ERROR_UNSUPPORTED_PTX_VERSION",                    \
                      IREE_STATUS_FAILED_PRECONDITION)                         \
  IREE_CUDA_MAP_ERROR("CUDA_ERROR_JIT_COMPILATION_DISABLED",                   \
                      IREE_STATUS_FAILED_PRECONDITION)                         \
  IREE_CUDA_MAP_ERROR("CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY",                  \
                      IREE_STATUS_FAILED_PRECONDITION)                         \
  IREE_CUDA_MAP_ERROR("CUDA_ERROR_INVALID_SOURCE",                             \
                      IREE_STATUS_FAILED_PRECONDITION)                         \
  IREE_CUDA_MAP_ERROR("CUDA_ERROR_FILE_NOT_FOUND", IREE_STATUS_NOT_FOUND)      \
  IREE_CUDA_MAP_ERROR("CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND",             \
                      IREE_STATUS_NOT_FOUND)                                   \
  IREE_CUDA_MAP_ERROR("CUDA_ERROR_SHARED_OBJECT_INIT_FAILED",                  \
                      IREE_STATUS_INTERNAL)                                    \
  IREE_CUDA_MAP_ERROR("CUDA_ERROR_OPERATING_SYSTEM", IREE_STATUS_INTERNAL)     \
  IREE_CUDA_MAP_ERROR("CUDA_ERROR_INVALID_HANDLE",                             \
                      IREE_STATUS_INVALID_ARGUMENT)                            \
  IREE_CUDA_MAP_ERROR("CUDA_ERROR_ILLEGAL_STATE", IREE_STATUS_INTERNAL)        \
  IREE_CUDA_MAP_ERROR("CUDA_ERROR_NOT_FOUND", IREE_STATUS_NOT_FOUND)           \
  IREE_CUDA_MAP_ERROR("CUDA_ERROR_NOT_READY", IREE_STATUS_UNAVAILABLE)         \
  IREE_CUDA_MAP_ERROR("CUDA_ERROR_ILLEGAL_ADDRESS", IREE_STATUS_ABORTED)       \
  IREE_CUDA_MAP_ERROR("CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES",                    \
                      IREE_STATUS_RESOURCE_EXHAUSTED)                          \
  IREE_CUDA_MAP_ERROR("CUDA_ERROR_LAUNCH_TIMEOUT",                             \
                      IREE_STATUS_DEADLINE_EXCEEDED)                           \
  IREE_CUDA_MAP_ERROR("CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE",                     \
                      IREE_STATUS_INTERNAL)                                    \
  IREE_CUDA_MAP_ERROR("CUDA_ERROR_CONTEXT_IS_DESTROYED", IREE_STATUS_INTERNAL) \
  IREE_CUDA_MAP_ERROR("CUDA_ERROR_ASSERT", IREE_STATUS_ABORTED)                \
  IREE_CUDA_MAP_ERROR("CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED",             \
                      IREE_STATUS_INTERNAL)                                    \
  IREE_CUDA_MAP_ERROR("CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED",                 \
                      IREE_STATUS_INTERNAL)                                    \
  IREE_CUDA_MAP_ERROR("CUDA_ERROR_HARDWARE_STACK_ERROR", IREE_STATUS_ABORTED)  \
  IREE_CUDA_MAP_ERROR("CUDA_ERROR_ILLEGAL_INSTRUCTION", IREE_STATUS_ABORTED)   \
  IREE_CUDA_MAP_ERROR("CUDA_ERROR_MISALIGNED_ADDRESS", IREE_STATUS_ABORTED)    \
  IREE_CUDA_MAP_ERROR("CUDA_ERROR_INVALID_ADDRESS_SPACE", IREE_STATUS_ABORTED) \
  IREE_CUDA_MAP_ERROR("CUDA_ERROR_INVALID_PC", IREE_STATUS_ABORTED)            \
  IREE_CUDA_MAP_ERROR("CUDA_ERROR_LAUNCH_FAILED", IREE_STATUS_ABORTED)         \
  IREE_CUDA_MAP_ERROR("CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE",               \
                      IREE_STATUS_OUT_OF_RANGE)                                \
  IREE_CUDA_MAP_ERROR("CUDA_ERROR_NOT_PERMITTED",                              \
                      IREE_STATUS_PERMISSION_DENIED)                           \
  IREE_CUDA_MAP_ERROR("CUDA_ERROR_NOT_SUPPORTED",                              \
                      IREE_STATUS_FAILED_PRECONDITION)                         \
  IREE_CUDA_MAP_ERROR("CUDA_ERROR_SYSTEM_NOT_READY", IREE_STATUS_UNAVAILABLE)  \
  IREE_CUDA_MAP_ERROR("CUDA_ERROR_SYSTEM_DRIVER_MISMATCH",                     \
                      IREE_STATUS_FAILED_PRECONDITION)                         \
  IREE_CUDA_MAP_ERROR("CUDA_ERROR_TIMEOUT", IREE_STATUS_DEADLINE_EXCEEDED)     \
  IREE_CUDA_MAP_ERROR("CUDA_ERROR_UNKNOWN", IREE_STATUS_UNKNOWN)

// Converts CUDA |error_name| to the corresponding IREE status code.
static iree_status_code_t iree_hal_cuda2_error_name_to_status_code(
    const char* error_name) {
#define IREE_CUDA_ERROR_TO_IREE_STATUS(cuda_error, iree_status)   \
  if (strncmp(error_name, cuda_error, strlen(cuda_error)) == 0) { \
    return iree_status;                                           \
  }
  IREE_CUDA_ERROR_LIST(IREE_CUDA_ERROR_TO_IREE_STATUS)
#undef IREE_CUDA_ERROR_TO_IREE_STATUS
  return IREE_STATUS_UNKNOWN;
}

#undef IREE_CUDA_ERROR_LIST

iree_status_t iree_hal_cuda2_result_to_status(
    const iree_hal_cuda2_dynamic_symbols_t* syms, CUresult result,
    const char* file, uint32_t line) {
  if (IREE_LIKELY(result == CUDA_SUCCESS)) return iree_ok_status();

  const char* error_name = NULL;
  if (syms->cuGetErrorName(result, &error_name) != CUDA_SUCCESS) {
    error_name = "CUDA_ERROR_UNKNOWN";
  }

  const char* error_string = NULL;
  if (syms->cuGetErrorString(result, &error_string) != CUDA_SUCCESS) {
    error_string = "unknown error";
  }

  return iree_make_status_with_location(
      file, line, iree_hal_cuda2_error_name_to_status_code(error_name),
      "CUDA error '%s' (%d): %s", error_name, result, error_string);
}

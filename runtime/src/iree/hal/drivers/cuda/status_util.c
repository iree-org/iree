// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/cuda/status_util.h"

#include <stddef.h>

#include "iree/hal/drivers/cuda/dynamic_symbols.h"

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
  return iree_make_status_with_location(file, line, IREE_STATUS_INTERNAL,
                                        "CUDA driver error '%s' (%d): %s",
                                        error_name, result, error_string);
}

iree_status_t iree_hal_nccl_result_to_status(
    iree_hal_cuda_dynamic_symbols_t* syms, ncclResult_t result,
    const char* file, uint32_t line) {
  iree_status_code_t code;

  switch (result) {
    case ncclSuccess:
      return iree_ok_status();
    case ncclUnhandledCudaError:
      code = IREE_STATUS_FAILED_PRECONDITION;
      break;
    case ncclSystemError:
      code = IREE_STATUS_INTERNAL;
      break;
    case ncclInternalError:
      code = IREE_STATUS_INTERNAL;
      break;
    case ncclInvalidArgument:
      code = IREE_STATUS_INVALID_ARGUMENT;
      break;
    case ncclInvalidUsage:
      code = IREE_STATUS_FAILED_PRECONDITION;
      break;
    case ncclRemoteError:
      code = IREE_STATUS_UNAVAILABLE;
      break;
    case ncclInProgress:
      code = IREE_STATUS_DEFERRED;
      break;
    default:
      code = IREE_STATUS_INTERNAL;
      break;
  }
  return iree_make_status_with_location(file, line, code, "NCCL error %d: %s",
                                        result,
                                        syms->ncclGetErrorString(result));
}

iree_status_t iree_hal_mpi_result_to_status(
    iree_hal_cuda_dynamic_symbols_t* syms, int result, const char* file,
    uint32_t line) {
  iree_status_code_t code;

  switch (result) {
    case 0:  // MPI_SUCCESS
      return iree_ok_status();
    default:
      code = IREE_STATUS_INTERNAL;
      break;
  }
  return iree_make_status_with_location(file, line, code, "MPI error %d",
                                        result);
}

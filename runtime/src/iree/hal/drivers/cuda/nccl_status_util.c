// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/cuda/nccl_status_util.h"

#include <stddef.h>

#include "iree/hal/drivers/cuda/nccl_dynamic_symbols.h"

iree_status_t iree_hal_cuda_nccl_result_to_status(
    const iree_hal_cuda_nccl_dynamic_symbols_t* syms, ncclResult_t result,
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

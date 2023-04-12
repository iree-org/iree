// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_CUDA_STATUS_UTIL_H_
#define IREE_HAL_DRIVERS_CUDA_STATUS_UTIL_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/drivers/cuda/dynamic_symbols.h"

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
// Status.
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

// Converts a ncclResult_t to an iree_status_t.
//
// Usage:
//   iree_status_t status = NCCL_RESULT_TO_STATUS(ncclDoThing(...));
#define NCCL_RESULT_TO_STATUS(syms, expr, ...) \
  iree_hal_nccl_result_to_status((syms), ((syms)->expr), __FILE__, __LINE__)

// Converts a ncclResult_t to a Status object.
iree_status_t iree_hal_nccl_result_to_status(
    iree_hal_cuda_dynamic_symbols_t* syms, ncclResult_t result,
    const char* file, uint32_t line);

// IREE_RETURN_IF_ERROR but implicitly converts the ncclResult_t return value to
// a Status.
//
// Usage:
//   NCCL_RETURN_IF_ERROR(ncclDoThing(...), "message");
#define NCCL_RETURN_IF_ERROR(syms, expr, ...)                                 \
  IREE_RETURN_IF_ERROR(iree_hal_nccl_result_to_status((syms), ((syms)->expr), \
                                                      __FILE__, __LINE__),    \
                       __VA_ARGS__)

// IREE_IGNORE_ERROR but implicitly converts the ncclResult_t return value to a
// Status.
//
// Usage:
//   NCCL_IGNORE_ERROR(ncclDoThing(...));
#define NCCL_IGNORE_ERROR(syms, expr)                                      \
  IREE_IGNORE_ERROR(iree_hal_nccl_result_to_status((syms), ((syms)->expr), \
                                                   __FILE__, __LINE__))

// Converts a mpi result to an iree_status_t.
//
// Usage:
//   iree_status_t status = MPI_RESULT_TO_STATUS(mpiDoThing(...));
#define MPI_RESULT_TO_STATUS(syms, expr, ...) \
  iree_hal_mpi_result_to_status((syms), ((syms)->expr), __FILE__, __LINE__)

// Converts a mpi result to a Status object.
iree_status_t iree_hal_mpi_result_to_status(
    iree_hal_cuda_dynamic_symbols_t* syms, int result, const char* file,
    uint32_t line);

// IREE_RETURN_IF_ERROR but implicitly converts the mpi return value to
// a Status.
//
// Usage:
//   MPI_RETURN_IF_ERROR(mpiDoThing(...), "message");
#define MPI_RETURN_IF_ERROR(syms, expr, ...)                                 \
  IREE_RETURN_IF_ERROR(iree_hal_mpi_result_to_status((syms), ((syms)->expr), \
                                                     __FILE__, __LINE__),    \
                       __VA_ARGS__)

// IREE_IGNORE_ERROR but implicitly converts the mpi return value to a
// Status.
//
// Usage:
//   MPI_IGNORE_ERROR(mpiDoThing(...));
#define MPI_IGNORE_ERROR(syms, expr)                                      \
  IREE_IGNORE_ERROR(iree_hal_mpi_result_to_status((syms), ((syms)->expr), \
                                                  __FILE__, __LINE__))

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_CUDA_STATUS_UTIL_H_

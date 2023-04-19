// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_UTILS_LIBMPI_H_
#define IREE_HAL_UTILS_LIBMPI_H_

#include "iree/base/internal/dynamic_library.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_mpi_dynamic_symbols_t {
#define MPI_PFN_DECL(mpiSymbolName, ...) int (*mpiSymbolName)(__VA_ARGS__);
#include "iree/hal/utils/libmpi_dynamic_symbols.h"
#undef MPI_PFN_DECL
} iree_hal_mpi_dynamic_symbols_t;

// Converts a mpi result to an iree_status_t.
//
// Usage:
//   iree_status_t status = MPI_RESULT_TO_STATUS(mpiDoThing(...));
#define MPI_RESULT_TO_STATUS(syms, expr, ...) \
  iree_hal_mpi_result_to_status((syms), ((syms)->expr), __FILE__, __LINE__)

// Converts a mpi result to a Status object.
iree_status_t iree_hal_mpi_result_to_status(
    iree_hal_mpi_dynamic_symbols_t* syms, int result, const char* file,
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

iree_status_t iree_hal_mpi_initialize_library(
    iree_allocator_t host_allocator, iree_dynamic_library_t** out_library,
    iree_hal_mpi_dynamic_symbols_t** out_syms);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_UTILS_LIBMPI_H_

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

// A collection of known MPI symbols (functions and handles)
//
// To support additional functions, define the function or handles
// in libmpi_dynamic_symbols.h
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

// Dynamically loads the OpenMPI library and sets up the symbol table.
//
// |out_library| - must be release by the caller with
// `iree_dynamic_library_release`.
// |out_syms| - only valid as long as the library is live. Must be released
// by the caller with `iree_allocator_free`.
//
// Returns: IREE_STATUS_SUCCESS if it found the library, and fills in the
// pointers to our_library and out_symbols
//
// Why only OpenMPI? Because we rely on internal symbols to access handles
// to operators and data types. See libmpi_dynamic_symbols.h
iree_status_t iree_hal_mpi_library_load(
    iree_allocator_t host_allocator, iree_dynamic_library_t** out_library,
    iree_hal_mpi_dynamic_symbols_t** out_syms);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_UTILS_LIBMPI_H_

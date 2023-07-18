// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_CUDA2_CUDA_DYNAMIC_SYMBOLS_H_
#define IREE_EXPERIMENTAL_CUDA2_CUDA_DYNAMIC_SYMBOLS_H_

#include "experimental/cuda2/cuda_headers.h"
#include "iree/base/api.h"
#include "iree/base/internal/dynamic_library.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// iree_dynamic_library_t allows dynamically loading a subset of the CUDA driver
// API. We load all the symbols in `cuda_dynamic_symbol_table.h` and fail if any
// of the symbol is not available. The functions signatures are matching the
// declarations in `cuda.h`.

// CUDA driver API dynamic symbols.
typedef struct iree_hal_cuda2_dynamic_symbols_t {
  // The dynamic library handle.
  iree_dynamic_library_t* dylib;

  // Concrete CUDA symbols defined by including the `dynamic_symbol_tables.h`.
#define IREE_CU_PFN_DECL(cudaSymbolName, ...) \
  CUresult (*cudaSymbolName)(__VA_ARGS__);
#include "experimental/cuda2/cuda_dynamic_symbol_table.h"  // IWYU pragma: export
#undef IREE_CU_PFN_DECL
} iree_hal_cuda2_dynamic_symbols_t;

// Initializes |out_syms| in-place with dynamically loaded CUDA symbols.
// iree_hal_cuda2_dynamic_symbols_deinitialize must be used to release the
// library resources.
iree_status_t iree_hal_cuda2_dynamic_symbols_initialize(
    iree_allocator_t host_allocator,
    iree_hal_cuda2_dynamic_symbols_t* out_syms);

// Deinitializes |syms| by unloading the backing library. All function pointers
// will be invalidated. They _may_ still work if there are other reasons the
// library remains loaded so be careful.
void iree_hal_cuda2_dynamic_symbols_deinitialize(
    iree_hal_cuda2_dynamic_symbols_t* syms);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_EXPERIMENTAL_CUDA2_CUDA_DYNAMIC_SYMBOLS_H_

// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_CUDA2_DYNAMIC_SYMBOLS_H_
#define IREE_EXPERIMENTAL_CUDA2_DYNAMIC_SYMBOLS_H_

#include "experimental/cuda2/cuda_headers.h"
#include "iree/base/api.h"
#include "iree/base/internal/dynamic_library.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// iree_dynamic_library_t allows dynamically loading a subset of CUDA driver API
// and the NCCL API. We load all the symbols in `dynamic_symbol_tables.h` and
// fail if any of the symbol is not available. The functions signatures are
// matching the declarations in `cuda.h` and `nccl.h"`.

//===----------------------------------------------------------------------===//
// CUDA dynamic symbols
//===----------------------------------------------------------------------===//

// CUDA driver API dynamic symbols.
typedef struct iree_hal_cuda2_dynamic_symbols_t {
  // The dynamic library handle.
  iree_dynamic_library_t* dylib;

  // Concrete CUDA symbols defined by including the `dynamic_symbol_tables.h`.
#define IREE_CU_PFN_DECL(cudaSymbolName, ...) \
  CUresult (*cudaSymbolName)(__VA_ARGS__);
// Ignore NCCL symbols
#define IREE_NCCL_PFN_DECL(ncclSymbolName, ...)
#define IREE_NCCL_PFN_DECL_STR_RETURN(ncclSymbolName, ...)
#include "experimental/cuda2/dynamic_symbol_tables.h"  // IWYU pragma: export
#undef IREE_CU_PFN_DECL
#undef IREE_NCCL_PFN_DECL
#undef IREE_NCCL_PFN_DECL_STR_RETURN
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

//===----------------------------------------------------------------------===//
// NCCL dynamic symbols
//===----------------------------------------------------------------------===//

// NCCL API dynamic symbols.
typedef struct iree_hal_cuda2_nccl_dynamic_symbols_t {
  // The dynamic library handle.
  iree_dynamic_library_t* dylib;

  // Concrete NCCL symbols defined by including the `dynamic_symbol_tables.h`.
#define IREE_NCCL_PFN_DECL(ncclSymbolName, ...) \
  ncclResult_t (*ncclSymbolName)(__VA_ARGS__);
#define IREE_NCCL_PFN_DECL_STR_RETURN(ncclSymbolName, ...) \
  const char* (*ncclSymbolName)(__VA_ARGS__);
// Ignore CUDA symbols
#define IREE_CU_PFN_DECL(cudaSymbolName, ...)
#include "experimental/cuda2/dynamic_symbol_tables.h"  // IWYU pragma: export
#undef IREE_NCCL_PFN_DECL
#undef IREE_NCCL_PFN_DECL_STR_RETURN
#undef IREE_CU_PFN_DECL
} iree_hal_cuda2_nccl_dynamic_symbols_t;

// Initializes |out_syms| in-place with dynamically loaded NCCL symbols.
// iree_hal_cuda2_dynamic_symbols_deinitialize must be used to release the
// library resources.
iree_status_t iree_hal_cuda2_nccl_dynamic_symbols_initialize(
    iree_allocator_t host_allocator,
    const iree_hal_cuda2_dynamic_symbols_t* cuda_library,
    iree_hal_cuda2_nccl_dynamic_symbols_t* out_syms);

// Deinitializes |syms| by unloading the backing library. All function pointers
// will be invalidated. They _may_ still work if there are other reasons the
// library remains loaded so be careful.
void iree_hal_cuda2_nccl_dynamic_symbols_deinitialize(
    iree_hal_cuda2_nccl_dynamic_symbols_t* syms);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_EXPERIMENTAL_CUDA2_DYNAMIC_SYMBOLS_H_

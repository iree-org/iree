// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_ROCM_DYNAMIC_SYMBOLS_H_
#define IREE_HAL_ROCM_DYNAMIC_SYMBOLS_H_

#include "experimental/rocm/rocm_headers.h"
#include "iree/base/api.h"
#include "iree/base/internal/dynamic_library.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// DynamicSymbols allow loading dynamically a subset of ROCM driver API. It
// loads all the function declared in `dynamic_symbol_tables.def` and fail if
// any of the symbol is not available. The functions signatures are matching
// the declarations in `hipruntime.h`.
typedef struct iree_hal_rocm_dynamic_symbols_t {
  iree_dynamic_library_t* loader_library;

#define IREE_HAL_ROCM_REQUIRED_PFN_DECL(rocmSymbolName, ...) \
  hipError_t (*rocmSymbolName)(__VA_ARGS__);
#define IREE_HAL_ROCM_REQUIRED_PFN_STR_DECL(rocmSymbolName, ...) \
  const char* (*rocmSymbolName)(__VA_ARGS__);
#define IREE_HAL_ROCM_OPTIONAL_PFN_DECL(rocmSymbolName, ...) \
  hipError_t (*rocmSymbolName)(__VA_ARGS__);
#include "experimental/rocm/dynamic_symbol_tables.h"  // IWYU pragma: export
#undef IREE_HAL_ROCM_REQUIRED_PFN_DECL
#undef IREE_HAL_ROCM_REQUIRED_PFN_STR_DECL
#undef IREE_HAL_ROCM_OPTIONAL_PFN_DECL
} iree_hal_rocm_dynamic_symbols_t;

// Initializes |out_syms| in-place with dynamically loaded ROCM symbols.
// iree_hal_rocm_dynamic_symbols_deinitialize must be used to release the
// library resources.
iree_status_t iree_hal_rocm_dynamic_symbols_initialize(
    iree_allocator_t allocator, iree_hal_rocm_dynamic_symbols_t* out_syms);

// Deinitializes |syms| by unloading the backing library. All function pointers
// will be invalidated. They _may_ still work if there are other reasons the
// library remains loaded so be careful.
void iree_hal_rocm_dynamic_symbols_deinitialize(
    iree_hal_rocm_dynamic_symbols_t* syms);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_ROCM_DYNAMIC_SYMBOLS_H_

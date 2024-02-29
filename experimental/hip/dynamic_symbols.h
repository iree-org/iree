// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_HIP_DYNAMIC_SYMBOLS_H_
#define IREE_EXPERIMENTAL_HIP_DYNAMIC_SYMBOLS_H_

#include "experimental/hip/hip_headers.h"
#include "iree/base/api.h"
#include "iree/base/internal/dynamic_library.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// iree_dynamic_library_t allows dynamically loading a subset of HIP driver API.
// We load all the symbols in `dynamic_symbol_tables.h` and fail if any of the
// symbol is not available. The functions signatures are matching the
// declarations in `hip_runtime_api.h`.

//===----------------------------------------------------------------------===//
// HIP dynamic symbols
//===----------------------------------------------------------------------===//

// HIP driver API dynamic symbols.
typedef struct iree_hal_hip_dynamic_symbols_t {
  // The dynamic library handle.
  iree_dynamic_library_t* dylib;

  // Concrete HIP symbols defined by including the `dynamic_symbol_tables.h`.
#define IREE_HAL_HIP_REQUIRED_PFN_DECL(hipSymbolName, ...) \
  hipError_t (*hipSymbolName)(__VA_ARGS__);
#define IREE_HAL_HIP_REQUIRED_PFN_STR_DECL(hipSymbolName, ...) \
  const char* (*hipSymbolName)(__VA_ARGS__);
#define IREE_HAL_HIP_OPTIONAL_PFN_DECL(hipSymbolName, ...) \
  hipError_t (*hipSymbolName)(__VA_ARGS__);
#include "experimental/hip/dynamic_symbol_tables.h"  // IWYU pragma: export
#undef IREE_HAL_HIP_REQUIRED_PFN_DECL
#undef IREE_HAL_HIP_REQUIRED_PFN_STR_DECL
#undef IREE_HAL_HIP_OPTIONAL_PFN_DECL
} iree_hal_hip_dynamic_symbols_t;

// Initializes |out_syms| in-place with dynamically loaded HIP symbols.
// iree_hal_hip_dynamic_symbols_deinitialize must be used to release the
// library resources.
iree_status_t iree_hal_hip_dynamic_symbols_initialize(
    iree_allocator_t host_allocator, iree_hal_hip_dynamic_symbols_t* out_syms);

// Deinitializes |syms| by unloading the backing library. All function pointers
// will be invalidated. They _may_ still work if there are other reasons the
// library remains loaded so be careful.
void iree_hal_hip_dynamic_symbols_deinitialize(
    iree_hal_hip_dynamic_symbols_t* syms);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_EXPERIMENTAL_HIP_DYNAMIC_SYMBOLS_H_

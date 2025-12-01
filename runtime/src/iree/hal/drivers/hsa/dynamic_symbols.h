// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_HSA_DYNAMIC_SYMBOLS_H_
#define IREE_HAL_DRIVERS_HSA_DYNAMIC_SYMBOLS_H_

#include "iree/base/api.h"
#include "iree/base/internal/dynamic_library.h"
#include "iree/hal/drivers/hsa/hsa_headers.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// iree_dynamic_library_t allows dynamically loading a subset of HSA runtime API.
// We load all the symbols in `dynamic_symbol_tables.h` and fail if any of the
// required symbols are not available.

//===----------------------------------------------------------------------===//
// HSA dynamic symbols
//===----------------------------------------------------------------------===//

// HSA runtime API dynamic symbols.
typedef struct iree_hal_hsa_dynamic_symbols_t {
  // The dynamic library handle.
  iree_dynamic_library_t* dylib;

  // Concrete HSA symbols defined by including the `dynamic_symbol_tables.h`.
#define IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_symbol_name, ...) \
  hsa_status_t (*hsa_symbol_name)(__VA_ARGS__);
#define IREE_HAL_HSA_OPTIONAL_PFN_DECL(hsa_symbol_name, ...) \
  hsa_status_t (*hsa_symbol_name)(__VA_ARGS__);
#include "iree/hal/drivers/hsa/dynamic_symbol_tables.h"  // IWYU pragma: export
#undef IREE_HAL_HSA_REQUIRED_PFN_DECL
#undef IREE_HAL_HSA_OPTIONAL_PFN_DECL
} iree_hal_hsa_dynamic_symbols_t;

// Initializes |out_syms| in-place with dynamically loaded HSA symbols.
// iree_hal_hsa_dynamic_symbols_deinitialize must be used to release the
// library resources.
//
// If |hsa_lib_search_path_count| is non zero, then |hsa_lib_search_paths|
// is a list of paths to guide searching for the dynamic libhsa-runtime64.so,
// which contains the backing HSA runtime library.
iree_status_t iree_hal_hsa_dynamic_symbols_initialize(
    iree_allocator_t host_allocator, iree_host_size_t hsa_lib_search_path_count,
    const iree_string_view_t* hsa_lib_search_paths,
    iree_hal_hsa_dynamic_symbols_t* out_syms);

// Deinitializes |syms| by unloading the backing library.
void iree_hal_hsa_dynamic_symbols_deinitialize(
    iree_hal_hsa_dynamic_symbols_t* syms);

// Gets the absolute path of the shared library providing the dynamic symbols.
iree_status_t iree_hal_hsa_dynamic_symbols_append_path_to_builder(
    iree_hal_hsa_dynamic_symbols_t* syms, iree_string_builder_t* out_path);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_HSA_DYNAMIC_SYMBOLS_H_


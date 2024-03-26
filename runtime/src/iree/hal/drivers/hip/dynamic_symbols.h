// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_HIP_DYNAMIC_SYMBOLS_H_
#define IREE_HAL_DRIVERS_HIP_DYNAMIC_SYMBOLS_H_

#include "iree/base/api.h"
#include "iree/base/internal/dynamic_library.h"
#include "iree/hal/drivers/hip/hip_headers.h"

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
#include "iree/hal/drivers/hip/dynamic_symbol_tables.h"  // IWYU pragma: export
#undef IREE_HAL_HIP_REQUIRED_PFN_DECL
#undef IREE_HAL_HIP_REQUIRED_PFN_STR_DECL
#undef IREE_HAL_HIP_OPTIONAL_PFN_DECL
} iree_hal_hip_dynamic_symbols_t;

// Initializes |out_syms| in-place with dynamically loaded HIP symbols.
// iree_hal_hip_dynamic_symbols_deinitialize must be used to release the
// library resources.
//
// If |hip_lib_search_path_count| is non zero, then |hip_lib_search_paths|
// is a list of paths to guide searching for the dynamic libamdhip64.so (or
// amdhip64.dll), which contains the backing HIP runtime library. If this
// is present, it overrides any other mechanism for finding the HIP runtime
// library. Default search heuristics are used (i.e. ask the system to find an
// appropriately named library) if there are zero entries.
// Each entry can be:
//
// * Directory in which to find a platform specific runtime library
//   name.
// * Specific fully qualified path to a file that will be loaded with no
//   further interpretation if the entry starts with "file:".
iree_status_t iree_hal_hip_dynamic_symbols_initialize(
    iree_allocator_t host_allocator, iree_host_size_t hip_lib_search_path_count,
    const iree_string_view_t* hip_lib_search_paths,
    iree_hal_hip_dynamic_symbols_t* out_syms);

// Deinitializes |syms| by unloading the backing library. All function pointers
// will be invalidated. They _may_ still work if there are other reasons the
// library remains loaded so be careful.
void iree_hal_hip_dynamic_symbols_deinitialize(
    iree_hal_hip_dynamic_symbols_t* syms);

// Gets the absolute path of the shared library or DLL providing the dynamic
// symbols. If not loaded from a shared library, the exact behavior is
// implementation dependent (i.e. it may return a failure status or it may
// return a path to an executable, etc).
iree_status_t iree_hal_hip_dynamic_symbols_append_path_to_builder(
    iree_hal_hip_dynamic_symbols_t* syms, iree_string_builder_t* out_path);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_HIP_DYNAMIC_SYMBOLS_H_

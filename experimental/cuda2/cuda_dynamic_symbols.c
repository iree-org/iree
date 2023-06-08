// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/cuda2/cuda_dynamic_symbols.h"

#include <string.h>

#include "experimental/cuda2/cuda_status_util.h"
#include "iree/base/assert.h"
#include "iree/base/internal/dynamic_library.h"
#include "iree/base/target_platform.h"
#include "iree/base/tracing.h"

static const char* iree_hal_cuda_dylib_names[] = {
#if defined(IREE_PLATFORM_WINDOWS)
    "nvcuda.dll",
#else
    "libcuda.so",
#endif  // IREE_PLATFORM_WINDOWS
};

#define IREE_CONCAT(A, B) A B

// Resolves all CUDA dynamic symbols in `dynamic_symbol_tables.h`, prefer _v2
// version if it exists.
static iree_status_t iree_hal_cuda2_dynamic_symbols_resolve_all(
    iree_hal_cuda2_dynamic_symbols_t* syms) {
#define IREE_CU_PFN_DECL(cuda_symbol_name, ...)                         \
  {                                                                     \
    static const char* name = #cuda_symbol_name;                        \
    IREE_RETURN_IF_ERROR(iree_dynamic_library_lookup_symbol(            \
        syms->dylib, name, (void**)&syms->cuda_symbol_name));           \
    static const char* name_v2 = IREE_CONCAT(#cuda_symbol_name, "_v2"); \
    void* fptr_v2;                                                      \
    iree_dynamic_library_lookup_symbol(syms->dylib, name_v2, &fptr_v2); \
    if (fptr_v2) syms->cuda_symbol_name = fptr_v2;                      \
  }
#include "experimental/cuda2/cuda_dynamic_symbol_table.h"  // IWYU pragma: keep
#undef IREE_CU_PFN_DECL
  return iree_ok_status();
}

#undef IREE_CONCAT

iree_status_t iree_hal_cuda2_dynamic_symbols_initialize(
    iree_allocator_t host_allocator,
    iree_hal_cuda2_dynamic_symbols_t* out_syms) {
  IREE_ASSERT_ARGUMENT(out_syms);
  IREE_TRACE_ZONE_BEGIN(z0);

  memset(out_syms, 0, sizeof(*out_syms));
  iree_status_t status = iree_dynamic_library_load_from_files(
      IREE_ARRAYSIZE(iree_hal_cuda_dylib_names), iree_hal_cuda_dylib_names,
      IREE_DYNAMIC_LIBRARY_FLAG_NONE, host_allocator, &out_syms->dylib);
  if (iree_status_is_not_found(status)) {
    iree_status_ignore(status);
    status = iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "CUDA driver library 'libcuda.so'/'nvcuda.dll' not available; please "
        "ensure installed and in dynamic library search path");
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_cuda2_dynamic_symbols_resolve_all(out_syms);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_cuda2_dynamic_symbols_deinitialize(out_syms);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_cuda2_dynamic_symbols_deinitialize(
    iree_hal_cuda2_dynamic_symbols_t* syms) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_dynamic_library_release(syms->dylib);
  memset(syms, 0, sizeof(*syms));

  IREE_TRACE_ZONE_END(z0);
}

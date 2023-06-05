// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/cuda2/dynamic_symbols.h"

#include <string.h>

#include "experimental/cuda2/status_util.h"
#include "iree/base/assert.h"
#include "iree/base/internal/dynamic_library.h"
#include "iree/base/target_platform.h"
#include "iree/base/tracing.h"

//===----------------------------------------------------------------------===//
// CUDA dynamic symbols
//===----------------------------------------------------------------------===//

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
// Ignore NCCL symbols
#define IREE_NCCL_PFN_DECL(nccl_symbol_name, ...)
#define IREE_NCCL_PFN_DECL_STR_RETURN(nccl_symbol_name, ...)
#include "experimental/cuda2/dynamic_symbol_tables.h"  // IWYU pragma: keep
#undef IREE_CU_PFN_DECL
#undef IREE_NCCL_PFN_DECL
#undef IREE_NCCL_PFN_DECL_STR_RETURN
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

//===----------------------------------------------------------------------===//
// NCCL dynamic symbols
//===----------------------------------------------------------------------===//

static const char* iree_hal_cuda_nccl_dylib_names[] = {
#if defined(IREE_PLATFORM_WINDOWS)
    "nccl.dll",
#else
    "libnccl.so",
#endif  // IREE_PLATFORM_WINDOWS
};

// Resolves all NCCL dynamic symbols in `dynamic_symbol_tables.h`, prefer _v2
// version if it exists.
static iree_status_t iree_hal_cuda2_nccl_dynamic_symbols_resolve_all(
    iree_hal_cuda2_nccl_dynamic_symbols_t* syms) {
#define IREE_NCCL_PFN_DECL(nccl_symbol_name, ...)             \
  {                                                           \
    static const char* name = #nccl_symbol_name;              \
    IREE_RETURN_IF_ERROR(iree_dynamic_library_lookup_symbol(  \
        syms->dylib, name, (void**)&syms->nccl_symbol_name)); \
  }
#define IREE_NCCL_PFN_DECL_STR_RETURN(nccl_symbol_name, ...)  \
  {                                                           \
    static const char* name = #nccl_symbol_name;              \
    IREE_RETURN_IF_ERROR(iree_dynamic_library_lookup_symbol(  \
        syms->dylib, name, (void**)&syms->nccl_symbol_name)); \
  }
// Ignore CUDA symbols
#define IREE_CU_PFN_DECL(cuda_symbol_name, ...)
#include "experimental/cuda2/dynamic_symbol_tables.h"  // IWYU pragma: keep
#undef IREE_NCCL_PFN_DECL
#undef IREE_NCCL_PFN_DECL_STR_RETURN
#undef IREE_CU_PFN_DECL
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda2_nccl_check_version(
    iree_dynamic_library_t* nccl_library) {
  ncclResult_t (*ncclGetVersion)(int*) = NULL;

  iree_status_t status = iree_dynamic_library_lookup_symbol(
      nccl_library, "ncclGetVersion", (void**)&ncclGetVersion);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "ncclGetVersion symbol not found in dynamic library");
  }

  // Check the NCCL version compatibility.
  int nccl_version = 0;
  ncclResult_t result = ncclGetVersion(&nccl_version);
  if (result != ncclSuccess) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "ncclGetVersion() failed with error %d", result);
  }

  int major = 0;
  int minor = 0;
  int patch = 0;
  if (nccl_version < 20000) {
    major = nccl_version / 1000;
    minor = (nccl_version % 1000) / 100;
  } else {
    major = nccl_version / 10000;
    minor = (nccl_version % 10000) / 100;
  }
  patch = nccl_version % 100;
  if (major != NCCL_MAJOR || minor != NCCL_MINOR || patch != NCCL_PATCH) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "NCCL version is %d.%d.%d, but %d.%d.%d is required", major, minor,
        patch, NCCL_MAJOR, NCCL_MINOR, NCCL_PATCH);
  }

  return iree_ok_status();
}

iree_status_t iree_hal_cuda2_nccl_dynamic_symbols_initialize(
    iree_allocator_t host_allocator,
    const iree_hal_cuda2_dynamic_symbols_t* cuda_library,
    iree_hal_cuda2_nccl_dynamic_symbols_t* out_syms) {
  IREE_ASSERT_ARGUMENT(out_syms);
  if (!cuda_library->dylib) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "CUDA dynamic symbols must be resolved prior to loading NCCL symbols");
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  memset(out_syms, 0, sizeof(*out_syms));
  iree_status_t status = iree_dynamic_library_load_from_files(
      IREE_ARRAYSIZE(iree_hal_cuda_nccl_dylib_names),
      iree_hal_cuda_nccl_dylib_names, IREE_DYNAMIC_LIBRARY_FLAG_NONE,
      host_allocator, &out_syms->dylib);
  if (iree_status_is_not_found(status)) {
    iree_status_ignore(status);
    status = iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "NCCL runtime library 'libnccl.so'/'nccl.dll' (version %d.%d.%d) not "
        "available; please ensure installed and in dynamic library search path",
        NCCL_MAJOR, NCCL_MINOR, NCCL_PATCH);
  }

  if (iree_status_is_ok(status)) {
    // Check the version first before resolving all symbols. This makes sure
    // that we have the right version and all symbols are available at the
    // time of resolving.
    status = iree_hal_cuda2_nccl_check_version(out_syms->dylib);
  }

  // Resolve all symbols; this will fail if any required symbols are missing.
  if (iree_status_is_ok(status)) {
    status = iree_hal_cuda2_nccl_dynamic_symbols_resolve_all(out_syms);
  }

  if (!iree_status_is_ok(status)) {
    iree_dynamic_library_release(out_syms->dylib);
    out_syms->dylib = NULL;
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_cuda2_nccl_dynamic_symbols_deinitialize(
    iree_hal_cuda2_nccl_dynamic_symbols_t* syms) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_dynamic_library_release(syms->dylib);
  memset(syms, 0, sizeof(*syms));

  IREE_TRACE_ZONE_END(z0);
}

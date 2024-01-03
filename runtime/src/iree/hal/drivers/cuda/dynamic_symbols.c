// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/cuda/dynamic_symbols.h"

#include <string.h>

#include "iree/base/internal/dynamic_library.h"
#include "iree/hal/drivers/cuda/status_util.h"

static const char* kCUDALoaderSearchNames[] = {
#if defined(IREE_PLATFORM_WINDOWS)
    "nvcuda.dll",
#else
    "libcuda.so",
#endif  // IREE_PLATFORM_WINDOWS
};

static const char* kNCCLLoaderSearchNames[] = {
#if defined(IREE_PLATFORM_WINDOWS)
    "nccl.dll",
#else
    "libnccl.so",
#endif  // IREE_PLATFORM_WINDOWS
};

// CUDA API version for cuGetProcAddress.
// 1000 * major + 10 * minor
#define IREE_CUDA_DRIVER_API_VERSION 11030

// Load CUDA entry points.
static iree_status_t iree_hal_cuda_dynamic_symbols_resolve_all(
    iree_hal_cuda_dynamic_symbols_t* syms) {
  // Since cuGetProcAddress is in the symbol table, it will be loaded again
  // through cuGetProcAddress. cuGetProcAddress_v2 is added in CUDA 12.0 and has
  // a new function signature. If IREE_CUDA_DRIVER_API_VERSION is increased to
  // >=12.0, then make sure we are using the correct signature.
  IREE_RETURN_IF_ERROR(iree_dynamic_library_lookup_symbol(
      syms->cuda_library, "cuGetProcAddress", (void**)&syms->cuGetProcAddress));
#define CU_PFN_DECL(cudaSymbolName, ...)                              \
  {                                                                   \
    static const char* kName = #cudaSymbolName;                       \
    CUDA_RETURN_IF_ERROR(                                             \
        syms,                                                         \
        cuGetProcAddress(kName, (void**)&syms->cudaSymbolName,        \
                         IREE_CUDA_DRIVER_API_VERSION,                \
                         CU_GET_PROC_ADDRESS_DEFAULT),                \
        "when resolving " #cudaSymbolName " using cuGetProcAddress"); \
  }
#define NCCL_PFN_DECL(ncclSymbolName, ...)
#define NCCL_PFN_DECL_STR_RETURN(ncclSymbolName, ...)
#include "iree/hal/drivers/cuda/dynamic_symbol_tables.h"  // IWYU pragma: keep
#undef CU_PFN_DECL
#undef NCCL_PFN_DECL
#undef NCCL_PFN_DECL_STR_RETURN
  return iree_ok_status();
}

// Load NCCL entry points.
static iree_status_t iree_hal_cuda_nccl_dynamic_symbols_resolve_all(
    iree_hal_cuda_dynamic_symbols_t* syms) {
#define CU_PFN_DECL(cudaSymbolName, ...)
#define NCCL_PFN_DECL(ncclSymbolName, ...)                          \
  {                                                                 \
    static const char* kName = #ncclSymbolName;                     \
    IREE_RETURN_IF_ERROR(iree_dynamic_library_lookup_symbol(        \
        syms->nccl_library, kName, (void**)&syms->ncclSymbolName)); \
  }
#define NCCL_PFN_DECL_STR_RETURN(ncclSymbolName, ...)               \
  {                                                                 \
    static const char* kName = #ncclSymbolName;                     \
    IREE_RETURN_IF_ERROR(iree_dynamic_library_lookup_symbol(        \
        syms->nccl_library, kName, (void**)&syms->ncclSymbolName)); \
  }
#include "iree/hal/drivers/cuda/dynamic_symbol_tables.h"  // IWYU pragma: keep
#undef CU_PFN_DECL
#undef NCCL_PFN_DECL
#undef NCCL_PFN_DECL_STR_RETURN
  return iree_ok_status();
}

iree_status_t iree_hal_cuda_dynamic_symbols_initialize(
    iree_allocator_t host_allocator,
    iree_hal_cuda_dynamic_symbols_t* out_syms) {
  IREE_TRACE_ZONE_BEGIN(z0);
  memset(out_syms, 0, sizeof(*out_syms));
  iree_status_t status = iree_dynamic_library_load_from_files(
      IREE_ARRAYSIZE(kCUDALoaderSearchNames), kCUDALoaderSearchNames,
      IREE_DYNAMIC_LIBRARY_FLAG_NONE, host_allocator, &out_syms->cuda_library);
  if (iree_status_is_not_found(status)) {
    iree_status_ignore(status);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "CUDA runtime library not available; ensure "
                            "installed and on path");
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_cuda_dynamic_symbols_resolve_all(out_syms);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_cuda_dynamic_symbols_deinitialize(out_syms);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_cuda_nccl_check_version(
    iree_dynamic_library_t* nccl_library) {
  IREE_ASSERT_ARGUMENT(nccl_library);

  ncclResult_t (*ncclGetVersion)(int*) = NULL;

  iree_status_t status = iree_dynamic_library_lookup_symbol(
      nccl_library, "ncclGetVersion", (void**)&ncclGetVersion);
  if (!iree_status_is_ok(status)) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "ncclGetVersion() not found");
  }

  // Check the NCCL version compatibility.
  int nccl_version = 0;
  ncclResult_t result = ncclGetVersion(&nccl_version);
  if (result != ncclSuccess) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "ncclGetVersion() failed (%d)", result);
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
  int required_minimum_version = NCCL_VERSION(NCCL_MAJOR, NCCL_MINOR, 0);
  if (major != NCCL_MAJOR || nccl_version < required_minimum_version) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "NCCL version is %d.%d.%d, but >=%d.%d and <%d is required", major,
        minor, patch, NCCL_MAJOR, NCCL_MINOR, NCCL_MAJOR + 1);
  }

  return iree_ok_status();
}

iree_status_t iree_hal_cuda_nccl_dynamic_symbols_initialize(
    iree_allocator_t host_allocator,
    iree_hal_cuda_dynamic_symbols_t* out_syms) {
  IREE_ASSERT_ARGUMENT(out_syms);
  if (!out_syms->cuda_library) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "CUDA dynamic symbols must be loaded prior to loading NCCL");
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  // TODO: rework this file - these functions are not safe as they mutate
  // each other's state in hard to follow ways.
  IREE_ASSERT(!out_syms->nccl_library);
  out_syms->nccl_library = NULL;

  // Attempt to load the NCCL shared library.
  iree_status_t status = iree_dynamic_library_load_from_files(
      IREE_ARRAYSIZE(kNCCLLoaderSearchNames), kNCCLLoaderSearchNames,
      IREE_DYNAMIC_LIBRARY_FLAG_NONE, host_allocator, &out_syms->nccl_library);
  if (iree_status_is_not_found(status)) {
    iree_status_ignore(status);
    status = iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "NCCL runtime library version %d.%d and greater not available; "
        "ensure installed and the shared library (nccl.dll/libnccl.so) "
        "is on your PATH/LD_LIBRARY_PATH.",
        NCCL_MAJOR, NCCL_MINOR);
  }

  if (iree_status_is_ok(status)) {
    // Check the version first before resolving all symbols. This makes sure
    // that we have the right version and all symbols are available at the
    // time of resolving.
    status = iree_hal_cuda_nccl_check_version(out_syms->nccl_library);
  }

  // Resolve all symbols; this will fail if any required symbols are missing.
  if (iree_status_is_ok(status)) {
    status = iree_hal_cuda_nccl_dynamic_symbols_resolve_all(out_syms);
  }

  if (!iree_status_is_ok(status)) {
    iree_dynamic_library_release(out_syms->nccl_library);
    out_syms->nccl_library = NULL;
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_cuda_dynamic_symbols_deinitialize(
    iree_hal_cuda_dynamic_symbols_t* syms) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_dynamic_library_release(syms->cuda_library);
  iree_dynamic_library_release(syms->nccl_library);
  memset(syms, 0, sizeof(*syms));

  IREE_TRACE_ZONE_END(z0);
}

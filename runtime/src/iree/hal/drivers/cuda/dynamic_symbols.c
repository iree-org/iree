// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/cuda/dynamic_symbols.h"

#include <string.h>

#include "iree/base/internal/dynamic_library.h"
#include "iree/base/target_platform.h"
#include "iree/base/tracing.h"

static const char* kCUDALoaderSearchNames[] = {
#if defined(IREE_PLATFORM_WINDOWS)
    "nvcuda.dll",
#else
    "libcuda.so",
#endif
};

#if IREE_HAL_CUDA_NCCL_ENABLE
static const char* kNCCLLoaderSearchNames[] = {
#if defined(IREE_PLATFORM_WINDOWS)
    "nccl.dll",
#else
    "libnccl.so",
#endif
};
#endif  // IREE_HAL_CUDA_NCCL_ENABLE

#define concat(A, B) A B

// Load CUDA entry points, prefer _v2 version if it exists.
static iree_status_t iree_hal_cuda_dynamic_symbols_resolve_all(
    iree_hal_cuda_dynamic_symbols_t* syms) {
#define CU_PFN_DECL(cudaSymbolName, ...)                                     \
  {                                                                          \
    static const char* kName = #cudaSymbolName;                              \
    IREE_RETURN_IF_ERROR(iree_dynamic_library_lookup_symbol(                 \
        syms->cuda_library, kName, (void**)&syms->cudaSymbolName));          \
    static const char* kNameV2 = concat(#cudaSymbolName, "_v2");             \
    void* funV2;                                                             \
    iree_dynamic_library_lookup_symbol(syms->cuda_library, kNameV2, &funV2); \
    if (funV2) syms->cudaSymbolName = funV2;                                 \
  }
#if IREE_HAL_CUDA_NCCL_ENABLE
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
#else
#define NCCL_PFN_DECL(ncclSymbolName, ...)
#define NCCL_PFN_DECL_STR_RETURN(ncclSymbolName, ...)
#endif
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
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "CUDA runtime library not available; ensure installed and on path");
  }
#if IREE_HAL_CUDA_NCCL_ENABLE
  status = iree_dynamic_library_load_from_files(
      IREE_ARRAYSIZE(kNCCLLoaderSearchNames), kNCCLLoaderSearchNames,
      IREE_DYNAMIC_LIBRARY_FLAG_NONE, host_allocator, &out_syms->nccl_library);
  if (iree_status_is_not_found(status)) {
    iree_status_ignore(status);
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "NCCL runtime library not available; ensure installed and on path");
  }
#endif  // IREE_HAL_CUDA_NCCL_ENABLE
  if (iree_status_is_ok(status)) {
    status = iree_hal_cuda_dynamic_symbols_resolve_all(out_syms);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_cuda_dynamic_symbols_deinitialize(out_syms);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_cuda_dynamic_symbols_deinitialize(
    iree_hal_cuda_dynamic_symbols_t* syms) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_dynamic_library_release(syms->cuda_library);
#if IREE_HAL_CUDA_NCCL_ENABLE
  iree_dynamic_library_release(syms->nccl_library);
#endif
  memset(syms, 0, sizeof(*syms));
  IREE_TRACE_ZONE_END(z0);
}

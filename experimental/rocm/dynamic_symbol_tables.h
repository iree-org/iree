// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

IREE_HAL_ROCM_REQUIRED_PFN_DECL(hipCtxCreate, hipCtx_t *, unsigned int,
                                hipDevice_t)
IREE_HAL_ROCM_REQUIRED_PFN_DECL(hipCtxDestroy, hipCtx_t)
IREE_HAL_ROCM_REQUIRED_PFN_DECL(hipDeviceGet, hipDevice_t *,
                                int)  // No direct, need to modify
IREE_HAL_ROCM_REQUIRED_PFN_DECL(hipGetDeviceCount, int *)
IREE_HAL_ROCM_OPTIONAL_PFN_DECL(hipGetDevicePropertiesR0600, hipDeviceProp_t *,
                                int)
IREE_HAL_ROCM_REQUIRED_PFN_DECL(hipDeviceGetName, char *, int,
                                hipDevice_t)  // No direct, need to modify
IREE_HAL_ROCM_REQUIRED_PFN_STR_DECL(
    hipGetErrorName,
    hipError_t)  // Unlike other functions hipGetErrorName(hipError_t) return
                 // const char* instead of hipError_t so it uses a different
                 // macro
IREE_HAL_ROCM_REQUIRED_PFN_STR_DECL(
    hipGetErrorString,
    hipError_t)  // Unlike other functions hipGetErrorName(hipError_t) return
                 // const char* instead of hipError_t so it uses a different
                 // macro
IREE_HAL_ROCM_REQUIRED_PFN_DECL(hipInit, unsigned int)
IREE_HAL_ROCM_REQUIRED_PFN_DECL(hipModuleLaunchKernel, hipFunction_t,
                                unsigned int, unsigned int, unsigned int,
                                unsigned int, unsigned int, unsigned int,
                                unsigned int, hipStream_t, void **, void **)
IREE_HAL_ROCM_REQUIRED_PFN_DECL(hipMemset, void *, int, size_t)
IREE_HAL_ROCM_REQUIRED_PFN_DECL(hipMemsetAsync, void *, int, size_t,
                                hipStream_t)
IREE_HAL_ROCM_REQUIRED_PFN_DECL(hipMemsetD32Async, void *, int, size_t,
                                hipStream_t)
IREE_HAL_ROCM_REQUIRED_PFN_DECL(hipMemsetD16Async, void *, short, size_t,
                                hipStream_t)
IREE_HAL_ROCM_REQUIRED_PFN_DECL(hipMemsetD8Async, void *, char, size_t,
                                hipStream_t)
IREE_HAL_ROCM_REQUIRED_PFN_DECL(hipMemcpy, void *, const void *, size_t,
                                hipMemcpyKind)
IREE_HAL_ROCM_REQUIRED_PFN_DECL(hipMemcpyAsync, void *, const void *, size_t,
                                hipMemcpyKind, hipStream_t)
IREE_HAL_ROCM_REQUIRED_PFN_DECL(hipMalloc, void **, size_t)
IREE_HAL_ROCM_REQUIRED_PFN_DECL(hipMallocManaged, hipDeviceptr_t *, size_t,
                                unsigned int)
IREE_HAL_ROCM_REQUIRED_PFN_DECL(hipFree, void *)
IREE_HAL_ROCM_REQUIRED_PFN_DECL(hipHostFree, void *)
IREE_HAL_ROCM_REQUIRED_PFN_DECL(hipMemAllocHost, void **, size_t, unsigned int)
IREE_HAL_ROCM_REQUIRED_PFN_DECL(hipHostGetDevicePointer, void **, void *,
                                unsigned int)
IREE_HAL_ROCM_REQUIRED_PFN_DECL(hipModuleGetFunction, hipFunction_t *,
                                hipModule_t, const char *)
IREE_HAL_ROCM_REQUIRED_PFN_DECL(hipModuleLoadDataEx, hipModule_t *,
                                const void *, unsigned int, hipJitOption *,
                                void **)
IREE_HAL_ROCM_REQUIRED_PFN_DECL(hipModuleLoadData, hipModule_t *, const void *)
IREE_HAL_ROCM_REQUIRED_PFN_DECL(hipModuleUnload, hipModule_t)
IREE_HAL_ROCM_REQUIRED_PFN_DECL(hipStreamCreateWithFlags, hipStream_t *,
                                unsigned int)
IREE_HAL_ROCM_REQUIRED_PFN_DECL(hipStreamDestroy, hipStream_t)
IREE_HAL_ROCM_REQUIRED_PFN_DECL(hipStreamSynchronize, hipStream_t)
IREE_HAL_ROCM_REQUIRED_PFN_DECL(hipStreamWaitEvent, hipStream_t, hipEvent_t,
                                unsigned int)
IREE_HAL_ROCM_REQUIRED_PFN_DECL(hipEventCreate, hipEvent_t *)
IREE_HAL_ROCM_REQUIRED_PFN_DECL(hipEventDestroy, hipEvent_t)
IREE_HAL_ROCM_REQUIRED_PFN_DECL(hipEventElapsedTime, float *, hipEvent_t,
                                hipEvent_t)
IREE_HAL_ROCM_REQUIRED_PFN_DECL(hipEventQuery, hipEvent_t)
IREE_HAL_ROCM_REQUIRED_PFN_DECL(hipEventRecord, hipEvent_t, hipStream_t)
IREE_HAL_ROCM_REQUIRED_PFN_DECL(hipEventSynchronize, hipEvent_t)
IREE_HAL_ROCM_REQUIRED_PFN_DECL(hipDeviceGetAttribute, int *,
                                hipDeviceAttribute_t, int)
IREE_HAL_ROCM_REQUIRED_PFN_DECL(hipFuncSetAttribute, const void *,
                                hipFuncAttribute, int)
IREE_HAL_ROCM_REQUIRED_PFN_DECL(hipDeviceGetUuid, hipUUID *, hipDevice_t)
IREE_HAL_ROCM_REQUIRED_PFN_DECL(hipDevicePrimaryCtxRetain, hipCtx_t *,
                                hipDevice_t)
IREE_HAL_ROCM_REQUIRED_PFN_DECL(hipCtxGetDevice, hipDevice_t *)
IREE_HAL_ROCM_REQUIRED_PFN_DECL(hipCtxSetCurrent, hipCtx_t)
IREE_HAL_ROCM_REQUIRED_PFN_DECL(hipDevicePrimaryCtxRelease, hipDevice_t)
IREE_HAL_ROCM_REQUIRED_PFN_DECL(hipMemPrefetchAsync, const void *, size_t, int,
                                hipStream_t)
IREE_HAL_ROCM_REQUIRED_PFN_DECL(hipMemcpyHtoDAsync, hipDeviceptr_t, void *,
                                size_t, hipStream_t)

// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

RC_PFN_DECL(hipCtxCreate, hipCtx_t *, unsigned int, hipDevice_t)
RC_PFN_DECL(hipCtxDestroy, hipCtx_t)
RC_PFN_DECL(hipDeviceGet, hipDevice_t *, int)  // No direct, need to modify
RC_PFN_DECL(hipGetDeviceCount, int *)
RC_PFN_DECL(hipGetDeviceProperties, hipDeviceProp_t *, int)
RC_PFN_DECL(hipDeviceGetName, char *, int,
            hipDevice_t)  // No direct, need to modify
RC_PFN_STR_DECL(
    hipGetErrorName,
    hipError_t)  // Unlike other functions hipGetErrorName(hipError_t) return
                 // const char* instead of hipError_t so it uses a different
                 // macro
RC_PFN_STR_DECL(
    hipGetErrorString,
    hipError_t)  // Unlike other functions hipGetErrorName(hipError_t) return
                 // const char* instead of hipError_t so it uses a different
                 // macro
RC_PFN_DECL(hipInit, unsigned int)
RC_PFN_DECL(hipModuleLaunchKernel, hipFunction_t, unsigned int, unsigned int,
            unsigned int, unsigned int, unsigned int, unsigned int,
            unsigned int, hipStream_t, void **, void **)
RC_PFN_DECL(hipMemset, void *, int, size_t)
RC_PFN_DECL(hipMemsetAsync, void *, int, size_t, hipStream_t)
RC_PFN_DECL(hipMemsetD32Async, void *, int, size_t, hipStream_t)
RC_PFN_DECL(hipMemsetD16Async, void *, short, size_t, hipStream_t)
RC_PFN_DECL(hipMemsetD8Async, void *, char, size_t, hipStream_t)
RC_PFN_DECL(hipMemcpy, void *, const void *, size_t, hipMemcpyKind)
RC_PFN_DECL(hipMemcpyAsync, void *, const void *, size_t, hipMemcpyKind,
            hipStream_t)
RC_PFN_DECL(hipMalloc, void **, size_t)
RC_PFN_DECL(hipMallocManaged, hipDeviceptr_t *, size_t, unsigned int)
RC_PFN_DECL(hipFree, void *)
RC_PFN_DECL(hipHostFree, void *)
RC_PFN_DECL(hipMemAllocHost, void **, size_t, unsigned int)
RC_PFN_DECL(hipHostGetDevicePointer, void **, void *, unsigned int)
RC_PFN_DECL(hipModuleGetFunction, hipFunction_t *, hipModule_t, const char *)
RC_PFN_DECL(hipModuleLoadDataEx, hipModule_t *, const void *, unsigned int,
            hipJitOption *, void **)
RC_PFN_DECL(hipModuleLoadData, hipModule_t *, const void *)
RC_PFN_DECL(hipModuleUnload, hipModule_t)
RC_PFN_DECL(hipStreamCreateWithFlags, hipStream_t *, unsigned int)
RC_PFN_DECL(hipStreamDestroy, hipStream_t)
RC_PFN_DECL(hipStreamSynchronize, hipStream_t)
RC_PFN_DECL(hipStreamWaitEvent, hipStream_t, hipEvent_t, unsigned int)
RC_PFN_DECL(hipEventCreate, hipEvent_t *)
RC_PFN_DECL(hipEventDestroy, hipEvent_t)
RC_PFN_DECL(hipEventElapsedTime, float *, hipEvent_t, hipEvent_t)
RC_PFN_DECL(hipEventQuery, hipEvent_t)
RC_PFN_DECL(hipEventRecord, hipEvent_t, hipStream_t)
RC_PFN_DECL(hipEventSynchronize, hipEvent_t)
RC_PFN_DECL(hipDeviceGetAttribute, int *, hipDeviceAttribute_t, int)
RC_PFN_DECL(hipFuncSetAttribute, const void *, hipFuncAttribute, int)

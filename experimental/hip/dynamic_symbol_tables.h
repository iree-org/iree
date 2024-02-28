// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// HIP symbols
//===----------------------------------------------------------------------===//

REQUIRED_IREE_HIP_PFN_DECL(hipCtxSetCurrent, hipCtx_t)
REQUIRED_IREE_HIP_PFN_DECL(hipDeviceGet, hipDevice_t *, int)
REQUIRED_IREE_HIP_PFN_DECL(hipDeviceGetAttribute, int *, hipDeviceAttribute_t,
                           int)
REQUIRED_IREE_HIP_PFN_DECL(hipDeviceGetName, char *, int, hipDevice_t)
REQUIRED_IREE_HIP_PFN_DECL(hipDeviceGetUuid, hipUUID *, hipDevice_t)
REQUIRED_IREE_HIP_PFN_DECL(hipDevicePrimaryCtxRelease, hipDevice_t)
REQUIRED_IREE_HIP_PFN_DECL(hipDevicePrimaryCtxRetain, hipCtx_t *, hipDevice_t)
OPTIONAL_IREE_HIP_PFN_DECL(hipDrvGraphAddMemcpyNode, hipGraphNode_t *,
                           hipGraph_t, const hipGraphNode_t *, size_t,
                           const HIP_MEMCPY3D *, hipCtx_t)
REQUIRED_IREE_HIP_PFN_DECL(hipEventCreate, hipEvent_t *)
REQUIRED_IREE_HIP_PFN_DECL(hipEventCreateWithFlags, hipEvent_t *, unsigned int)
REQUIRED_IREE_HIP_PFN_DECL(hipEventDestroy, hipEvent_t)
REQUIRED_IREE_HIP_PFN_DECL(hipEventElapsedTime, float *, hipEvent_t, hipEvent_t)
REQUIRED_IREE_HIP_PFN_DECL(hipEventQuery, hipEvent_t)
REQUIRED_IREE_HIP_PFN_DECL(hipEventRecord, hipEvent_t, hipStream_t)
REQUIRED_IREE_HIP_PFN_DECL(hipEventSynchronize, hipEvent_t)
REQUIRED_IREE_HIP_PFN_DECL(hipFree, void *)
REQUIRED_IREE_HIP_PFN_DECL(hipFreeAsync, void *, hipStream_t)
REQUIRED_IREE_HIP_PFN_DECL(hipFuncSetAttribute, const void *, hipFuncAttribute,
                           int)
REQUIRED_IREE_HIP_PFN_DECL(hipGetDeviceCount, int *)
REQUIRED_IREE_HIP_PFN_DECL(hipGetDeviceProperties, hipDeviceProp_t *, int)
// hipGetErrorName(hipError_t) and hipGetErrorString(hipError_t) return
// const char* instead of hipError_t so it uses a different macro.
REQUIRED_IREE_HIP_PFN_STR_DECL(hipGetErrorName, hipError_t)
REQUIRED_IREE_HIP_PFN_STR_DECL(hipGetErrorString, hipError_t)
REQUIRED_IREE_HIP_PFN_DECL(hipGraphAddEmptyNode, hipGraphNode_t *, hipGraph_t,
                           const hipGraphNode_t *, size_t)
REQUIRED_IREE_HIP_PFN_DECL(hipGraphAddKernelNode, hipGraphNode_t *, hipGraph_t,
                           const hipGraphNode_t *, size_t,
                           const hipKernelNodeParams *)
REQUIRED_IREE_HIP_PFN_DECL(hipGraphAddMemsetNode, hipGraphNode_t *, hipGraph_t,
                           const hipGraphNode_t *, size_t,
                           const hipMemsetParams *)
REQUIRED_IREE_HIP_PFN_DECL(hipGraphCreate, hipGraph_t *, unsigned int)
REQUIRED_IREE_HIP_PFN_DECL(hipGraphDestroy, hipGraph_t)
REQUIRED_IREE_HIP_PFN_DECL(hipGraphExecDestroy, hipGraphExec_t)
REQUIRED_IREE_HIP_PFN_DECL(hipGraphInstantiate, hipGraphExec_t *, hipGraph_t,
                           hipGraphNode_t *, char *, size_t)
REQUIRED_IREE_HIP_PFN_DECL(hipGraphLaunch, hipGraphExec_t, hipStream_t)
REQUIRED_IREE_HIP_PFN_DECL(hipHostFree, void *)
REQUIRED_IREE_HIP_PFN_DECL(hipHostGetDevicePointer, void **, void *,
                           unsigned int)
REQUIRED_IREE_HIP_PFN_DECL(hipHostMalloc, void **, size_t, unsigned int)
REQUIRED_IREE_HIP_PFN_DECL(hipHostRegister, void *, size_t, unsigned int)
REQUIRED_IREE_HIP_PFN_DECL(hipHostUnregister, void *)
REQUIRED_IREE_HIP_PFN_DECL(hipInit, unsigned int)
REQUIRED_IREE_HIP_PFN_DECL(hipLaunchHostFunc, hipStream_t, hipHostFn_t, void *)
REQUIRED_IREE_HIP_PFN_DECL(hipLaunchKernel, const void *, dim3, dim3, void **,
                           size_t, hipStream_t)
REQUIRED_IREE_HIP_PFN_DECL(hipMalloc, void **, size_t)
REQUIRED_IREE_HIP_PFN_DECL(hipMallocFromPoolAsync, void **, size_t,
                           hipMemPool_t, hipStream_t)
REQUIRED_IREE_HIP_PFN_DECL(hipMallocManaged, hipDeviceptr_t *, size_t,
                           unsigned int)
REQUIRED_IREE_HIP_PFN_DECL(hipMemcpy, void *, const void *, size_t,
                           hipMemcpyKind)
REQUIRED_IREE_HIP_PFN_DECL(hipMemcpyAsync, void *, const void *, size_t,
                           hipMemcpyKind, hipStream_t)
REQUIRED_IREE_HIP_PFN_DECL(hipMemcpyHtoDAsync, hipDeviceptr_t, void *, size_t,
                           hipStream_t)
REQUIRED_IREE_HIP_PFN_DECL(hipMemPoolCreate, hipMemPool_t *,
                           const hipMemPoolProps *)
REQUIRED_IREE_HIP_PFN_DECL(hipMemPoolDestroy, hipMemPool_t)
REQUIRED_IREE_HIP_PFN_DECL(hipMemPoolGetAttribute, hipMemPool_t, hipMemPoolAttr,
                           void *)
REQUIRED_IREE_HIP_PFN_DECL(hipMemPoolSetAttribute, hipMemPool_t, hipMemPoolAttr,
                           void *)
REQUIRED_IREE_HIP_PFN_DECL(hipMemPoolTrimTo, hipMemPool_t, size_t)
REQUIRED_IREE_HIP_PFN_DECL(hipMemPrefetchAsync, const void *, size_t, int,
                           hipStream_t)
REQUIRED_IREE_HIP_PFN_DECL(hipMemset, void *, int, size_t)
REQUIRED_IREE_HIP_PFN_DECL(hipMemsetAsync, void *, int, size_t, hipStream_t)
REQUIRED_IREE_HIP_PFN_DECL(hipMemsetD8Async, void *, char, size_t, hipStream_t)
REQUIRED_IREE_HIP_PFN_DECL(hipMemsetD16Async, void *, short, size_t,
                           hipStream_t)
REQUIRED_IREE_HIP_PFN_DECL(hipMemsetD32Async, void *, int, size_t, hipStream_t)
REQUIRED_IREE_HIP_PFN_DECL(hipModuleGetFunction, hipFunction_t *, hipModule_t,
                           const char *)
REQUIRED_IREE_HIP_PFN_DECL(hipModuleLaunchKernel, hipFunction_t, unsigned int,
                           unsigned int, unsigned int, unsigned int,
                           unsigned int, unsigned int, unsigned int,
                           hipStream_t, void **, void **)
REQUIRED_IREE_HIP_PFN_DECL(hipModuleLoadData, hipModule_t *, const void *)
REQUIRED_IREE_HIP_PFN_DECL(hipModuleLoadDataEx, hipModule_t *, const void *,
                           unsigned int, hipJitOption *, void **)
REQUIRED_IREE_HIP_PFN_DECL(hipModuleUnload, hipModule_t)
REQUIRED_IREE_HIP_PFN_DECL(hipStreamCreateWithFlags, hipStream_t *,
                           unsigned int)
REQUIRED_IREE_HIP_PFN_DECL(hipStreamDestroy, hipStream_t)
REQUIRED_IREE_HIP_PFN_DECL(hipStreamSynchronize, hipStream_t)
REQUIRED_IREE_HIP_PFN_DECL(hipStreamWaitEvent, hipStream_t, hipEvent_t,
                           unsigned int)

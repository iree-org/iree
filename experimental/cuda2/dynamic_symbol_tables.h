// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// CUDA symbols
//===----------------------------------------------------------------------===//
IREE_CU_PFN_DECL(cuCtxCreate, CUcontext*, unsigned int, CUdevice)
IREE_CU_PFN_DECL(cuCtxDestroy, CUcontext)
IREE_CU_PFN_DECL(cuDevicePrimaryCtxRetain, CUcontext*, CUdevice)
IREE_CU_PFN_DECL(cuDevicePrimaryCtxRelease, CUdevice)
IREE_CU_PFN_DECL(cuCtxSetCurrent, CUcontext)
IREE_CU_PFN_DECL(cuCtxPushCurrent, CUcontext)
IREE_CU_PFN_DECL(cuCtxPopCurrent, CUcontext*)
IREE_CU_PFN_DECL(cuDeviceGet, CUdevice*, int)
IREE_CU_PFN_DECL(cuDeviceGetCount, int*)
IREE_CU_PFN_DECL(cuDeviceGetName, char*, int, CUdevice)
IREE_CU_PFN_DECL(cuDeviceGetAttribute, int*, CUdevice_attribute, CUdevice)
IREE_CU_PFN_DECL(cuDeviceGetUuid, CUuuid*, CUdevice)
IREE_CU_PFN_DECL(cuEventCreate, CUevent*, unsigned int)
IREE_CU_PFN_DECL(cuEventDestroy, CUevent)
IREE_CU_PFN_DECL(cuEventElapsedTime, float*, CUevent, CUevent)
IREE_CU_PFN_DECL(cuEventQuery, CUevent)
IREE_CU_PFN_DECL(cuEventRecord, CUevent, CUstream)
IREE_CU_PFN_DECL(cuEventSynchronize, CUevent)
IREE_CU_PFN_DECL(cuGetErrorName, CUresult, const char**)
IREE_CU_PFN_DECL(cuGetErrorString, CUresult, const char**)
IREE_CU_PFN_DECL(cuGraphAddMemcpyNode, CUgraphNode*, CUgraph,
                 const CUgraphNode*, size_t, const CUDA_MEMCPY3D*, CUcontext)
IREE_CU_PFN_DECL(cuGraphAddMemsetNode, CUgraphNode*, CUgraph,
                 const CUgraphNode*, size_t, const CUDA_MEMSET_NODE_PARAMS*,
                 CUcontext)
IREE_CU_PFN_DECL(cuGraphAddKernelNode, CUgraphNode*, CUgraph,
                 const CUgraphNode*, size_t, const CUDA_KERNEL_NODE_PARAMS*)
IREE_CU_PFN_DECL(cuGraphCreate, CUgraph*, unsigned int)
IREE_CU_PFN_DECL(cuGraphDestroy, CUgraph)
IREE_CU_PFN_DECL(cuGraphExecDestroy, CUgraphExec)
IREE_CU_PFN_DECL(cuGraphGetNodes, CUgraph, CUgraphNode*, size_t*)
IREE_CU_PFN_DECL(cuGraphInstantiate, CUgraphExec*, CUgraph, CUgraphNode*, char*,
                 size_t)
IREE_CU_PFN_DECL(cuGraphLaunch, CUgraphExec, CUstream)
IREE_CU_PFN_DECL(cuInit, unsigned int)
IREE_CU_PFN_DECL(cuMemAllocManaged, CUdeviceptr*, size_t, unsigned int)
IREE_CU_PFN_DECL(cuMemPrefetchAsync, CUdeviceptr, size_t, CUdevice, CUstream)
IREE_CU_PFN_DECL(cuMemAlloc, CUdeviceptr*, size_t)
IREE_CU_PFN_DECL(cuMemFree, CUdeviceptr)
IREE_CU_PFN_DECL(cuMemFreeHost, void*)
IREE_CU_PFN_DECL(cuMemHostAlloc, void**, size_t, unsigned int)
IREE_CU_PFN_DECL(cuMemHostRegister, void*, size_t, unsigned int)
IREE_CU_PFN_DECL(cuMemHostUnregister, void*)
IREE_CU_PFN_DECL(cuMemHostGetDevicePointer, CUdeviceptr*, void*, unsigned int)
IREE_CU_PFN_DECL(cuModuleGetFunction, CUfunction*, CUmodule, const char*)
IREE_CU_PFN_DECL(cuModuleLoadDataEx, CUmodule*, const void*, unsigned int,
                 CUjit_option*, void**)
IREE_CU_PFN_DECL(cuModuleUnload, CUmodule)
IREE_CU_PFN_DECL(cuStreamCreate, CUstream*, unsigned int)
IREE_CU_PFN_DECL(cuStreamDestroy, CUstream)
IREE_CU_PFN_DECL(cuStreamSynchronize, CUstream)
IREE_CU_PFN_DECL(cuStreamWaitEvent, CUstream, CUevent, unsigned int)
IREE_CU_PFN_DECL(cuMemsetD32Async, unsigned long long, unsigned int, size_t,
                 CUstream)
IREE_CU_PFN_DECL(cuMemsetD16Async, unsigned long long, unsigned short, size_t,
                 CUstream)
IREE_CU_PFN_DECL(cuMemsetD8Async, unsigned long long, unsigned char, size_t,
                 CUstream)
IREE_CU_PFN_DECL(cuMemcpyAsync, CUdeviceptr, CUdeviceptr, size_t, CUstream)
IREE_CU_PFN_DECL(cuMemcpyHtoDAsync_v2, CUdeviceptr, const void*, size_t,
                 CUstream)
IREE_CU_PFN_DECL(cuFuncSetAttribute, CUfunction, CUfunction_attribute, int)
IREE_CU_PFN_DECL(cuLaunchKernel, CUfunction, unsigned int, unsigned int,
                 unsigned int, unsigned int, unsigned int, unsigned int,
                 unsigned int, CUstream, void**, void**)

//===----------------------------------------------------------------------===//
// NCCL symbols
//===----------------------------------------------------------------------===//
IREE_NCCL_PFN_DECL(ncclGetVersion, int*)
IREE_NCCL_PFN_DECL(ncclGetUniqueId, ncclUniqueId*)
IREE_NCCL_PFN_DECL(ncclCommInitRankConfig, ncclComm_t*, int, ncclUniqueId, int,
                   ncclConfig_t*)
IREE_NCCL_PFN_DECL(ncclCommInitRank, ncclComm_t*, int, ncclUniqueId, int)
IREE_NCCL_PFN_DECL(ncclCommInitAll, ncclComm_t*, int, const int*)
IREE_NCCL_PFN_DECL(ncclCommSplit, ncclComm_t, int, int, ncclComm_t*,
                   ncclConfig_t*)
IREE_NCCL_PFN_DECL(ncclCommFinalize, ncclComm_t)
IREE_NCCL_PFN_DECL(ncclCommDestroy, ncclComm_t)
IREE_NCCL_PFN_DECL(ncclCommAbort, ncclComm_t)
IREE_NCCL_PFN_DECL_STR_RETURN(ncclGetErrorString, ncclResult_t)
IREE_NCCL_PFN_DECL_STR_RETURN(ncclGetLastError, ncclComm_t)
IREE_NCCL_PFN_DECL(ncclCommGetAsyncError, ncclComm_t, ncclResult_t*)
IREE_NCCL_PFN_DECL(ncclCommCount, const ncclComm_t, int*)
IREE_NCCL_PFN_DECL(ncclCommCuDevice, const ncclComm_t, int*)
IREE_NCCL_PFN_DECL(ncclCommUserRank, const ncclComm_t, int*)
IREE_NCCL_PFN_DECL(ncclRedOpCreatePreMulSum, ncclRedOp_t*, void*,
                   ncclDataType_t, ncclScalarResidence_t, ncclComm_t)
IREE_NCCL_PFN_DECL(ncclRedOpDestroy, ncclRedOp_t, ncclComm_t)
IREE_NCCL_PFN_DECL(ncclReduce, const void*, void*, size_t, ncclDataType_t,
                   ncclRedOp_t, int, ncclComm_t, cudaStream_t)
IREE_NCCL_PFN_DECL(ncclBcast, void*, size_t, ncclDataType_t, int, ncclComm_t,
                   cudaStream_t)
IREE_NCCL_PFN_DECL(ncclBroadcast, const void*, void*, size_t, ncclDataType_t,
                   int, ncclComm_t, cudaStream_t)
IREE_NCCL_PFN_DECL(ncclAllReduce, const void*, void*, size_t, ncclDataType_t,
                   ncclRedOp_t, ncclComm_t, cudaStream_t)
IREE_NCCL_PFN_DECL(ncclReduceScatter, const void*, void*, size_t,
                   ncclDataType_t, ncclRedOp_t, ncclComm_t, cudaStream_t)
IREE_NCCL_PFN_DECL(ncclAllGather, const void*, void*, size_t, ncclDataType_t,
                   ncclComm_t, cudaStream_t)
IREE_NCCL_PFN_DECL(ncclSend, const void*, size_t, ncclDataType_t, int,
                   ncclComm_t, cudaStream_t)
IREE_NCCL_PFN_DECL(ncclRecv, void*, size_t, ncclDataType_t, int, ncclComm_t,
                   cudaStream_t)
IREE_NCCL_PFN_DECL(ncclGroupStart)
IREE_NCCL_PFN_DECL(ncclGroupEnd)

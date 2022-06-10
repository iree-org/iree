// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

CU_PFN_DECL(cuCtxCreate, CUcontext*, unsigned int, CUdevice)
CU_PFN_DECL(cuCtxDestroy, CUcontext)
CU_PFN_DECL(cuDeviceGet, CUdevice*, int)
CU_PFN_DECL(cuDeviceGetCount, int*)
CU_PFN_DECL(cuDeviceGetName, char*, int, CUdevice)
CU_PFN_DECL(cuDeviceGetAttribute, int*, CUdevice_attribute, CUdevice)
CU_PFN_DECL(cuDeviceGetUuid_v2, CUuuid*, CUdevice)
CU_PFN_DECL(cuGetErrorName, CUresult, const char**)
CU_PFN_DECL(cuGetErrorString, CUresult, const char**)
CU_PFN_DECL(cuGraphAddMemcpyNode, CUgraphNode*, CUgraph, const CUgraphNode*,
            size_t, const CUDA_MEMCPY3D*, CUcontext)
CU_PFN_DECL(cuGraphAddMemsetNode, CUgraphNode*, CUgraph, const CUgraphNode*,
            size_t, const CUDA_MEMSET_NODE_PARAMS*, CUcontext)
CU_PFN_DECL(cuGraphAddKernelNode, CUgraphNode*, CUgraph, const CUgraphNode*,
            size_t, const CUDA_KERNEL_NODE_PARAMS*)
CU_PFN_DECL(cuGraphCreate, CUgraph*, unsigned int)
CU_PFN_DECL(cuGraphDestroy, CUgraph)
CU_PFN_DECL(cuGraphExecDestroy, CUgraphExec)
CU_PFN_DECL(cuGraphGetNodes, CUgraph, CUgraphNode*, size_t*)
CU_PFN_DECL(cuGraphInstantiate, CUgraphExec*, CUgraph, CUgraphNode*, char*,
            size_t)
CU_PFN_DECL(cuGraphLaunch, CUgraphExec, CUstream)
CU_PFN_DECL(cuInit, unsigned int)
CU_PFN_DECL(cuMemAllocManaged, CUdeviceptr*, size_t, unsigned int)
CU_PFN_DECL(cuMemPrefetchAsync, CUdeviceptr, size_t, CUdevice, CUstream)
CU_PFN_DECL(cuMemAlloc, CUdeviceptr*, size_t)
CU_PFN_DECL(cuMemFree, CUdeviceptr)
CU_PFN_DECL(cuMemFreeHost, void*)
CU_PFN_DECL(cuMemHostAlloc, void**, size_t, unsigned int)
CU_PFN_DECL(cuMemHostGetDevicePointer, CUdeviceptr*, void*, unsigned int)
CU_PFN_DECL(cuModuleGetFunction, CUfunction*, CUmodule, const char*)
CU_PFN_DECL(cuModuleLoadDataEx, CUmodule*, const void*, unsigned int,
            CUjit_option*, void**)
CU_PFN_DECL(cuModuleUnload, CUmodule)
CU_PFN_DECL(cuStreamCreate, CUstream*, unsigned int)
CU_PFN_DECL(cuStreamDestroy, CUstream)
CU_PFN_DECL(cuStreamSynchronize, CUstream)
CU_PFN_DECL(cuStreamWaitEvent, CUstream, CUevent, unsigned int)
CU_PFN_DECL(cuMemsetD32Async, unsigned long long, unsigned int, size_t,
            CUstream)
CU_PFN_DECL(cuMemsetD16Async, unsigned long long, unsigned short, size_t,
            CUstream)
CU_PFN_DECL(cuMemsetD8Async, unsigned long long, unsigned char, size_t,
            CUstream)
CU_PFN_DECL(cuMemcpyAsync, CUdeviceptr, CUdeviceptr, size_t, CUstream)
CU_PFN_DECL(cuMemcpyHtoDAsync_v2, CUdeviceptr, const void*, size_t, CUstream)
CU_PFN_DECL(cuFuncSetAttribute, CUfunction, CUfunction_attribute, int)
CU_PFN_DECL(cuLaunchKernel, CUfunction, unsigned int, unsigned int,
            unsigned int, unsigned int, unsigned int, unsigned int,
            unsigned int, CUstream, void**, void**)

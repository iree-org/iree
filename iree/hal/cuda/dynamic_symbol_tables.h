// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

CU_PFN_DECL(cuCtxCreate, CUcontext*, unsigned int, CUdevice)
CU_PFN_DECL(cuCtxDestroy, CUcontext)
CU_PFN_DECL(cuDeviceGet, CUdevice*, int)
CU_PFN_DECL(cuDeviceGetCount, int*)
CU_PFN_DECL(cuDeviceGetName, char*, int, CUdevice)
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

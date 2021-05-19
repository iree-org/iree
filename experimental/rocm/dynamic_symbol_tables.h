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

RC_PFN_DECL(hipCtxCreate, hipCtx_t *, unsigned int, hipDevice_t)
RC_PFN_DECL(hipCtxDestroy, hipCtx_t)
RC_PFN_DECL(hipDeviceGet, hipDevice_t *, int)  // No direct, need to modify
RC_PFN_DECL(hipGetDeviceCount, int *)
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
RC_PFN_DECL(hipMemcpy, void *, const void *, size_t, hipMemcpyKind)
RC_PFN_DECL(hipMemcpyAsync, void *, const void *, size_t, hipMemcpyKind,
            hipStream_t)
RC_PFN_DECL(hipMalloc, void **, size_t)
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

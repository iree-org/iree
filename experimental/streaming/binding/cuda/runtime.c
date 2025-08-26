// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/streaming/binding/cuda/runtime.h"

#include <stddef.h>
#include <string.h>

#include "experimental/streaming/binding/cuda/driver.h"
#include "experimental/streaming/internal.h"

// Thread-local storage for the last CUDA error.
static iree_thread_local cudaError_t iree_cuda_thread_error = cudaSuccess;

// Sets the last error for the current thread.
static void iree_cuda_set_error(cudaError_t error) {
  iree_cuda_thread_error = error;
}

//===----------------------------------------------------------------------===//
// Initialization and Version Management
//===----------------------------------------------------------------------===//

cudaError_t CUDAAPI cudaDriverGetVersion(int* driverVersion) {
  // TODO: Implement driver version query.
  if (!driverVersion) return cudaErrorInvalidValue;
  *driverVersion = 0;
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaRuntimeGetVersion(int* runtimeVersion) {
  // TODO: Implement runtime version query.
  if (!runtimeVersion) return cudaErrorInvalidValue;
  *runtimeVersion = CUDART_VERSION;
  return cudaSuccess;
}

cudaError_t CUDAAPI
cudaGetDriverEntryPoint(const char* symbol, void** funcPtr, uint64_t flags,
                        cudaDriverEntryPointQueryResult* driverStatus) {
  // TODO: Implement get driver entry point.
  if (!funcPtr) return cudaErrorInvalidValue;
  *funcPtr = NULL;
  if (driverStatus) *driverStatus = cudaDriverEntryPointSymbolNotFound;
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaGetDriverEntryPointByVersion(
    const char* symbol, void** funcPtr, unsigned int cudaVersion,
    uint64_t flags, cudaDriverEntryPointQueryResult* driverStatus) {
  // TODO: Implement get driver entry point by version.
  if (!funcPtr) return cudaErrorInvalidValue;
  *funcPtr = NULL;
  if (driverStatus) *driverStatus = cudaDriverEntryPointSymbolNotFound;
  return cudaErrorNotSupported;
}

//===----------------------------------------------------------------------===//
// Device Management
//===----------------------------------------------------------------------===//

cudaError_t CUDAAPI cudaGetDevice(int* device) {
  // TODO: Implement get current device.
  if (!device) return cudaErrorInvalidValue;
  *device = 0;
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaSetDevice(int device) {
  // TODO: Implement set current device.
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaGetDeviceCount(int* count) {
  // TODO: Implement get device count.
  if (!count) return cudaErrorInvalidValue;
  *count = 0;
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaGetDeviceProperties(cudaDeviceProp* prop, int device) {
  // TODO: Implement get device properties.
  if (!prop) return cudaErrorInvalidValue;
  memset(prop, 0, sizeof(*prop));
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr,
                                           int device) {
  // TODO: Implement get device attribute.
  if (!value) return cudaErrorInvalidValue;
  *value = 0;
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaDeviceGetDefaultMemPool(cudaMemPool_t* memPool,
                                                int device) {
  // TODO: Implement get default memory pool.
  if (!memPool) return cudaErrorInvalidValue;
  *memPool = NULL;
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaDeviceSetMemPool(int device, cudaMemPool_t memPool) {
  // TODO: Implement set memory pool.
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaDeviceGetMemPool(cudaMemPool_t* memPool, int device) {
  // TODO: Implement get memory pool.
  if (!memPool) return cudaErrorInvalidValue;
  *memPool = NULL;
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaDeviceSetCacheConfig(cudaFuncCache cacheConfig) {
  // TODO: Implement set cache config.
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaDeviceGetCacheConfig(cudaFuncCache* pCacheConfig) {
  // TODO: Implement get cache config.
  if (!pCacheConfig) return cudaErrorInvalidValue;
  *pCacheConfig = cudaFuncCachePreferNone;
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaChooseDevice(int* device, const cudaDeviceProp* prop) {
  // TODO: Implement choose device.
  if (!device || !prop) return cudaErrorInvalidValue;
  *device = 0;
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaSetDeviceFlags(unsigned int flags) {
  // TODO: Implement set device flags.
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaGetDeviceFlags(unsigned int* flags) {
  // TODO: Implement get device flags.
  if (!flags) return cudaErrorInvalidValue;
  *flags = 0;
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaDeviceReset(void) {
  // TODO: Implement device reset.
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaDeviceSynchronize(void) {
  // TODO: Implement device synchronize.
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaDeviceGetLimit(size_t* pValue, cudaLimit limit) {
  // TODO: Implement get device limit.
  if (!pValue) return cudaErrorInvalidValue;
  *pValue = 0;
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaDeviceSetLimit(cudaLimit limit, size_t value) {
  // TODO: Implement set device limit.
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaDeviceCanAccessPeer(int* canAccessPeer, int device,
                                            int peerDevice) {
  // TODO: Implement can access peer.
  if (!canAccessPeer) return cudaErrorInvalidValue;
  *canAccessPeer = 0;
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaDeviceEnablePeerAccess(int peerDevice,
                                               unsigned int flags) {
  // TODO: Implement enable peer access.
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaDeviceDisablePeerAccess(int peerDevice) {
  // TODO: Implement disable peer access.
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaDeviceGetPCIBusId(char* pciBusId, int len, int device) {
  // TODO: Implement get PCI bus ID.
  if (!pciBusId || len <= 0) return cudaErrorInvalidValue;
  pciBusId[0] = '\0';
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaDeviceGetByPCIBusId(int* device, const char* pciBusId) {
  // TODO: Implement get device by PCI bus ID.
  if (!device || !pciBusId) return cudaErrorInvalidValue;
  *device = 0;
  return cudaErrorNotSupported;
}

//===----------------------------------------------------------------------===//
// Error Handling
//===----------------------------------------------------------------------===//

cudaError_t CUDAAPI cudaGetLastError(void) {
  cudaError_t error = iree_cuda_thread_error;
  iree_cuda_thread_error = cudaSuccess;
  return error;
}

cudaError_t CUDAAPI cudaPeekAtLastError(void) { return iree_cuda_thread_error; }

const char* CUDAAPI cudaGetErrorString(cudaError_t error) {
  // TODO: Implement full error string mapping.
  switch (error) {
    case cudaSuccess:
      return "no error";
    case cudaErrorInvalidValue:
      return "invalid argument value";
    case cudaErrorMemoryAllocation:
      return "out of memory";
    case cudaErrorInitializationError:
      return "initialization error";
    case cudaErrorNotSupported:
      return "operation not supported";
    default:
      return "unknown error";
  }
}

const char* CUDAAPI cudaGetErrorName(cudaError_t error) {
  // TODO: Implement full error name mapping.
  switch (error) {
    case cudaSuccess:
      return "cudaSuccess";
    case cudaErrorInvalidValue:
      return "cudaErrorInvalidValue";
    case cudaErrorMemoryAllocation:
      return "cudaErrorMemoryAllocation";
    case cudaErrorInitializationError:
      return "cudaErrorInitializationError";
    case cudaErrorNotSupported:
      return "cudaErrorNotSupported";
    default:
      return "cudaErrorUnknown";
  }
}

//===----------------------------------------------------------------------===//
// Memory Management - Basic
//===----------------------------------------------------------------------===//

cudaError_t CUDAAPI cudaMalloc(void** devPtr, size_t size) {
  // TODO: Implement device memory allocation.
  if (!devPtr) return cudaErrorInvalidValue;
  *devPtr = NULL;
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaFree(void* devPtr) {
  // TODO: Implement device memory free.
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaMallocHost(void** ptr, size_t size) {
  // TODO: Implement host memory allocation.
  if (!ptr) return cudaErrorInvalidValue;
  *ptr = NULL;
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaFreeHost(void* ptr) {
  // TODO: Implement host memory free.
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaMallocManaged(void** devPtr, size_t size,
                                      unsigned int flags) {
  // TODO: Implement managed memory allocation.
  if (!devPtr) return cudaErrorInvalidValue;
  *devPtr = NULL;
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaMallocPitch(void** devPtr, size_t* pitch, size_t width,
                                    size_t height) {
  // TODO: Implement pitched memory allocation.
  if (!devPtr || !pitch) return cudaErrorInvalidValue;
  *devPtr = NULL;
  *pitch = 0;
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaHostAlloc(void** pHost, size_t size,
                                  unsigned int flags) {
  // TODO: Implement host allocation with flags.
  if (!pHost) return cudaErrorInvalidValue;
  *pHost = NULL;
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaHostRegister(void* ptr, size_t size,
                                     unsigned int flags) {
  // TODO: Implement host memory registration.
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaHostUnregister(void* ptr) {
  // TODO: Implement host memory unregistration.
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaHostGetDevicePointer(void** pDevice, void* pHost,
                                             unsigned int flags) {
  // TODO: Implement get device pointer from host.
  if (!pDevice) return cudaErrorInvalidValue;
  *pDevice = NULL;
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaHostGetFlags(unsigned int* pFlags, void* pHost) {
  // TODO: Implement get host flags.
  if (!pFlags) return cudaErrorInvalidValue;
  *pFlags = 0;
  return cudaErrorNotSupported;
}

//===----------------------------------------------------------------------===//
// Memory Management - Transfers
//===----------------------------------------------------------------------===//

cudaError_t CUDAAPI cudaMemcpy(void* dst, const void* src, size_t count,
                               cudaMemcpyKind kind) {
  // TODO: Implement synchronous memory copy.
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaMemcpyAsync(void* dst, const void* src, size_t count,
                                    cudaMemcpyKind kind, cudaStream_t stream) {
  // TODO: Implement asynchronous memory copy.
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaMemcpy2D(void* dst, size_t dpitch, const void* src,
                                 size_t spitch, size_t width, size_t height,
                                 cudaMemcpyKind kind) {
  // TODO: Implement 2D memory copy.
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaMemcpy2DAsync(void* dst, size_t dpitch, const void* src,
                                      size_t spitch, size_t width,
                                      size_t height, cudaMemcpyKind kind,
                                      cudaStream_t stream) {
  // TODO: Implement asynchronous 2D memory copy.
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaMemcpy3D(const cudaMemcpy3DParms* p) {
  // TODO: Implement 3D memory copy.
  if (!p) return cudaErrorInvalidValue;
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaMemcpy3DAsync(const cudaMemcpy3DParms* p,
                                      cudaStream_t stream) {
  // TODO: Implement asynchronous 3D memory copy.
  if (!p) return cudaErrorInvalidValue;
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaMemset(void* devPtr, int value, size_t count) {
  // TODO: Implement memory set.
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaMemsetAsync(void* devPtr, int value, size_t count,
                                    cudaStream_t stream) {
  // TODO: Implement asynchronous memory set.
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaMemset2D(void* devPtr, size_t pitch, int value,
                                 size_t width, size_t height) {
  // TODO: Implement 2D memory set.
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaMemset2DAsync(void* devPtr, size_t pitch, int value,
                                      size_t width, size_t height,
                                      cudaStream_t stream) {
  // TODO: Implement asynchronous 2D memory set.
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaMemset3D(cudaPitchedPtr pitchedDevPtr, int value,
                                 cudaExtent extent) {
  // TODO: Implement 3D memory set.
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaMemset3DAsync(cudaPitchedPtr pitchedDevPtr, int value,
                                      cudaExtent extent, cudaStream_t stream) {
  // TODO: Implement asynchronous 3D memory set.
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaMemGetInfo(size_t* free, size_t* total) {
  // TODO: Implement get memory info.
  if (free) *free = 0;
  if (total) *total = 0;
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaMemPrefetchAsync(const void* devPtr, size_t count,
                                         cudaMemLocation location,
                                         unsigned int flags,
                                         cudaStream_t stream) {
  // TODO: Implement memory prefetch.
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaMemAdvise(const void* devPtr, size_t count,
                                  cudaMemoryAdvise advice, int device) {
  // TODO: Implement memory advise.
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaMemGetAddressRange(void** pbase, size_t* psize,
                                           void* devPtr) {
  // TODO: Implement get address range.
  if (pbase) *pbase = NULL;
  if (psize) *psize = 0;
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaPointerGetAttributes(cudaPointerAttributes* attributes,
                                             const void* ptr) {
  // TODO: Implement get pointer attributes.
  if (!attributes) return cudaErrorInvalidValue;
  memset(attributes, 0, sizeof(*attributes));
  return cudaErrorNotSupported;
}

//===----------------------------------------------------------------------===//
// Stream Management
//===----------------------------------------------------------------------===//

cudaError_t CUDAAPI cudaStreamCreate(cudaStream_t* pStream) {
  // TODO: Implement stream creation.
  if (!pStream) return cudaErrorInvalidValue;
  *pStream = NULL;
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaStreamCreateWithFlags(cudaStream_t* pStream,
                                              unsigned int flags) {
  // TODO: Implement stream creation with flags.
  if (!pStream) return cudaErrorInvalidValue;
  *pStream = NULL;
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaStreamCreateWithPriority(cudaStream_t* pStream,
                                                 unsigned int flags,
                                                 int priority) {
  // TODO: Implement stream creation with priority.
  if (!pStream) return cudaErrorInvalidValue;
  *pStream = NULL;
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaStreamDestroy(cudaStream_t stream) {
  // TODO: Implement stream destruction.
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaStreamSynchronize(cudaStream_t stream) {
  // TODO: Implement stream synchronization.
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaStreamQuery(cudaStream_t stream) {
  // TODO: Implement stream query.
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event,
                                        unsigned int flags) {
  // TODO: Implement stream wait event.
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaStreamAddCallback(cudaStream_t stream,
                                          cudaStreamCallback_t callback,
                                          void* userData, unsigned int flags) {
  // TODO: Implement stream add callback.
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaStreamGetFlags(cudaStream_t hStream,
                                       unsigned int* flags) {
  // TODO: Implement get stream flags.
  if (!flags) return cudaErrorInvalidValue;
  *flags = 0;
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaStreamGetPriority(cudaStream_t hStream, int* priority) {
  // TODO: Implement get stream priority.
  if (!priority) return cudaErrorInvalidValue;
  *priority = 0;
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaStreamAttachMemAsync(cudaStream_t stream, void* devPtr,
                                             size_t length,
                                             unsigned int flags) {
  // TODO: Implement stream attach memory.
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaStreamBeginCapture(cudaStream_t stream,
                                           cudaStreamCaptureMode mode) {
  // TODO: Implement stream begin capture.
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaStreamEndCapture(cudaStream_t stream,
                                         cudaGraph_t* pGraph) {
  // TODO: Implement stream end capture.
  if (!pGraph) return cudaErrorInvalidValue;
  *pGraph = NULL;
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaStreamIsCapturing(
    cudaStream_t stream, cudaStreamCaptureStatus* pCaptureStatus) {
  // TODO: Implement stream is capturing.
  if (!pCaptureStatus) return cudaErrorInvalidValue;
  *pCaptureStatus = cudaStreamCaptureStatusNone;
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaStreamGetCaptureInfo(
    cudaStream_t stream, cudaStreamCaptureStatus* captureStatus_out,
    unsigned long long* id_out, cudaGraph_t* graph_out,
    const cudaGraphNode_t** dependencies_out,
    const cudaGraphEdgeData** edgeData_out, size_t* numDependencies_out) {
  // TODO: Implement stream get capture info.
  if (captureStatus_out) *captureStatus_out = cudaStreamCaptureStatusNone;
  if (id_out) *id_out = 0;
  if (graph_out) *graph_out = NULL;
  if (dependencies_out) *dependencies_out = NULL;
  if (edgeData_out) *edgeData_out = NULL;
  if (numDependencies_out) *numDependencies_out = 0;
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaStreamUpdateCaptureDependencies(
    cudaStream_t stream, cudaGraphNode_t* dependencies, size_t numDependencies,
    unsigned int flags) {
  // TODO: Implement stream update capture dependencies.
  return cudaErrorNotSupported;
}

//===----------------------------------------------------------------------===//
// Event Management
//===----------------------------------------------------------------------===//

cudaError_t CUDAAPI cudaEventCreate(cudaEvent_t* event) {
  // TODO: Implement event creation.
  if (!event) return cudaErrorInvalidValue;
  *event = NULL;
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaEventCreateWithFlags(cudaEvent_t* event,
                                             unsigned int flags) {
  // TODO: Implement event creation with flags.
  if (!event) return cudaErrorInvalidValue;
  *event = NULL;
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaEventDestroy(cudaEvent_t event) {
  // TODO: Implement event destruction.
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
  // TODO: Implement event record.
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaEventRecordWithFlags(cudaEvent_t event,
                                             cudaStream_t stream,
                                             unsigned int flags) {
  // TODO: Implement event record with flags.
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaEventSynchronize(cudaEvent_t event) {
  // TODO: Implement event synchronization.
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaEventQuery(cudaEvent_t event) {
  // TODO: Implement event query.
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaEventElapsedTime(float* ms, cudaEvent_t start,
                                         cudaEvent_t end) {
  // TODO: Implement event elapsed time.
  if (!ms) return cudaErrorInvalidValue;
  *ms = 0.0f;
  return cudaErrorNotSupported;
}

//===----------------------------------------------------------------------===//
// Execution Control
//===----------------------------------------------------------------------===//

cudaError_t CUDAAPI cudaFuncGetAttributes(cudaFuncAttributes* attr,
                                          const void* func) {
  // TODO: Implement get function attributes.
  if (!attr) return cudaErrorInvalidValue;
  memset(attr, 0, sizeof(*attr));
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaFuncSetAttribute(const void* func,
                                         cudaFuncAttribute attribute,
                                         int value) {
  // TODO: Implement set function attribute.
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaFuncSetCacheConfig(const void* func,
                                           cudaFuncCache cacheConfig) {
  // TODO: Implement set function cache config.
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaLaunchKernel(const void* func, dim3 gridDim,
                                     dim3 blockDim, void** args,
                                     size_t sharedMem, cudaStream_t stream) {
  // TODO: Implement launch kernel.
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaLaunchCooperativeKernel(const void* func, dim3 gridDim,
                                                dim3 blockDim, void** args,
                                                size_t sharedMem,
                                                cudaStream_t stream) {
  // TODO: Implement launch cooperative kernel.
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn,
                                       void* userData) {
  // TODO: Implement launch host function.
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaSetDoubleForDevice(double* d) {
  // TODO: Implement set double for device.
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaSetDoubleForHost(double* d) {
  // TODO: Implement set double for host.
  return cudaErrorNotSupported;
}

//===----------------------------------------------------------------------===//
// Occupancy
//===----------------------------------------------------------------------===//

cudaError_t CUDAAPI cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    int* numBlocks, const void* func, int blockSize, size_t dynamicSMemSize) {
  // TODO: Implement occupancy max active blocks.
  if (!numBlocks) return cudaErrorInvalidValue;
  *numBlocks = 0;
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int* numBlocks, const void* func, int blockSize, size_t dynamicSMemSize,
    unsigned int flags) {
  // TODO: Implement occupancy max active blocks with flags.
  if (!numBlocks) return cudaErrorInvalidValue;
  *numBlocks = 0;
  return cudaErrorNotSupported;
}

//===----------------------------------------------------------------------===//
// Peer Device Memory Access
//===----------------------------------------------------------------------===//

cudaError_t CUDAAPI cudaMemcpyPeer(void* dst, int dstDevice, const void* src,
                                   int srcDevice, size_t count) {
  // TODO: Implement peer memory copy.
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaMemcpyPeerAsync(void* dst, int dstDevice,
                                        const void* src, int srcDevice,
                                        size_t count, cudaStream_t stream) {
  // TODO: Implement asynchronous peer memory copy.
  return cudaErrorNotSupported;
}

//===----------------------------------------------------------------------===//
// Unified Addressing
//===----------------------------------------------------------------------===//

cudaError_t CUDAAPI cudaGetDevicePointer(void** pDevice, void* pHost,
                                         unsigned int flags) {
  // TODO: Implement get device pointer.
  if (!pDevice) return cudaErrorInvalidValue;
  *pDevice = NULL;
  return cudaErrorNotSupported;
}

//===----------------------------------------------------------------------===//
// Graph Management
//===----------------------------------------------------------------------===//

cudaError_t CUDAAPI cudaGraphCreate(cudaGraph_t* pGraph, unsigned int flags) {
  // TODO: Implement graph creation.
  if (!pGraph) return cudaErrorInvalidValue;
  *pGraph = NULL;
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaGraphDestroy(cudaGraph_t graph) {
  // TODO: Implement graph destruction.
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaGraphInstantiate(cudaGraphExec_t* pGraphExec,
                                         cudaGraph_t graph,
                                         unsigned long long flags) {
  // TODO: Implement graph instantiation.
  if (!pGraphExec) return cudaErrorInvalidValue;
  *pGraphExec = NULL;
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaGraphInstantiateWithFlags(cudaGraphExec_t* pGraphExec,
                                                  cudaGraph_t graph,
                                                  unsigned long long flags) {
  // TODO: Implement graph instantiation with flags.
  if (!pGraphExec) return cudaErrorInvalidValue;
  *pGraphExec = NULL;
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaGraphInstantiateWithParams(
    cudaGraphExec_t* pGraphExec, cudaGraph_t graph,
    cudaGraphInstantiateParams* instantiateParams) {
  // TODO: Implement graph instantiation with params.
  if (!pGraphExec) return cudaErrorInvalidValue;
  *pGraphExec = NULL;
  if (instantiateParams) {
    instantiateParams->result_out = cudaGraphInstantiateError;
    instantiateParams->errNode_out = NULL;
  }
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaGraphExecDestroy(cudaGraphExec_t graphExec) {
  // TODO: Implement graph exec destruction.
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaGraphLaunch(cudaGraphExec_t graphExec,
                                    cudaStream_t stream) {
  // TODO: Implement graph launch.
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaGraphUpload(cudaGraphExec_t graphExec,
                                    cudaStream_t stream) {
  // TODO: Implement graph upload.
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaGraphAddKernelNode(
    cudaGraphNode_t* pGraphNode, cudaGraph_t graph,
    const cudaGraphNode_t* pDependencies, size_t numDependencies,
    const cudaKernelNodeParams* pNodeParams) {
  // TODO: Implement add kernel node.
  if (!pGraphNode) return cudaErrorInvalidValue;
  *pGraphNode = NULL;
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaGraphAddMemcpyNode(
    cudaGraphNode_t* pGraphNode, cudaGraph_t graph,
    const cudaGraphNode_t* pDependencies, size_t numDependencies,
    const cudaMemcpy3DParms* pCopyParams) {
  // TODO: Implement add memcpy node.
  if (!pGraphNode) return cudaErrorInvalidValue;
  *pGraphNode = NULL;
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaGraphAddMemsetNode(
    cudaGraphNode_t* pGraphNode, cudaGraph_t graph,
    const cudaGraphNode_t* pDependencies, size_t numDependencies,
    const cudaMemsetParams* pMemsetParams) {
  // TODO: Implement add memset node.
  if (!pGraphNode) return cudaErrorInvalidValue;
  *pGraphNode = NULL;
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaGraphAddHostNode(
    cudaGraphNode_t* pGraphNode, cudaGraph_t graph,
    const cudaGraphNode_t* pDependencies, size_t numDependencies,
    const cudaHostNodeParams* pNodeParams) {
  // TODO: Implement add host node.
  if (!pGraphNode) return cudaErrorInvalidValue;
  *pGraphNode = NULL;
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI
cudaGraphAddChildGraphNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph,
                           const cudaGraphNode_t* pDependencies,
                           size_t numDependencies, cudaGraph_t childGraph) {
  // TODO: Implement add child graph node.
  if (!pGraphNode) return cudaErrorInvalidValue;
  *pGraphNode = NULL;
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaGraphAddEmptyNode(cudaGraphNode_t* pGraphNode,
                                          cudaGraph_t graph,
                                          const cudaGraphNode_t* pDependencies,
                                          size_t numDependencies) {
  // TODO: Implement add empty node.
  if (!pGraphNode) return cudaErrorInvalidValue;
  *pGraphNode = NULL;
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaGraphClone(cudaGraph_t* pGraphClone,
                                   cudaGraph_t originalGraph) {
  // TODO: Implement graph clone.
  if (!pGraphClone) return cudaErrorInvalidValue;
  *pGraphClone = NULL;
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaGraphNodeGetType(cudaGraphNode_t node,
                                         cudaGraphNodeType* pType) {
  // TODO: Implement get node type.
  if (!pType) return cudaErrorInvalidValue;
  *pType = cudaGraphNodeTypeEmpty;
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t* nodes,
                                      size_t* numNodes) {
  // TODO: Implement get graph nodes.
  if (!numNodes) return cudaErrorInvalidValue;
  if (nodes) {
    // Caller provided buffer, fill it.
  } else {
    // Query mode, return count.
    *numNodes = 0;
  }
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaGraphGetRootNodes(cudaGraph_t graph,
                                          cudaGraphNode_t* pRootNodes,
                                          size_t* pNumRootNodes) {
  // TODO: Implement get root nodes.
  if (!pNumRootNodes) return cudaErrorInvalidValue;
  if (pRootNodes) {
    // Caller provided buffer, fill it.
  } else {
    // Query mode, return count.
    *pNumRootNodes = 0;
  }
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaGraphNodeGetDependencies(cudaGraphNode_t node,
                                                 cudaGraphNode_t* pDependencies,
                                                 size_t* pNumDependencies) {
  // TODO: Implement get node dependencies.
  if (!pNumDependencies) return cudaErrorInvalidValue;
  if (pDependencies) {
    // Caller provided buffer, fill it.
  } else {
    // Query mode, return count.
    *pNumDependencies = 0;
  }
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaGraphNodeGetDependentNodes(
    cudaGraphNode_t node, cudaGraphNode_t* pDependentNodes,
    size_t* pNumDependentNodes) {
  // TODO: Implement get dependent nodes.
  if (!pNumDependentNodes) return cudaErrorInvalidValue;
  if (pDependentNodes) {
    // Caller provided buffer, fill it.
  } else {
    // Query mode, return count.
    *pNumDependentNodes = 0;
  }
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaGraphAddDependencies(cudaGraph_t graph,
                                             const cudaGraphNode_t* from,
                                             const cudaGraphNode_t* to,
                                             size_t numDependencies) {
  // TODO: Implement add dependencies.
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaGraphRemoveDependencies(cudaGraph_t graph,
                                                const cudaGraphNode_t* from,
                                                const cudaGraphNode_t* to,
                                                size_t numDependencies) {
  // TODO: Implement remove dependencies.
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaGraphDestroyNode(cudaGraphNode_t node) {
  // TODO: Implement destroy node.
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaGraphExecUpdate(cudaGraphExec_t hGraphExec,
                                        cudaGraph_t hGraph,
                                        cudaGraphExecUpdateResultInfo* resultInfo) {
  // TODO: Implement graph exec update.
  if (resultInfo) {
    resultInfo->result = cudaGraphExecUpdateError;
    resultInfo->errorNode = NULL;
    resultInfo->errorFromNode = NULL;
  }
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaGraphExecKernelNodeSetParams(
    cudaGraphExec_t hGraphExec, cudaGraphNode_t node,
    const cudaKernelNodeParams* pNodeParams) {
  // TODO: Implement set kernel node params.
  return cudaErrorNotSupported;
}

//===----------------------------------------------------------------------===//
// Memory Pools
//===----------------------------------------------------------------------===//

cudaError_t CUDAAPI cudaMemPoolCreate(cudaMemPool_t* pool,
                                      const cudaMemPoolProps* poolProps) {
  // TODO: Implement memory pool creation.
  if (!pool) return cudaErrorInvalidValue;
  *pool = NULL;
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaMemPoolDestroy(cudaMemPool_t pool) {
  // TODO: Implement memory pool destruction.
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaMemPoolSetAttribute(cudaMemPool_t pool,
                                            cudaMemPoolAttr attr, void* value) {
  // TODO: Implement set memory pool attribute.
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaMemPoolGetAttribute(cudaMemPool_t pool,
                                            cudaMemPoolAttr attr, void* value) {
  // TODO: Implement get memory pool attribute.
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaMallocAsync(void** ptr, size_t size,
                                    cudaStream_t hStream) {
  // TODO: Implement asynchronous memory allocation.
  if (!ptr) return cudaErrorInvalidValue;
  *ptr = NULL;
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaFreeAsync(void* ptr, cudaStream_t hStream) {
  // TODO: Implement asynchronous memory free.
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaMemPoolTrimTo(cudaMemPool_t pool,
                                      size_t minBytesToKeep) {
  // TODO: Implement memory pool trim.
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaMemPoolSetAccess(cudaMemPool_t pool,
                                         const cudaMemAccessDesc* map,
                                         size_t count) {
  // TODO: Implement set memory pool access.
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaMemPoolGetAccess(cudaMemAccessFlags* flags,
                                         cudaMemPool_t pool,
                                         cudaMemLocation* location) {
  // TODO: Implement get memory pool access.
  if (!flags) return cudaErrorInvalidValue;
  *flags = cudaMemAccessFlagsProtNone;
  return cudaErrorNotSupported;
}

//===----------------------------------------------------------------------===//
// IPC
//===----------------------------------------------------------------------===//

cudaError_t CUDAAPI cudaIpcGetEventHandle(cudaIpcEventHandle_t* handle,
                                          cudaEvent_t event) {
  // TODO: Implement IPC get event handle.
  if (!handle) return cudaErrorInvalidValue;
  memset(handle, 0, sizeof(*handle));
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaIpcOpenEventHandle(cudaEvent_t* event,
                                           cudaIpcEventHandle_t handle) {
  // TODO: Implement IPC open event handle.
  if (!event) return cudaErrorInvalidValue;
  *event = NULL;
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaIpcGetMemHandle(cudaIpcMemHandle_t* handle,
                                        void* devPtr) {
  // TODO: Implement IPC get memory handle.
  if (!handle) return cudaErrorInvalidValue;
  memset(handle, 0, sizeof(*handle));
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaIpcOpenMemHandle(void** devPtr,
                                         cudaIpcMemHandle_t handle,
                                         unsigned int flags) {
  // TODO: Implement IPC open memory handle.
  if (!devPtr) return cudaErrorInvalidValue;
  *devPtr = NULL;
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaIpcCloseMemHandle(void* devPtr) {
  // TODO: Implement IPC close memory handle.
  return cudaErrorNotSupported;
}

//===----------------------------------------------------------------------===//
// Profiler
//===----------------------------------------------------------------------===//

cudaError_t CUDAAPI cudaProfilerStart(void) {
  // TODO: Implement profiler start.
  return cudaErrorNotSupported;
}

cudaError_t CUDAAPI cudaProfilerStop(void) {
  // TODO: Implement profiler stop.
  return cudaErrorNotSupported;
}

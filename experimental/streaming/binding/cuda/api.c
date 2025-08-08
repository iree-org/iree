// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/streaming/binding/cuda/api.h"

#include "experimental/streaming/internal.h"

//===----------------------------------------------------------------------===//
// Flag translation functions
//===----------------------------------------------------------------------===//

static iree_hal_streaming_stream_flags_t cuda_stream_flags_to_internal(
    unsigned int cuda_flags) {
  iree_hal_streaming_stream_flags_t flags = IREE_HAL_STREAMING_STREAM_FLAG_NONE;
  if (cuda_flags & CU_STREAM_NON_BLOCKING) {
    flags |= IREE_HAL_STREAMING_STREAM_FLAG_NON_BLOCKING;
  }
  return flags;
}

static iree_hal_streaming_event_flags_t cuda_event_flags_to_internal(
    unsigned int cuda_flags) {
  iree_hal_streaming_event_flags_t flags = IREE_HAL_STREAMING_EVENT_FLAG_NONE;
  if (cuda_flags & CU_EVENT_BLOCKING_SYNC) {
    flags |= IREE_HAL_STREAMING_EVENT_FLAG_BLOCKING_SYNC;
  }
  if (cuda_flags & CU_EVENT_DISABLE_TIMING) {
    flags |= IREE_HAL_STREAMING_EVENT_FLAG_DISABLE_TIMING;
  }
  if (cuda_flags & CU_EVENT_INTERPROCESS) {
    flags |= IREE_HAL_STREAMING_EVENT_FLAG_INTERPROCESS;
  }
  return flags;
}

static iree_hal_streaming_memory_flags_t cuda_memory_flags_to_internal(
    unsigned int cuda_flags) {
  iree_hal_streaming_memory_flags_t flags = IREE_HAL_STREAMING_MEMORY_FLAG_NONE;
  if (cuda_flags & CU_MEMHOSTALLOC_PORTABLE) {
    flags |= IREE_HAL_STREAMING_MEMORY_FLAG_PORTABLE;
  }
  if (cuda_flags & CU_MEMHOSTALLOC_DEVICEMAP) {
    flags |= IREE_HAL_STREAMING_MEMORY_FLAG_PINNED;
  }
  if (cuda_flags & CU_MEMHOSTALLOC_WRITECOMBINED) {
    flags |= IREE_HAL_STREAMING_MEMORY_FLAG_WRITE_COMBINED;
  }
  return flags;
}

static iree_hal_streaming_graph_instantiate_flags_t
cuda_graph_instantiate_flags_to_internal(unsigned long long cuda_flags) {
  iree_hal_streaming_graph_instantiate_flags_t flags =
      IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_NONE;
  if (cuda_flags & CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH) {
    flags |= IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;
  }
  if (cuda_flags & CUDA_GRAPH_INSTANTIATE_FLAG_UPLOAD) {
    flags |= IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_UPLOAD;
  }
  if (cuda_flags & CUDA_GRAPH_INSTANTIATE_FLAG_DEVICE_LAUNCH) {
    flags |= IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_DEVICE_LAUNCH;
  }
  if (cuda_flags & CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY) {
    flags |= IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY;
  }
  return flags;
}

static iree_hal_streaming_mem_pool_attr_t cuda_mempool_attr_to_internal(
    CUmemPool_attribute attr) {
  switch (attr) {
    case CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES:
      return IREE_HAL_STREAMING_MEM_POOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES;
    case CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC:
      return IREE_HAL_STREAMING_MEM_POOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC;
    case CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES:
      return IREE_HAL_STREAMING_MEM_POOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES;
    case CU_MEMPOOL_ATTR_RELEASE_THRESHOLD:
      return IREE_HAL_STREAMING_MEM_POOL_ATTR_RELEASE_THRESHOLD;
    case CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT:
      return IREE_HAL_STREAMING_MEM_POOL_ATTR_RESERVED_MEM_CURRENT;
    case CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH:
      return IREE_HAL_STREAMING_MEM_POOL_ATTR_RESERVED_MEM_HIGH;
    case CU_MEMPOOL_ATTR_USED_MEM_CURRENT:
      return IREE_HAL_STREAMING_MEM_POOL_ATTR_USED_MEM_CURRENT;
    case CU_MEMPOOL_ATTR_USED_MEM_HIGH:
      return IREE_HAL_STREAMING_MEM_POOL_ATTR_USED_MEM_HIGH;
    default:
      return IREE_HAL_STREAMING_MEM_POOL_ATTR_RESERVED_MEM_CURRENT;
  }
}

static iree_hal_streaming_mem_handle_type_t cuda_mem_handle_type_to_internal(
    CUmemAllocationHandleType handle_type) {
  switch (handle_type) {
    case CU_MEM_HANDLE_TYPE_NONE:
      return IREE_HAL_STREAMING_MEM_HANDLE_TYPE_NONE;
    case CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR:
      return IREE_HAL_STREAMING_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    case CU_MEM_HANDLE_TYPE_WIN32:
      return IREE_HAL_STREAMING_MEM_HANDLE_TYPE_WIN32;
    case CU_MEM_HANDLE_TYPE_WIN32_KMT:
      return IREE_HAL_STREAMING_MEM_HANDLE_TYPE_WIN32_KMT;
    default:
      return IREE_HAL_STREAMING_MEM_HANDLE_TYPE_NONE;
  }
}

static iree_hal_streaming_mem_location_type_t
cuda_mem_location_type_to_internal(CUmemLocationType type) {
  switch (type) {
    case CU_MEM_LOCATION_TYPE_INVALID:
      return IREE_HAL_STREAMING_MEM_LOCATION_TYPE_INVALID;
    case CU_MEM_LOCATION_TYPE_DEVICE:
      return IREE_HAL_STREAMING_MEM_LOCATION_TYPE_DEVICE;
    case CU_MEM_LOCATION_TYPE_HOST:
      return IREE_HAL_STREAMING_MEM_LOCATION_TYPE_HOST;
    case CU_MEM_LOCATION_TYPE_HOST_NUMA:
      return IREE_HAL_STREAMING_MEM_LOCATION_TYPE_HOST_NUMA;
    case CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT:
      return IREE_HAL_STREAMING_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT;
    default:
      return IREE_HAL_STREAMING_MEM_LOCATION_TYPE_INVALID;
  }
}

static iree_hal_streaming_mem_access_flags_t cuda_mem_access_flags_to_internal(
    CUmemAccess_flags flags) {
  switch (flags) {
    case CU_MEM_ACCESS_FLAGS_PROT_NONE:
      return IREE_HAL_STREAMING_MEM_ACCESS_FLAG_PROT_NONE;
    case CU_MEM_ACCESS_FLAGS_PROT_READ:
      return IREE_HAL_STREAMING_MEM_ACCESS_FLAG_PROT_READ;
    case CU_MEM_ACCESS_FLAGS_PROT_READWRITE:
      return IREE_HAL_STREAMING_MEM_ACCESS_FLAG_PROT_READWRITE;
    default:
      return IREE_HAL_STREAMING_MEM_ACCESS_FLAG_PROT_NONE;
  }
}

static iree_hal_streaming_context_limit_t cuda_limit_to_internal(
    CUlimit limit) {
  switch (limit) {
    case CU_LIMIT_STACK_SIZE:
      return IREE_HAL_STREAMING_CONTEXT_LIMIT_STACK_SIZE;
    case CU_LIMIT_PRINTF_FIFO_SIZE:
      return IREE_HAL_STREAMING_CONTEXT_LIMIT_PRINTF_FIFO_SIZE;
    case CU_LIMIT_MALLOC_HEAP_SIZE:
      return IREE_HAL_STREAMING_CONTEXT_LIMIT_MALLOC_HEAP_SIZE;
    case CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH:
      return IREE_HAL_STREAMING_CONTEXT_LIMIT_DEV_RUNTIME_SYNC_DEPTH;
    case CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT:
      return IREE_HAL_STREAMING_CONTEXT_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT;
    case CU_LIMIT_MAX_L2_FETCH_GRANULARITY:
      return IREE_HAL_STREAMING_CONTEXT_LIMIT_MAX_L2_FETCH_GRANULARITY;
    case CU_LIMIT_PERSISTING_L2_CACHE_SIZE:
      return IREE_HAL_STREAMING_CONTEXT_LIMIT_PERSISTING_L2_CACHE_SIZE;
    default:
      // Return an invalid value that will trigger error in internal API.
      return (iree_hal_streaming_context_limit_t)-1;
  }
}

//===----------------------------------------------------------------------===//
// Status conversion
//===----------------------------------------------------------------------===//

static CUresult iree_status_to_cu_result(iree_status_t status) {
  if (iree_status_is_ok(status)) {
    return CUDA_SUCCESS;
  }

  // Map IREE status codes to CUDA error codes.
  iree_status_code_t code = iree_status_code(status);
  iree_status_free(status);

  switch (code) {
    case IREE_STATUS_INVALID_ARGUMENT:
      return CUDA_ERROR_INVALID_VALUE;
    case IREE_STATUS_OUT_OF_RANGE:
      return CUDA_ERROR_INVALID_VALUE;
    case IREE_STATUS_RESOURCE_EXHAUSTED:
      return CUDA_ERROR_OUT_OF_MEMORY;
    case IREE_STATUS_NOT_FOUND:
      return CUDA_ERROR_NOT_FOUND;
    case IREE_STATUS_PERMISSION_DENIED:
      return CUDA_ERROR_INVALID_CONTEXT;
    case IREE_STATUS_UNIMPLEMENTED:
      return CUDA_ERROR_NOT_SUPPORTED;
    case IREE_STATUS_UNAVAILABLE:
      return CUDA_ERROR_NOT_READY;
    case IREE_STATUS_FAILED_PRECONDITION:
      return CUDA_ERROR_NOT_INITIALIZED;
    default:
      return CUDA_ERROR_UNKNOWN;
  }
}

//===----------------------------------------------------------------------===//
// Initialization
//===----------------------------------------------------------------------===//

CUDAAPI CUresult cuInit(unsigned int Flags) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = iree_hal_streaming_init_global(
      IREE_HAL_STREAMING_INIT_FLAG_NONE, iree_allocator_system());
  CUresult result = iree_status_to_cu_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuHALDeinit(void) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_streaming_cleanup_global();
  IREE_TRACE_ZONE_END(z0);
  return CUDA_SUCCESS;
}

CUDAAPI CUresult cuDriverGetVersion(int* driverVersion) {
  if (!driverVersion) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  *driverVersion = 12020;  // CUDA 12.2
  return CUDA_SUCCESS;
}

CUDAAPI CUresult cuRuntimeGetVersion(int* runtimeVersion) {
  if (!runtimeVersion) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  *runtimeVersion = 12020;  // CUDA 12.2
  return CUDA_SUCCESS;
}

CUDAAPI CUresult cuGetDevice(int* device) {
  if (!device) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  iree_hal_streaming_context_t* context = iree_hal_streaming_context_current();
  if (!context) {
    return CUDA_ERROR_NOT_INITIALIZED;
  }

  *device = context->device_ordinal;
  return CUDA_SUCCESS;
}

CUDAAPI CUresult cuSetDevice(int device) {
  iree_hal_streaming_device_t* device_obj =
      iree_hal_streaming_device_entry(device);
  if (!device_obj) {
    return CUDA_ERROR_INVALID_DEVICE;
  }

  // Get or create the primary context lazily.
  iree_hal_streaming_context_t* primary_context = NULL;
  iree_status_t status =
      iree_hal_streaming_device_get_or_create_primary_context(device_obj,
                                                              &primary_context);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return CUDA_ERROR_OUT_OF_MEMORY;
  }

  // Set the primary context as current.
  status = iree_hal_streaming_context_set_current(primary_context);
  CUresult result = iree_status_to_cu_result(status);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
  }
  return result;
}

//===----------------------------------------------------------------------===//
// Context management
//===----------------------------------------------------------------------===//

// Helper function to convert CUDA context flags to internal flags.
static iree_hal_streaming_context_flags_t
iree_hal_streaming_cuda_context_flags_to_internal(unsigned int cuda_flags) {
  iree_hal_streaming_context_flags_t flags = {0};

  // Convert scheduling flags.
  int sched_flags = cuda_flags & CU_CTX_SCHED_MASK;
  switch (sched_flags) {
    case CU_CTX_SCHED_SPIN:
      flags.scheduling_mode = IREE_HAL_STREAMING_SCHEDULING_MODE_SPIN;
      break;
    case CU_CTX_SCHED_YIELD:
      flags.scheduling_mode = IREE_HAL_STREAMING_SCHEDULING_MODE_YIELD;
      break;
    case CU_CTX_SCHED_BLOCKING_SYNC:
      flags.scheduling_mode = IREE_HAL_STREAMING_SCHEDULING_MODE_BLOCKING_SYNC;
      break;
    default:
      flags.scheduling_mode = IREE_HAL_STREAMING_SCHEDULING_MODE_AUTO;
      break;
  }

  // Convert other flags.
  if (cuda_flags & CU_CTX_MAP_HOST) {
    flags.map_host_memory = true;
  }
  if (cuda_flags & CU_CTX_LMEM_RESIZE_TO_MAX) {
    flags.resize_local_mem_to_max = true;
  }

  return flags;
}

CUDAAPI CUresult cuCtxCreate(CUcontext* pctx, unsigned int flags,
                             CUdevice dev) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!pctx) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  iree_hal_streaming_device_t* device = iree_hal_streaming_device_entry(dev);
  if (!device) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_DEVICE;
  }

  // Create a new context for the device.
  // Get the host allocator from the device registry.
  iree_hal_streaming_device_registry_t* device_registry =
      iree_hal_streaming_device_registry();
  if (!device_registry) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_NOT_INITIALIZED;
  }

  iree_hal_streaming_context_t* context = NULL;
  iree_status_t status = iree_hal_streaming_context_create(
      device, iree_hal_streaming_cuda_context_flags_to_internal(flags),
      device_registry->host_allocator, &context);

  if (iree_status_is_ok(status)) {
    *pctx = (CUcontext)context;
    // Make it current.
    status = iree_hal_streaming_context_set_current(context);
  }

  CUresult result = iree_status_to_cu_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuCtxDestroy(CUcontext ctx) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!ctx) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  // Check if this is the current context.
  // If so, clear it from TLS to avoid a dangling reference.
  if (ctx && ctx == (CUcontext)iree_hal_streaming_context_current()) {
    // This will release the TLS reference.
    iree_hal_streaming_context_set_current(NULL);
  }

  // Release the context.
  if (ctx) {
    iree_hal_streaming_context_release((iree_hal_streaming_context_t*)ctx);
  }

  IREE_TRACE_ZONE_END(z0);
  return CUDA_SUCCESS;
}

CUDAAPI CUresult cuCtxPushCurrent(CUcontext ctx) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status =
      iree_hal_streaming_context_push((iree_hal_streaming_context_t*)ctx);
  CUresult result = iree_status_to_cu_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuCtxPopCurrent(CUcontext* pctx) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_streaming_context_t* context = NULL;
  iree_status_t status = iree_hal_streaming_context_pop(&context);
  if (iree_status_is_ok(status) && pctx) {
    *pctx = (CUcontext)context;
  }
  CUresult result = iree_status_to_cu_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuCtxSetCurrent(CUcontext ctx) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = iree_hal_streaming_context_set_current(
      (iree_hal_streaming_context_t*)ctx);
  CUresult result = iree_status_to_cu_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuCtxGetCurrent(CUcontext* pctx) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!pctx) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  *pctx = (CUcontext)iree_hal_streaming_context_current();
  IREE_TRACE_ZONE_END(z0);
  return CUDA_SUCCESS;
}

CUDAAPI CUresult cuCtxGetDevice(CUdevice* device) {
  if (!device) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  iree_hal_streaming_context_t* context = iree_hal_streaming_context_current();
  if (!context) {
    return CUDA_ERROR_INVALID_CONTEXT;
  }

  *device = (CUdevice)context->device_ordinal;
  return CUDA_SUCCESS;
}

CUDAAPI CUresult cuCtxGetFlags(unsigned int* flags) {
  if (!flags) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  iree_hal_streaming_context_t* context = iree_hal_streaming_context_current();
  if (!context) {
    return CUDA_ERROR_INVALID_CONTEXT;
  }

  // Convert internal flags structure to CUDA flags.
  *flags = 0;

  // Convert scheduling mode.
  switch (context->flags.scheduling_mode) {
    case IREE_HAL_STREAMING_SCHEDULING_MODE_SPIN:
      *flags |= CU_CTX_SCHED_SPIN;
      break;
    case IREE_HAL_STREAMING_SCHEDULING_MODE_YIELD:
      *flags |= CU_CTX_SCHED_YIELD;
      break;
    case IREE_HAL_STREAMING_SCHEDULING_MODE_BLOCKING_SYNC:
      *flags |= CU_CTX_SCHED_BLOCKING_SYNC;
      break;
    case IREE_HAL_STREAMING_SCHEDULING_MODE_AUTO:
    default:
      // CU_CTX_SCHED_AUTO is 0, so nothing to add.
      break;
  }

  // Convert other flags.
  if (context->flags.map_host_memory) {
    *flags |= CU_CTX_MAP_HOST;
  }
  if (context->flags.resize_local_mem_to_max) {
    *flags |= CU_CTX_LMEM_RESIZE_TO_MAX;
  }

  return CUDA_SUCCESS;
}

CUDAAPI CUresult cuCtxSynchronize(void) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_streaming_context_t* context = iree_hal_streaming_context_current();
  if (!context) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_CONTEXT;
  }

  iree_status_t status = iree_hal_streaming_context_synchronize(context);
  CUresult result = iree_status_to_cu_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuCtxGetLimit(size_t* pvalue, CUlimit limit) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!pvalue) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  // Get current context.
  iree_hal_streaming_context_t* context = iree_hal_streaming_context_current();
  if (!context) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_CONTEXT;
  }

  // Get the limit value using internal API.
  iree_status_t status = iree_hal_streaming_context_limit(
      context, cuda_limit_to_internal(limit), pvalue);

  CUresult result = iree_status_to_cu_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuCtxSetLimit(CUlimit limit, size_t value) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Get current context.
  iree_hal_streaming_context_t* context = iree_hal_streaming_context_current();
  if (!context) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_CONTEXT;
  }

  // Set the limit value using internal API.
  iree_status_t status = iree_hal_streaming_context_set_limit(
      context, cuda_limit_to_internal(limit), value);

  CUresult result = iree_status_to_cu_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuCtxGetApiVersion(CUcontext ctx, unsigned int* version) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!version) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  *version = 12020;  // CUDA 12.2
  IREE_TRACE_ZONE_END(z0);
  return CUDA_SUCCESS;
}

CUDAAPI CUresult cuCtxGetStreamPriorityRange(int* leastPriority,
                                             int* greatestPriority) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (leastPriority) *leastPriority = -1;
  if (greatestPriority) *greatestPriority = 0;
  IREE_TRACE_ZONE_END(z0);
  return CUDA_SUCCESS;
}

CUDAAPI CUresult cuCtxEnablePeerAccess(CUcontext peerContext,
                                       unsigned int Flags) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!peerContext) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  iree_hal_streaming_context_t* context = iree_hal_streaming_context_current();
  if (!context) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_CONTEXT;
  }

  iree_status_t status = iree_hal_streaming_context_enable_peer_access(
      context, (iree_hal_streaming_context_t*)peerContext);

  CUresult result = iree_status_to_cu_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuCtxDisablePeerAccess(CUcontext peerContext) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!peerContext) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  iree_hal_streaming_context_t* context = iree_hal_streaming_context_current();
  if (!context) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_CONTEXT;
  }

  iree_status_t status = iree_hal_streaming_context_disable_peer_access(
      context, (iree_hal_streaming_context_t*)peerContext);

  CUresult result = iree_status_to_cu_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

//===----------------------------------------------------------------------===//
// Device management
//===----------------------------------------------------------------------===//

CUDAAPI CUresult cuDeviceGet(CUdevice* device, int ordinal) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!device) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  iree_hal_streaming_device_registry_t* device_registry =
      iree_hal_streaming_device_registry();
  if (!device_registry || ordinal < 0 ||
      ordinal >= (int)device_registry->device_count) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_DEVICE;
  }

  *device = ordinal;
  IREE_TRACE_ZONE_END(z0);
  return CUDA_SUCCESS;
}

CUDAAPI CUresult cuDeviceGetCount(int* count) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!count) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  iree_hal_streaming_device_registry_t* device_registry =
      iree_hal_streaming_device_registry();
  if (!device_registry) {
    *count = 0;
    IREE_TRACE_ZONE_END(z0);
    return CUDA_SUCCESS;
  }

  *count = (int)device_registry->device_count;
  IREE_TRACE_ZONE_END(z0);
  return CUDA_SUCCESS;
}

CUDAAPI CUresult cuDeviceGetName(char* name, int len, CUdevice dev) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!name || len <= 0) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  iree_status_t status = iree_hal_streaming_device_name(
      (iree_hal_streaming_device_ordinal_t)dev, name, (size_t)len);
  CUresult result = iree_status_to_cu_result(status);

  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuDeviceGetUuid(CUuuid* uuid, CUdevice dev) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!uuid) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  iree_hal_streaming_device_t* device = iree_hal_streaming_device_entry(dev);
  if (!device) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_DEVICE;
  }

  // Generate a UUID from device info.
  memset(uuid, 0, sizeof(CUuuid));
  memcpy(uuid->bytes, &device->info.device_id, sizeof(device->info.device_id));

  IREE_TRACE_ZONE_END(z0);
  return CUDA_SUCCESS;
}

CUDAAPI CUresult cuDeviceTotalMem(size_t* bytes, CUdevice dev) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!bytes) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  iree_device_size_t free_memory = 0;
  iree_device_size_t total_memory = 0;
  iree_status_t status =
      iree_hal_streaming_device_memory_info(dev, &free_memory, &total_memory);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_DEVICE;
  }

  *bytes = total_memory;
  IREE_TRACE_ZONE_END(z0);
  return CUDA_SUCCESS;
}

CUDAAPI CUresult cuDeviceGetAttribute(int* pi, CUdevice_attribute attrib,
                                      CUdevice dev) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!pi) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  iree_hal_streaming_device_t* device = iree_hal_streaming_device_entry(dev);
  if (!device) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_DEVICE;
  }

  // Map attributes to device properties.
  switch (attrib) {
    case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK:
      *pi = device->max_threads_per_block;
      break;
    case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X:
      *pi = device->max_block_dim[0];
      break;
    case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y:
      *pi = device->max_block_dim[1];
      break;
    case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z:
      *pi = device->max_block_dim[2];
      break;
    case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X:
      *pi = device->max_grid_dim[0];
      break;
    case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y:
      *pi = device->max_grid_dim[1];
      break;
    case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z:
      *pi = device->max_grid_dim[2];
      break;
    case CU_DEVICE_ATTRIBUTE_WARP_SIZE:
      *pi = device->warp_size;
      break;
    case CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT:
      *pi = device->multiprocessor_count;
      break;
    case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR:
      *pi = device->compute_capability_major;
      break;
    case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR:
      *pi = device->compute_capability_minor;
      break;
    case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN:
      // Return the maximum shared memory per block when opted in.
      // For now, return the same as MAX_SHARED_MEMORY_PER_BLOCK.
      // This would typically be higher on devices that support > 48KB.
      *pi = 49152;  // 48KB default, actual value would be device-specific.
      break;
    case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR:
      // Total shared memory per SM/multiprocessor.
      // Common values: 64KB, 96KB, 128KB depending on architecture.
      *pi = 65536;  // 64KB default.
      break;
    case CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED:
      // GPU Direct RDMA with CUDA VMM support.
      // Currently not supported in stream HAL.
      *pi = 0;
      break;
    default:
      // Return sensible defaults for other attributes.
      *pi = 0;
      break;
  }

  IREE_TRACE_ZONE_END(z0);
  return CUDA_SUCCESS;
}

CUDAAPI CUresult cuDeviceCanAccessPeer(int* canAccessPeer, CUdevice dev,
                                       CUdevice peerDev) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!canAccessPeer) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  // Use the P2P query function.
  bool can_access = false;
  iree_status_t status = iree_hal_streaming_device_can_access_peer(
      (int)dev, (int)peerDev, &can_access);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    *canAccessPeer = 0;
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_DEVICE;
  }

  *canAccessPeer = can_access ? 1 : 0;
  IREE_TRACE_ZONE_END(z0);
  return CUDA_SUCCESS;
}

CUDAAPI CUresult cuDeviceGetP2PAttribute(int* value,
                                         CUdevice_P2PAttribute attrib,
                                         CUdevice srcDevice,
                                         CUdevice dstDevice) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!value) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  // Look up P2P link.
  iree_hal_streaming_p2p_link_t* link =
      iree_hal_streaming_device_lookup_p2p_link((int)srcDevice, (int)dstDevice);
  if (!link) {
    *value = 0;
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_DEVICE;
  }

  // Map CUDA P2P attribute enum to the appropriate link field.
  switch (attrib) {
    case CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED:
      *value = link->access_supported ? 1 : 0;
      break;
    case CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED:
      *value = link->native_atomic_supported ? 1 : 0;
      break;
    case CU_DEVICE_P2P_ATTRIBUTE_CUDA_ARRAY_ACCESS_SUPPORTED:
      *value = link->cuda_array_access_supported ? 1 : 0;
      break;
    default:
      // Unsupported attribute.
      *value = 0;
      break;
  }

  IREE_TRACE_ZONE_END(z0);
  return CUDA_SUCCESS;
}

CUDAAPI CUresult cuDevicePrimaryCtxRetain(CUcontext* pctx, CUdevice dev) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!pctx) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  iree_hal_streaming_device_t* device = iree_hal_streaming_device_entry(dev);
  if (!device) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_DEVICE;
  }

  // Retain the primary context, creating it if necessary.
  iree_hal_streaming_context_t* primary_context = NULL;
  iree_status_t status = iree_hal_streaming_device_retain_primary_context(
      device, &primary_context);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_OUT_OF_MEMORY;
  }

  *pctx = (CUcontext)primary_context;

  IREE_TRACE_ZONE_END(z0);
  return CUDA_SUCCESS;
}

CUDAAPI CUresult cuDevicePrimaryCtxRelease(CUdevice dev) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_streaming_device_t* device = iree_hal_streaming_device_entry(dev);
  if (!device) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_DEVICE;
  }

  // Release the primary context (destroys when ref count reaches 0).
  iree_status_t status =
      iree_hal_streaming_device_release_primary_context(device);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_CONTEXT;
  }

  IREE_TRACE_ZONE_END(z0);
  return CUDA_SUCCESS;
}

CUDAAPI CUresult cuDevicePrimaryCtxSetFlags(CUdevice dev, unsigned int flags) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Validate device.
  iree_hal_streaming_device_registry_t* device_registry =
      iree_hal_streaming_device_registry();
  if (dev < 0 || !device_registry ||
      dev >= (int)device_registry->device_count) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_DEVICE;
  }

  // Convert CUDA flags to internal strongly-typed structure.
  iree_hal_streaming_context_flags_t context_flags = {0};

  // Extract scheduling mode from flags.
  unsigned int sched_flags = flags & (CU_CTX_SCHED_SPIN | CU_CTX_SCHED_YIELD |
                                      CU_CTX_SCHED_BLOCKING_SYNC);
  if (sched_flags == CU_CTX_SCHED_SPIN) {
    context_flags.scheduling_mode = IREE_HAL_STREAMING_SCHEDULING_MODE_SPIN;
  } else if (sched_flags == CU_CTX_SCHED_YIELD) {
    context_flags.scheduling_mode = IREE_HAL_STREAMING_SCHEDULING_MODE_YIELD;
  } else if (sched_flags == CU_CTX_SCHED_BLOCKING_SYNC) {
    context_flags.scheduling_mode =
        IREE_HAL_STREAMING_SCHEDULING_MODE_BLOCKING_SYNC;
  } else {
    context_flags.scheduling_mode = IREE_HAL_STREAMING_SCHEDULING_MODE_AUTO;
  }

  // Extract other flags.
  context_flags.map_host_memory = (flags & CU_CTX_MAP_HOST) != 0;
  context_flags.resize_local_mem_to_max =
      (flags & CU_CTX_LMEM_RESIZE_TO_MAX) != 0;

  // Set the flags on the device.
  iree_status_t status =
      iree_hal_streaming_device_set_primary_context_flags(dev, &context_flags);

  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  IREE_TRACE_ZONE_END(z0);
  return CUDA_SUCCESS;
}

CUDAAPI CUresult cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int* flags,
                                            int* active) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Validate device.
  iree_hal_streaming_device_registry_t* device_registry =
      iree_hal_streaming_device_registry();
  if (dev < 0 || !device_registry ||
      dev >= (int)device_registry->device_count) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_DEVICE;
  }

  // Get the primary context state.
  iree_hal_streaming_context_flags_t context_flags;
  bool is_active;
  iree_status_t status = iree_hal_streaming_device_primary_context_state(
      dev, &context_flags, &is_active);

  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_DEVICE;
  }

  // Convert internal flags back to CUDA flags.
  if (flags) {
    *flags = 0;

    // Convert scheduling mode.
    switch (context_flags.scheduling_mode) {
      case IREE_HAL_STREAMING_SCHEDULING_MODE_SPIN:
        *flags |= CU_CTX_SCHED_SPIN;
        break;
      case IREE_HAL_STREAMING_SCHEDULING_MODE_YIELD:
        *flags |= CU_CTX_SCHED_YIELD;
        break;
      case IREE_HAL_STREAMING_SCHEDULING_MODE_BLOCKING_SYNC:
        *flags |= CU_CTX_SCHED_BLOCKING_SYNC;
        break;
      case IREE_HAL_STREAMING_SCHEDULING_MODE_AUTO:
      default:
        // CU_CTX_SCHED_AUTO is 0, so nothing to add.
        break;
    }

    // Add other flags.
    if (context_flags.map_host_memory) {
      *flags |= CU_CTX_MAP_HOST;
    }
    if (context_flags.resize_local_mem_to_max) {
      *flags |= CU_CTX_LMEM_RESIZE_TO_MAX;
    }
  }

  if (active) {
    *active = is_active ? 1 : 0;
  }

  IREE_TRACE_ZONE_END(z0);
  return CUDA_SUCCESS;
}

CUDAAPI CUresult cuDevicePrimaryCtxReset(CUdevice dev) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Validate device.
  iree_hal_streaming_device_registry_t* device_registry =
      iree_hal_streaming_device_registry();
  if (dev < 0 || !device_registry ||
      dev >= (int)device_registry->device_count) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_DEVICE;
  }

  iree_hal_streaming_device_t* device = iree_hal_streaming_device_entry(dev);
  if (!device) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_DEVICE;
  }

  // Reset the primary context by:
  // 1. Waiting for all operations to complete (if it exists)
  // 2. Releasing the current context
  // 3. The context will be recreated lazily on next access

  if (device->primary_context) {
    // Wait for all operations on the context to complete.
    iree_status_t status = iree_hal_streaming_context_wait_idle(
        device->primary_context, iree_infinite_timeout());
    if (!iree_status_is_ok(status)) {
      iree_status_free(status);
      IREE_TRACE_ZONE_END(z0);
      return CUDA_ERROR_UNKNOWN;
    }

    // Lock to ensure thread safety during reset.
    iree_slim_mutex_lock(&device->primary_context_mutex);

    // Release the old context.
    iree_hal_streaming_context_release(device->primary_context);
    device->primary_context = NULL;

    // Also clear memory pools.
    if (device->default_mem_pool) {
      iree_hal_streaming_mem_pool_release(device->default_mem_pool);
      device->default_mem_pool = NULL;
    }
    if (device->current_mem_pool) {
      iree_hal_streaming_mem_pool_release(device->current_mem_pool);
      device->current_mem_pool = NULL;
    }

    iree_slim_mutex_unlock(&device->primary_context_mutex);

    // Clear current context if it was the primary context.
    iree_hal_streaming_context_t* current_context =
        iree_hal_streaming_context_current();
    if (current_context == device->primary_context) {
      iree_hal_streaming_context_set_current(NULL);
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return CUDA_SUCCESS;
}

CUDAAPI CUresult cuDeviceGetByPCIBusId(CUdevice* dev, const char* pciBusId) {
  IREE_TRACE_ZONE_BEGIN(z0);
  // TODO: Implement PCI bus ID lookup.
  IREE_TRACE_ZONE_END(z0);
  return CUDA_ERROR_NOT_SUPPORTED;
}

CUDAAPI CUresult cuDeviceGetPCIBusId(char* pciBusId, int len, CUdevice dev) {
  IREE_TRACE_ZONE_BEGIN(z0);
  // TODO: Implement PCI bus ID retrieval.
  IREE_TRACE_ZONE_END(z0);
  return CUDA_ERROR_NOT_SUPPORTED;
}

//===----------------------------------------------------------------------===//
// Event management
//===----------------------------------------------------------------------===//

CUDAAPI CUresult cuEventCreateWithFlags(CUevent* event, unsigned flags) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!event) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  iree_hal_streaming_context_t* context = iree_hal_streaming_context_current();
  if (!context) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_CONTEXT;
  }

  iree_hal_streaming_event_t* stream_event = NULL;
  iree_status_t status = iree_hal_streaming_event_create(
      context, cuda_event_flags_to_internal(flags), iree_allocator_system(),
      &stream_event);

  if (iree_status_is_ok(status)) {
    *event = (CUevent)stream_event;
  }

  CUresult result = iree_status_to_cu_result(status);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
  }
  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuEventCreate(CUevent* phEvent, unsigned int Flags) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!phEvent) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }
  iree_hal_streaming_context_t* context = iree_hal_streaming_context_current();
  if (!context) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_CONTEXT;
  }

  iree_hal_streaming_event_t* event = NULL;
  iree_status_t status = iree_hal_streaming_event_create(
      context, cuda_event_flags_to_internal(Flags), context->host_allocator,
      &event);

  if (iree_status_is_ok(status)) {
    *phEvent = (CUevent)event;
  }

  CUresult result = iree_status_to_cu_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuEventRecord(CUevent hEvent, CUstream hStream) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status =
      iree_hal_streaming_event_record((iree_hal_streaming_event_t*)hEvent,
                                      (iree_hal_streaming_stream_t*)hStream);
  CUresult result = iree_status_to_cu_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuEventQuery(CUevent hEvent) {
  int is_complete = 0;
  iree_status_t status = iree_hal_streaming_event_query(
      (iree_hal_streaming_event_t*)hEvent, &is_complete);
  CUresult result =
      iree_status_is_ok(status)
          ? (is_complete == 1 ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY)
          : iree_status_to_cu_result(status);
  return result;
}

CUDAAPI CUresult cuEventSynchronize(CUevent hEvent) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status =
      iree_hal_streaming_event_synchronize((iree_hal_streaming_event_t*)hEvent);
  CUresult result = iree_status_to_cu_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuEventDestroy(CUevent hEvent) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_streaming_event_release((iree_hal_streaming_event_t*)hEvent);
  IREE_TRACE_ZONE_END(z0);
  return CUDA_SUCCESS;
}

CUDAAPI CUresult cuEventElapsedTime(float* pMilliseconds, CUevent hStart,
                                    CUevent hEnd) {
  if (!pMilliseconds) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  iree_status_t status = iree_hal_streaming_event_elapsed_time(
      pMilliseconds, (iree_hal_streaming_event_t*)hStart,
      (iree_hal_streaming_event_t*)hEnd);
  CUresult result = iree_status_to_cu_result(status);
  return result;
}

CUDAAPI CUresult cuIpcGetEventHandle(CUipcEventHandle* pHandle, CUevent event) {
  IREE_TRACE_ZONE_BEGIN(z0);
  // TODO: Implement IPC event handles.
  IREE_TRACE_ZONE_END(z0);
  return CUDA_ERROR_NOT_SUPPORTED;
}

CUDAAPI CUresult cuIpcOpenEventHandle(CUevent* phEvent,
                                      CUipcEventHandle handle) {
  IREE_TRACE_ZONE_BEGIN(z0);
  // TODO: Implement IPC event handles.
  IREE_TRACE_ZONE_END(z0);
  return CUDA_ERROR_NOT_SUPPORTED;
}

//===----------------------------------------------------------------------===//
// Memory management
//===----------------------------------------------------------------------===//

CUDAAPI CUresult cuMemGetInfo(size_t* free, size_t* total) {
  iree_hal_streaming_context_t* context = iree_hal_streaming_context_current();
  if (!context) {
    return CUDA_ERROR_INVALID_CONTEXT;
  }

  iree_device_size_t free_memory = 0;
  iree_device_size_t total_memory = 0;
  iree_status_t status = iree_hal_streaming_device_memory_info(
      context->device_ordinal, &free_memory, &total_memory);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return CUDA_ERROR_INVALID_DEVICE;
  }

  if (free) *free = free_memory;
  if (total) *total = total_memory;

  return CUDA_SUCCESS;
}

CUDAAPI CUresult cuMemAlloc(CUdeviceptr* dptr, size_t bytesize) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!dptr) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  iree_hal_streaming_context_t* context = iree_hal_streaming_context_current();
  if (!context) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_CONTEXT;
  }

  iree_hal_streaming_buffer_t* buffer = NULL;
  iree_status_t status = iree_hal_streaming_memory_allocate_device(
      context, bytesize, IREE_HAL_STREAMING_MEMORY_FLAG_NONE, &buffer);

  if (iree_status_is_ok(status)) {
    *dptr = (CUdeviceptr)buffer->device_ptr;
  }

  CUresult result = iree_status_to_cu_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuMemAllocPitch(CUdeviceptr* dptr, size_t* pPitch,
                                 size_t WidthInBytes, size_t Height,
                                 unsigned int ElementSizeBytes) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!dptr || !pPitch) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  // Validate ElementSizeBytes - should be 4, 8, or 16 for coalesced access.
  if (ElementSizeBytes != 4 && ElementSizeBytes != 8 &&
      ElementSizeBytes != 16) {
    // We still allow it but it may have reduced performance.
    IREE_TRACE_ZONE_APPEND_TEXT(z0, "Non-optimal ElementSizeBytes");
  }

  // Get current context.
  iree_hal_streaming_context_t* context = iree_hal_streaming_context_current();
  if (!context) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_CONTEXT;
  }

  // Allocate pitched memory.
  size_t pitch = 0;
  iree_hal_streaming_buffer_t* buffer = NULL;
  iree_status_t status = iree_hal_streaming_memory_allocate_device_pitched(
      context, WidthInBytes, Height, ElementSizeBytes, &pitch, &buffer);

  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_OUT_OF_MEMORY;
  }

  // Return device pointer and pitch.
  *dptr = (CUdeviceptr)iree_hal_streaming_buffer_device_pointer(buffer);
  *pPitch = pitch;

  IREE_TRACE_ZONE_END(z0);
  return CUDA_SUCCESS;
}

CUDAAPI CUresult cuMemFree(CUdeviceptr dptr) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_streaming_context_t* context = iree_hal_streaming_context_current();
  if (!context) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_CONTEXT;
  }

  iree_status_t status = iree_hal_streaming_memory_free_device(context, dptr);

  CUresult result = iree_status_to_cu_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuMemGetAddressRange(CUdeviceptr* pbase, size_t* psize,
                                      CUdeviceptr dptr) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!pbase || !psize) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  iree_hal_streaming_context_t* context = iree_hal_streaming_context_current();
  if (!context) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_CONTEXT;
  }

  iree_hal_streaming_deviceptr_t base = 0;
  size_t size = 0;
  iree_status_t status = iree_hal_streaming_memory_address_range(
      context, (iree_hal_streaming_deviceptr_t)dptr, &base, &size);

  if (iree_status_is_ok(status)) {
    *pbase = (CUdeviceptr)base;
    *psize = size;
  }

  CUresult result = iree_status_to_cu_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuMemAllocHost(void** pp, size_t bytesize) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!pp) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  iree_hal_streaming_context_t* context = iree_hal_streaming_context_current();
  if (!context) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_CONTEXT;
  }

  iree_hal_streaming_buffer_t* buffer = NULL;
  iree_status_t status = iree_hal_streaming_memory_allocate_host(
      context, bytesize, IREE_HAL_STREAMING_MEMORY_FLAG_NONE, &buffer);

  if (iree_status_is_ok(status)) {
    *pp = buffer->host_ptr;
  }

  CUresult result = iree_status_to_cu_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuMemFreeHost(void* p) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_streaming_context_t* context = iree_hal_streaming_context_current();
  if (!context) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_CONTEXT;
  }

  iree_status_t status = iree_hal_streaming_memory_free_host(context, p);

  CUresult result = iree_status_to_cu_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuMemHostAlloc(void** pp, size_t bytesize,
                                unsigned int Flags) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!pp) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  iree_hal_streaming_context_t* context = iree_hal_streaming_context_current();
  if (!context) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_CONTEXT;
  }

  iree_hal_streaming_buffer_t* buffer = NULL;
  iree_status_t status = iree_hal_streaming_memory_allocate_host(
      context, bytesize, cuda_memory_flags_to_internal(Flags), &buffer);

  if (iree_status_is_ok(status)) {
    *pp = buffer->host_ptr;
  }

  CUresult result = iree_status_to_cu_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuMemHostGetDevicePointer(CUdeviceptr* pdptr, void* p,
                                           unsigned int Flags) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!pdptr || !p) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  iree_hal_streaming_context_t* context = iree_hal_streaming_context_current();
  if (!context) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_CONTEXT;
  }

  // Look up the buffer from the host pointer.
  iree_hal_streaming_buffer_ref_t buffer_ref;
  iree_status_t status = iree_hal_streaming_memory_lookup(
      context, (iree_hal_streaming_deviceptr_t)p, &buffer_ref);

  if (iree_status_is_ok(status)) {
    // For registered host memory, the device pointer is the same as host
    // pointer.
    *pdptr = (CUdeviceptr)(buffer_ref.buffer->device_ptr + buffer_ref.offset);
  }

  CUresult result = iree_status_to_cu_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuMemHostGetFlags(unsigned int* pFlags, void* p) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!pFlags || !p) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  iree_hal_streaming_context_t* context = iree_hal_streaming_context_current();
  if (!context) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_CONTEXT;
  }

  // Get the internal flags.
  iree_hal_streaming_host_register_flags_t internal_flags;
  iree_status_t status =
      iree_hal_streaming_memory_host_flags(context, p, &internal_flags);

  if (iree_status_is_ok(status)) {
    // Convert internal flags back to CUDA flags.
    unsigned int cuda_flags = 0;
    if (internal_flags & IREE_HAL_STREAMING_HOST_REGISTER_FLAG_PORTABLE) {
      cuda_flags |= CU_MEMHOSTREGISTER_PORTABLE;
    }
    if (internal_flags & IREE_HAL_STREAMING_HOST_REGISTER_FLAG_MAPPED) {
      cuda_flags |= CU_MEMHOSTREGISTER_DEVICEMAP;
    }
    if (internal_flags & IREE_HAL_STREAMING_HOST_REGISTER_FLAG_WRITE_COMBINED) {
      cuda_flags |= CU_MEMHOSTREGISTER_IOMEMORY;
    }
    if (internal_flags & IREE_HAL_STREAMING_HOST_REGISTER_FLAG_READ_ONLY) {
      cuda_flags |= CU_MEMHOSTREGISTER_READ_ONLY;
    }
    *pFlags = cuda_flags;
  }

  CUresult result = iree_status_to_cu_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuMemAllocManaged(CUdeviceptr* dptr, size_t bytesize,
                                   unsigned int flags) {
  IREE_TRACE_ZONE_BEGIN(z0);
  // TODO: Implement managed memory allocation.
  IREE_TRACE_ZONE_END(z0);
  return CUDA_ERROR_NOT_SUPPORTED;
}

CUDAAPI CUresult cuMemHostRegister(void* p, size_t bytesize,
                                   unsigned int Flags) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!p || bytesize == 0) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  iree_hal_streaming_context_t* context = iree_hal_streaming_context_current();
  if (!context) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_CONTEXT;
  }

  // Convert CUDA flags to internal flags.
  iree_hal_streaming_host_register_flags_t internal_flags =
      IREE_HAL_STREAMING_HOST_REGISTER_FLAG_DEFAULT;
  if (Flags & CU_MEMHOSTREGISTER_PORTABLE) {
    internal_flags |= IREE_HAL_STREAMING_HOST_REGISTER_FLAG_PORTABLE;
  }
  if (Flags & CU_MEMHOSTREGISTER_DEVICEMAP) {
    internal_flags |= IREE_HAL_STREAMING_HOST_REGISTER_FLAG_MAPPED;
  }
  if (Flags & CU_MEMHOSTREGISTER_IOMEMORY) {
    internal_flags |= IREE_HAL_STREAMING_HOST_REGISTER_FLAG_WRITE_COMBINED;
  }
  if (Flags & CU_MEMHOSTREGISTER_READ_ONLY) {
    internal_flags |= IREE_HAL_STREAMING_HOST_REGISTER_FLAG_READ_ONLY;
  }

  // Register the host memory using the internal function.
  iree_hal_streaming_buffer_t* buffer = NULL;
  iree_status_t status = iree_hal_streaming_memory_register_host(
      context, p, bytesize, internal_flags, &buffer);

  CUresult result = iree_status_to_cu_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuMemHostUnregister(void* p) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!p) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  iree_hal_streaming_context_t* context = iree_hal_streaming_context_current();
  if (!context) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_CONTEXT;
  }

  // Unregister the host memory using the internal function.
  iree_status_t status = iree_hal_streaming_memory_unregister_host(context, p);

  CUresult result = iree_status_to_cu_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_streaming_context_t* context = iree_hal_streaming_context_current();
  if (!context) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_CONTEXT;
  }

  // Use device-to-device copy (synchronous).
  iree_status_t status = iree_hal_streaming_memcpy_device_to_device(
      context, (iree_hal_streaming_deviceptr_t)dst,
      (iree_hal_streaming_deviceptr_t)src, ByteCount, NULL);

  CUresult result = iree_status_to_cu_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuMemcpyPeer(CUdeviceptr dstDevice, CUcontext dstContext,
                              CUdeviceptr srcDevice, CUcontext srcContext,
                              size_t ByteCount) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!dstContext || !srcContext) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  // Perform peer-to-peer copy between contexts.
  iree_status_t status = iree_hal_streaming_memcpy_peer(
      (iree_hal_streaming_context_t*)dstContext,
      (iree_hal_streaming_deviceptr_t)dstDevice,
      (iree_hal_streaming_context_t*)srcContext,
      (iree_hal_streaming_deviceptr_t)srcDevice, ByteCount, NULL);

  CUresult result = iree_status_to_cu_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void* srcHost,
                              size_t ByteCount) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_streaming_context_t* context = iree_hal_streaming_context_current();
  if (!context) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_CONTEXT;
  }

  iree_status_t status = iree_hal_streaming_memcpy_host_to_device(
      context, dstDevice, srcHost, ByteCount, NULL);

  CUresult result = iree_status_to_cu_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuMemcpyDtoH(void* dstHost, CUdeviceptr srcDevice,
                              size_t ByteCount) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_streaming_context_t* context = iree_hal_streaming_context_current();
  if (!context) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_CONTEXT;
  }

  iree_status_t status = iree_hal_streaming_memcpy_device_to_host(
      context, dstHost, srcDevice, ByteCount, NULL);

  CUresult result = iree_status_to_cu_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                              size_t ByteCount) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_streaming_context_t* context = iree_hal_streaming_context_current();
  if (!context) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_CONTEXT;
  }

  iree_status_t status = iree_hal_streaming_memcpy_device_to_device(
      context, (iree_hal_streaming_deviceptr_t)dstDevice,
      (iree_hal_streaming_deviceptr_t)srcDevice, ByteCount, NULL);

  CUresult result = iree_status_to_cu_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src,
                               size_t ByteCount, CUstream hStream) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_streaming_context_t* context = iree_hal_streaming_context_current();
  if (!context) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_CONTEXT;
  }

  // Resolve NULL stream to default stream.
  if (!hStream) {
    hStream = (CUstream)context->default_stream;
  }

  // Use device-to-device copy (asynchronous).
  iree_status_t status = iree_hal_streaming_memcpy_device_to_device(
      context, (iree_hal_streaming_deviceptr_t)dst,
      (iree_hal_streaming_deviceptr_t)src, ByteCount,
      (iree_hal_streaming_stream_t*)hStream);

  CUresult result = iree_status_to_cu_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuMemcpyPeerAsync(CUdeviceptr dstDevice, CUcontext dstContext,
                                   CUdeviceptr srcDevice, CUcontext srcContext,
                                   size_t ByteCount, CUstream hStream) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!dstContext || !srcContext) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  // Perform peer-to-peer copy between contexts (asynchronous).
  iree_status_t status = iree_hal_streaming_memcpy_peer(
      (iree_hal_streaming_context_t*)dstContext,
      (iree_hal_streaming_deviceptr_t)dstDevice,
      (iree_hal_streaming_context_t*)srcContext,
      (iree_hal_streaming_deviceptr_t)srcDevice, ByteCount,
      (iree_hal_streaming_stream_t*)hStream);

  CUresult result = iree_status_to_cu_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuMemcpyHtoDAsync(CUdeviceptr dstDevice, const void* srcHost,
                                   size_t ByteCount, CUstream hStream) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_streaming_context_t* context = iree_hal_streaming_context_current();
  if (!context) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_CONTEXT;
  }

  iree_status_t status = iree_hal_streaming_memcpy_host_to_device(
      context, (iree_hal_streaming_deviceptr_t)dstDevice, srcHost, ByteCount,
      (iree_hal_streaming_stream_t*)hStream);

  CUresult result = iree_status_to_cu_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuMemcpyDtoHAsync(void* dstHost, CUdeviceptr srcDevice,
                                   size_t ByteCount, CUstream hStream) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_streaming_context_t* context = iree_hal_streaming_context_current();
  if (!context) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_CONTEXT;
  }

  iree_status_t status = iree_hal_streaming_memcpy_device_to_host(
      context, dstHost, (iree_hal_streaming_deviceptr_t)srcDevice, ByteCount,
      (iree_hal_streaming_stream_t*)hStream);

  CUresult result = iree_status_to_cu_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuMemcpyDtoDAsync(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                                   size_t ByteCount, CUstream hStream) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_streaming_context_t* context = iree_hal_streaming_context_current();
  if (!context) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_CONTEXT;
  }

  iree_status_t status = iree_hal_streaming_memcpy_device_to_device(
      context, (iree_hal_streaming_deviceptr_t)dstDevice,
      (iree_hal_streaming_deviceptr_t)srcDevice, ByteCount,
      (iree_hal_streaming_stream_t*)hStream);

  CUresult result = iree_status_to_cu_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuMemset(void* dst, int value, size_t sizeBytes) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!dst) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  iree_hal_streaming_context_t* context = iree_hal_streaming_context_current();
  if (!context) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_CONTEXT;
  }

  uint8_t pattern = (uint8_t)value;
  iree_status_t status = iree_hal_streaming_memory_memset(
      context, (iree_hal_streaming_deviceptr_t)dst, sizeBytes, &pattern,
      sizeof(pattern), context->default_stream);

  CUresult result = iree_status_to_cu_result(status);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
  }
  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuMemsetAsync(void* dst, int value, size_t sizeBytes,
                               CUstream hStream) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!dst) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  iree_hal_streaming_context_t* context = iree_hal_streaming_context_current();
  if (!context) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_CONTEXT;
  }

  uint8_t pattern = (uint8_t)value;
  iree_status_t status = iree_hal_streaming_memory_memset(
      context, (iree_hal_streaming_deviceptr_t)dst, sizeBytes, &pattern,
      sizeof(pattern),
      hStream ? (iree_hal_streaming_stream_t*)hStream
              : context->default_stream);

  CUresult result = iree_status_to_cu_result(status);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
  }
  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuMemcpyWithStream(void* dst, const void* src,
                                    size_t sizeBytes, int kind,
                                    CUstream hStream) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!dst || !src) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  iree_hal_streaming_context_t* context = iree_hal_streaming_context_current();
  if (!context) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_CONTEXT;
  }

  iree_hal_streaming_stream_t* stream = (iree_hal_streaming_stream_t*)hStream;

  // Map the copy kind and perform the appropriate transfer.
  iree_status_t status = iree_ok_status();
  switch (kind) {
    case 1:  // Host to device
      status = iree_hal_streaming_memcpy_host_to_device(
          context, (iree_hal_streaming_deviceptr_t)dst, src, sizeBytes, stream);
      break;
    case 2:  // Device to host
      status = iree_hal_streaming_memcpy_device_to_host(
          context, dst, (iree_hal_streaming_deviceptr_t)src, sizeBytes, stream);
      break;
    case 3:  // Device to device
      status = iree_hal_streaming_memcpy_device_to_device(
          context, (iree_hal_streaming_deviceptr_t)dst,
          (iree_hal_streaming_deviceptr_t)src, sizeBytes, stream);
      break;
    default:
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "unsupported memory copy kind: %d", kind);
      break;
  }

  CUresult result = iree_status_to_cu_result(status);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
  }
  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuMemsetD8(CUdeviceptr dstDevice, unsigned char uc, size_t N) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_streaming_context_t* context = iree_hal_streaming_context_current();
  if (!context) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_CONTEXT;
  }

  iree_status_t status = iree_hal_streaming_memory_memset(
      context, (iree_hal_streaming_deviceptr_t)dstDevice, N * 1, &uc, 1,
      context->default_stream);

  CUresult result = iree_status_to_cu_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuMemsetD16(CUdeviceptr dstDevice, unsigned short us,
                             size_t N) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_streaming_context_t* context = iree_hal_streaming_context_current();
  if (!context) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_CONTEXT;
  }

  iree_status_t status = iree_hal_streaming_memory_memset(
      context, (iree_hal_streaming_deviceptr_t)dstDevice, N * 2, &us, 2,
      context->default_stream);

  CUresult result = iree_status_to_cu_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuMemsetD32(CUdeviceptr dstDevice, unsigned int ui, size_t N) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_streaming_context_t* context = iree_hal_streaming_context_current();
  if (!context) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_CONTEXT;
  }

  iree_status_t status = iree_hal_streaming_memory_memset(
      context, (iree_hal_streaming_deviceptr_t)dstDevice, N * 4, &ui, 4,
      context->default_stream);

  CUresult result = iree_status_to_cu_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc,
                                 size_t N, CUstream hStream) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_streaming_context_t* context = iree_hal_streaming_context_current();
  if (!context) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_CONTEXT;
  }

  iree_status_t status = iree_hal_streaming_memory_memset(
      context, (iree_hal_streaming_deviceptr_t)dstDevice, N * 1, &uc, 1,
      hStream ? (iree_hal_streaming_stream_t*)hStream
              : context->default_stream);

  CUresult result = iree_status_to_cu_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuMemsetD16Async(CUdeviceptr dstDevice, unsigned short us,
                                  size_t N, CUstream hStream) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_streaming_context_t* context = iree_hal_streaming_context_current();
  if (!context) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_CONTEXT;
  }

  iree_status_t status = iree_hal_streaming_memory_memset(
      context, (iree_hal_streaming_deviceptr_t)dstDevice, N * 2, &us, 2,
      hStream ? (iree_hal_streaming_stream_t*)hStream
              : context->default_stream);

  CUresult result = iree_status_to_cu_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui,
                                  size_t N, CUstream hStream) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_streaming_context_t* context = iree_hal_streaming_context_current();
  if (!context) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_CONTEXT;
  }

  iree_status_t status = iree_hal_streaming_memory_memset(
      context, (iree_hal_streaming_deviceptr_t)dstDevice, N * 4, &ui, 4,
      hStream ? (iree_hal_streaming_stream_t*)hStream
              : context->default_stream);

  CUresult result = iree_status_to_cu_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuIpcGetMemHandle(CUipcMemHandle* pHandle, CUdeviceptr dptr) {
  IREE_TRACE_ZONE_BEGIN(z0);
  // TODO: Implement IPC memory handles.
  IREE_TRACE_ZONE_END(z0);
  return CUDA_ERROR_NOT_SUPPORTED;
}

CUDAAPI CUresult cuIpcOpenMemHandle(CUdeviceptr* pdptr, CUipcMemHandle handle,
                                    unsigned int Flags) {
  IREE_TRACE_ZONE_BEGIN(z0);
  // TODO: Implement IPC memory handles.
  IREE_TRACE_ZONE_END(z0);
  return CUDA_ERROR_NOT_SUPPORTED;
}

CUDAAPI CUresult cuIpcCloseMemHandle(CUdeviceptr dptr) {
  IREE_TRACE_ZONE_BEGIN(z0);
  // TODO: Implement IPC memory handles.
  IREE_TRACE_ZONE_END(z0);
  return CUDA_ERROR_NOT_SUPPORTED;
}

//===----------------------------------------------------------------------===//
// Module management
//===----------------------------------------------------------------------===//

CUDAAPI CUresult cuModuleLoad(CUmodule* module, const char* fname) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!module || !fname) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  iree_hal_streaming_context_t* context = iree_hal_streaming_context_current();
  if (!context) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_CONTEXT;
  }

  iree_hal_executable_caching_mode_t caching_mode =
      IREE_HAL_EXECUTABLE_CACHING_MODE_ALLOW_PERSISTENT_CACHING |
      IREE_HAL_EXECUTABLE_CACHING_MODE_ALLOW_OPTIMIZATION;
  iree_hal_streaming_module_t* stream_module = NULL;
  iree_status_t status = iree_hal_streaming_module_create_from_file(
      context, caching_mode, iree_make_cstring_view(fname),
      context->host_allocator, &stream_module);

  if (iree_status_is_ok(status)) {
    *module = (CUmodule)stream_module;
  }

  CUresult result = iree_status_to_cu_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuModuleLoadData(CUmodule* module, const void* image) {
  // Call the extended version with no options.
  return cuModuleLoadDataEx(module, image, 0, NULL, NULL);
}

CUDAAPI CUresult cuModuleLoadDataEx(CUmodule* module, const void* image,
                                    unsigned int numOptions,
                                    CUjit_option* options,
                                    void** optionValues) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!module || !image) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  iree_hal_streaming_context_t* context = iree_hal_streaming_context_current();
  if (!context) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_CONTEXT;
  }

  // Process JIT options if provided.
  // Note: Most JIT options are informational or optimization hints that may
  // not apply to our HAL backend. We parse them for compatibility but may not
  // use all of them.
  for (unsigned int i = 0; i < numOptions; ++i) {
    if (!options || !optionValues) continue;
    switch (options[i]) {
      case CU_JIT_MAX_REGISTERS:
        // Maximum number of registers per thread.
        // This could influence kernel compilation but may be backend-specific.
        break;
      case CU_JIT_THREADS_PER_BLOCK:
        // Minimum number of threads per block.
        break;
      case CU_JIT_WALL_TIME:
        // Wall time for compilation in milliseconds.
        break;
      case CU_JIT_INFO_LOG_BUFFER:
        // Buffer for informational log.
        break;
      case CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES:
        // Size of info log buffer.
        break;
      case CU_JIT_ERROR_LOG_BUFFER:
        // Buffer for error log.
        break;
      case CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES:
        // Size of error log buffer.
        break;
      case CU_JIT_OPTIMIZATION_LEVEL:
        // Optimization level (0-4).
        break;
      case CU_JIT_TARGET_FROM_CUCONTEXT:
        // Use target from current context.
        break;
      case CU_JIT_TARGET:
        // Explicit compute capability target.
        break;
      case CU_JIT_FALLBACK_STRATEGY:
        // Fallback strategy for compilation.
        break;
      case CU_JIT_GENERATE_DEBUG_INFO:
        // Generate debug information.
        break;
      case CU_JIT_LOG_VERBOSE:
        // Enable verbose logging.
        break;
      case CU_JIT_GENERATE_LINE_INFO:
        // Generate line number information.
        break;
      case CU_JIT_CACHE_MODE:
        // Cache mode for compiled kernels.
        break;
      case CU_JIT_NEW_SM3X_OPT:
        // SM 3.x specific optimizations.
        break;
      case CU_JIT_FAST_COMPILE:
        // Fast compilation mode.
        break;
      case CU_JIT_GLOBAL_SYMBOL_NAMES:
        // Array of global symbol names.
        break;
      case CU_JIT_GLOBAL_SYMBOL_ADDRESSES:
        // Array of global symbol addresses.
        break;
      case CU_JIT_GLOBAL_SYMBOL_COUNT:
        // Number of global symbols.
        break;
      default:
        // Unknown option, ignore.
        break;
    }
  }

  // TODO: Determine caching mode from JIT options if available.
  iree_hal_executable_caching_mode_t caching_mode =
      IREE_HAL_EXECUTABLE_CACHING_MODE_ALIAS_PROVIDED_DATA |
      IREE_HAL_EXECUTABLE_CACHING_MODE_ALLOW_OPTIMIZATION;

  iree_hal_streaming_module_t* stream_module = NULL;
  iree_status_t status = iree_hal_streaming_module_create_from_memory(
      context, caching_mode, iree_make_const_byte_span(image, 0),
      context->host_allocator, &stream_module);

  if (iree_status_is_ok(status)) {
    *module = (CUmodule)stream_module;
  }

  CUresult result = iree_status_to_cu_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuModuleLoadFatBinary(CUmodule* module, const void* fatCubin) {
  IREE_TRACE_ZONE_BEGIN(z0);
  CUresult result = cuModuleLoadDataEx(module, fatCubin, 0, NULL, NULL);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuModuleUnload(CUmodule hmod) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_streaming_module_release((iree_hal_streaming_module_t*)hmod);
  IREE_TRACE_ZONE_END(z0);
  return CUDA_SUCCESS;
}

CUDAAPI CUresult cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod,
                                     const char* name) {
  if (!hfunc || !name) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  iree_hal_streaming_module_t* module = (iree_hal_streaming_module_t*)hmod;
  if (!module) {
    return CUDA_ERROR_INVALID_HANDLE;
  }

  iree_hal_streaming_symbol_t* symbol = NULL;
  iree_status_t status =
      iree_hal_streaming_module_function(module, name, &symbol);

  if (iree_status_is_ok(status)) {
    *hfunc = (CUfunction)symbol;
  }

  CUresult result = iree_status_to_cu_result(status);
  return result;
}

CUDAAPI CUresult cuModuleGetGlobal(CUdeviceptr* dptr, size_t* bytes,
                                   CUmodule hmod, const char* name) {
  if (!dptr || !hmod || !name) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  iree_hal_streaming_module_t* module = (iree_hal_streaming_module_t*)hmod;
  iree_hal_streaming_deviceptr_t device_ptr = 0;
  iree_device_size_t size = 0;
  iree_status_t status =
      iree_hal_streaming_module_global(module, name, &device_ptr, &size);

  if (iree_status_is_ok(status)) {
    *dptr = (CUdeviceptr)device_ptr;
    if (bytes) *bytes = (size_t)size;
  }

  CUresult result = iree_status_to_cu_result(status);
  return result;
}

//===----------------------------------------------------------------------===//
// Function management
//===----------------------------------------------------------------------===//

CUDAAPI CUresult cuFuncGetAttribute(int* pi, CUfunction_attribute attrib,
                                    CUfunction hfunc) {
  if (!pi || !hfunc) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  // Cast to symbol pointer.
  iree_hal_streaming_symbol_t* symbol = (iree_hal_streaming_symbol_t*)hfunc;

  // Verify it's a function.
  if (symbol->type != IREE_HAL_STREAMING_SYMBOL_TYPE_FUNCTION) {
    return CUDA_ERROR_INVALID_HANDLE;
  }

  // Return attribute value based on what we have cached.
  switch (attrib) {
    case CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK:
      *pi = symbol->max_threads_per_block;
      break;
    case CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES:
      *pi = symbol->shared_size_bytes;
      break;
    case CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES:
      // We don't track constant memory usage.
      *pi = 0;
      break;
    case CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES:
      // Local memory is typically 0 for modern GPUs.
      *pi = 0;
      break;
    case CU_FUNC_ATTRIBUTE_NUM_REGS:
      *pi = symbol->num_regs;
      break;
    case CU_FUNC_ATTRIBUTE_PTX_VERSION:
      // Return a default PTX version.
      *pi = 70;  // PTX 7.0
      break;
    case CU_FUNC_ATTRIBUTE_BINARY_VERSION:
      // Return a default binary version.
      *pi = 75;  // SM 7.5
      break;
    case CU_FUNC_ATTRIBUTE_CACHE_MODE_CA:
      // Cache mode is not tracked.
      *pi = 0;
      break;
    case CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES:
      // Return the kernel's maximum dynamic shared memory size.
      *pi = symbol->max_dynamic_shared_size_bytes;
      break;
    case CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT:
      // Carveout percentage not tracked.
      *pi = 0;
      break;
    case CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET:
      *pi = 0;  // Clusters not required.
      break;
    case CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH:
    case CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT:
    case CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH:
      *pi = 1;  // Default cluster dimensions.
      break;
    default:
      return CUDA_ERROR_INVALID_VALUE;
  }

  return CUDA_SUCCESS;
}

CUDAAPI CUresult cuFuncSetAttribute(CUfunction hfunc,
                                    CUfunction_attribute attrib, int value) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!hfunc) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  // Cast to symbol pointer.
  iree_hal_streaming_symbol_t* symbol = (iree_hal_streaming_symbol_t*)hfunc;

  // Verify it's a function.
  if (symbol->type != IREE_HAL_STREAMING_SYMBOL_TYPE_FUNCTION) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_HANDLE;
  }

  // Only certain attributes can be set.
  CUresult result = CUDA_SUCCESS;
  switch (attrib) {
    case CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES:
      // Store the maximum dynamic shared memory size for this function.
      // This is used by kernels that need more than 48KB shared memory.
      // Note that this is not actually used for anything but queries.
      symbol->max_dynamic_shared_size_bytes = (uint32_t)value;
      break;
    case CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT:
      // This controls the L1/shared memory split.
      // Values are percentages (0, 25, 50, 75, 100).
      // We don't actually configure this in the stream HAL yet.
      if (value != 0 && value != 25 && value != 50 && value != 75 &&
          value != 100) {
        result = CUDA_ERROR_INVALID_VALUE;
      }
      break;
    default:
      // Most attributes are read-only.
      result = CUDA_ERROR_INVALID_VALUE;
      break;
  }

  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!hfunc) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  // Validate cache configuration.
  CUresult result = CUDA_SUCCESS;
  switch (config) {
    case CU_FUNC_CACHE_PREFER_NONE:
    case CU_FUNC_CACHE_PREFER_SHARED:
    case CU_FUNC_CACHE_PREFER_L1:
    case CU_FUNC_CACHE_PREFER_EQUAL:
      // These are all valid configurations.
      // We don't actually configure cache in the stream HAL yet,
      // but we accept the values.
      break;
    default:
      result = CUDA_ERROR_INVALID_VALUE;
      break;
  }

  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuFuncSetSharedMemConfig(CUfunction hfunc,
                                          CUsharedconfig config) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!hfunc) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  // Validate shared memory configuration.
  CUresult result = CUDA_SUCCESS;
  switch (config) {
    case CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE:
    case CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE:
    case CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE:
      // These are all valid configurations.
      // We don't actually configure shared memory banks in the stream HAL yet,
      // but we accept the values.
      break;
    default:
      result = CUDA_ERROR_INVALID_VALUE;
      break;
  }

  IREE_TRACE_ZONE_END(z0);
  return result;
}

//===----------------------------------------------------------------------===//
// Occupancy calculation
//===----------------------------------------------------------------------===//

CUDAAPI CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(
    int* numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!numBlocks || !func || blockSize <= 0) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  // Get the current context and device.
  iree_hal_streaming_context_t* context = iree_hal_streaming_context_current();
  if (!context) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_CONTEXT;
  }

  // Get device properties.
  iree_hal_streaming_device_t* device = context->device_entry;
  if (!device) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_DEVICE;
  }

  // Get symbol.
  iree_hal_streaming_symbol_t* symbol = (iree_hal_streaming_symbol_t*)func;

  // Verify it's a function.
  if (symbol->type != IREE_HAL_STREAMING_SYMBOL_TYPE_FUNCTION) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_HANDLE;
  }

  // Use shared occupancy calculation.
  uint32_t max_blocks = 0;
  iree_status_t status =
      iree_hal_streaming_calculate_max_active_blocks_per_multiprocessor(
          device, symbol, (uint32_t)blockSize,
          (iree_host_size_t)dynamicSMemSize, &max_blocks);

  if (iree_status_is_ok(status)) {
    *numBlocks = (int)max_blocks;
  }

  CUresult result = iree_status_to_cu_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int* numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize,
    unsigned int flags) {
  // For now, ignore flags and call the base function.
  // Flags might affect caching behavior but not occupancy calculation.
  return cuOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, func, blockSize,
                                                     dynamicSMemSize);
}

CUDAAPI CUresult cuOccupancyMaxPotentialBlockSize(
    int* minGridSize, int* blockSize, CUfunction func,
    CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize,
    int blockSizeLimit) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!minGridSize || !blockSize || !func) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  // Get the current context and device.
  iree_hal_streaming_context_t* context = iree_hal_streaming_context_current();
  if (!context) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_CONTEXT;
  }

  // Get device properties.
  iree_hal_streaming_device_t* device = context->device_entry;
  if (!device) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_DEVICE;
  }

  // Get symbol.
  iree_hal_streaming_symbol_t* symbol = (iree_hal_streaming_symbol_t*)func;

  // Verify it's a function.
  if (symbol->type != IREE_HAL_STREAMING_SYMBOL_TYPE_FUNCTION) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_HANDLE;
  }

  // Call IREE function with adapted types.
  uint32_t out_block_size = 0;
  uint32_t out_min_grid_size = 0;
  iree_status_t status = iree_hal_streaming_calculate_optimal_block_size(
      device, symbol, (uint32_t)dynamicSMemSize,
      (iree_hal_streaming_block_to_dynamic_smem_fn_t)blockSizeToDynamicSMemSize,
      (uint32_t)blockSizeLimit, &out_block_size, &out_min_grid_size);

  if (iree_status_is_ok(status)) {
    *blockSize = (int)out_block_size;
    *minGridSize = (int)out_min_grid_size;
  }

  CUresult result = iree_status_to_cu_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

//===----------------------------------------------------------------------===//
// Execution control
//===----------------------------------------------------------------------===//

CUDAAPI CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX,
                                unsigned int gridDimY, unsigned int gridDimZ,
                                unsigned int blockDimX, unsigned int blockDimY,
                                unsigned int blockDimZ,
                                unsigned int sharedMemBytes, CUstream hStream,
                                void** kernelParams, void** extra) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Resolve NULL stream to default stream.
  if (!hStream) {
    iree_hal_streaming_context_t* context =
        iree_hal_streaming_context_current();
    if (!context) {
      IREE_TRACE_ZONE_END(z0);
      return CUDA_ERROR_INVALID_CONTEXT;
    }
    hStream = (CUstream)context->default_stream;
  }

  // Extract params pointer from CUDA's parameter format.
  void* params_ptr = NULL;
  if (extra) {
    // Extra format: {CU_LAUNCH_PARAM_BUFFER_POINTER, &buffer,
    //                CU_LAUNCH_PARAM_BUFFER_SIZE, &size, CU_LAUNCH_PARAM_END}
    if (extra[0] == CU_LAUNCH_PARAM_BUFFER_POINTER) {
      params_ptr = *(void**)extra[1];
    }
  } else if (kernelParams) {
    // kernelParams is an array of pointers to the actual parameters.
    // This needs to be packed by the streaming layer.
    params_ptr = kernelParams;
  }

  const iree_hal_streaming_dispatch_params_t params = {
      .grid_dim = {gridDimX, gridDimY, gridDimZ},
      .block_dim = {blockDimX, blockDimY, blockDimZ},
      .shared_memory_bytes = sharedMemBytes,
      .buffer = params_ptr,
      .flags = IREE_HAL_STREAMING_DISPATCH_FLAG_NONE,
  };
  iree_status_t status =
      iree_hal_streaming_launch_kernel((iree_hal_streaming_symbol_t*)f, &params,
                                       (iree_hal_streaming_stream_t*)hStream);

  CUresult result = iree_status_to_cu_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuLaunchCooperativeKernel(
    CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
    unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
    unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream,
    void** kernelParams) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!f) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  // Resolve NULL stream to default stream.
  if (!hStream) {
    iree_hal_streaming_context_t* context =
        iree_hal_streaming_context_current();
    if (!context) {
      IREE_TRACE_ZONE_END(z0);
      return CUDA_ERROR_INVALID_CONTEXT;
    }
    hStream = (CUstream)context->default_stream;
  }

  // Get the current device.
  iree_hal_streaming_context_t* context =
      ((iree_hal_streaming_stream_t*)hStream)->context;
  iree_hal_streaming_device_t* device = context->device_entry;

  // Get symbol.
  iree_hal_streaming_symbol_t* symbol = (iree_hal_streaming_symbol_t*)f;

  // Calculate maximum blocks for cooperative launch.
  // This will return 0 if the device doesn't support cooperative launch.
  int block_size = blockDimX * blockDimY * blockDimZ;
  int max_blocks = 0;
  iree_status_t status = iree_hal_streaming_calculate_max_cooperative_blocks(
      device, symbol, block_size, sharedMemBytes, &max_blocks);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  // Verify grid size doesn't exceed max active blocks.
  // If max_blocks is 0 (device doesn't support cooperative launch) or
  // grid is too large, return error.
  int total_blocks = gridDimX * gridDimY * gridDimZ;
  if (max_blocks == 0 || total_blocks > max_blocks) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE;
  }

  // Set up dispatch params with cooperative flag.
  // Cooperative launch always uses kernelParams format.
  const iree_hal_streaming_dispatch_params_t params = {
      .grid_dim = {gridDimX, gridDimY, gridDimZ},
      .block_dim = {blockDimX, blockDimY, blockDimZ},
      .shared_memory_bytes = sharedMemBytes,
      .buffer = kernelParams,
      .flags = IREE_HAL_STREAMING_DISPATCH_FLAG_COOPERATIVE,
  };

  status =
      iree_hal_streaming_launch_kernel((iree_hal_streaming_symbol_t*)f, &params,
                                       (iree_hal_streaming_stream_t*)hStream);

  CUresult result = iree_status_to_cu_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuLaunchHostFunc(CUstream hStream, CUhostFn fn,
                                  void* userData) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!fn) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  // If no stream is specified, use the default stream.
  iree_hal_streaming_stream_t* stream = (iree_hal_streaming_stream_t*)hStream;
  if (!stream) {
    iree_hal_streaming_context_t* context =
        iree_hal_streaming_context_current();
    if (!context) {
      IREE_TRACE_ZONE_END(z0);
      return CUDA_ERROR_INVALID_CONTEXT;
    }
    stream = context->default_stream;
  }

  iree_status_t status =
      iree_hal_streaming_launch_host_function(stream, fn, userData);

  CUresult result = iree_status_to_cu_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

//===----------------------------------------------------------------------===//
// Stream management
//===----------------------------------------------------------------------===//

CUDAAPI CUresult cuStreamCreateWithFlags(CUstream* phStream,
                                         unsigned int flags) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!phStream) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  iree_hal_streaming_context_t* context = iree_hal_streaming_context_current();
  if (!context) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_CONTEXT;
  }

  iree_hal_streaming_stream_t* stream = NULL;
  iree_status_t status = iree_hal_streaming_stream_create(
      context, cuda_stream_flags_to_internal(flags), 0, iree_allocator_system(),
      &stream);

  if (iree_status_is_ok(status)) {
    *phStream = (CUstream)stream;
  }

  CUresult result = iree_status_to_cu_result(status);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
  }
  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuStreamCreate(CUstream* phStream, unsigned int Flags) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!phStream) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  iree_hal_streaming_context_t* context = iree_hal_streaming_context_current();
  if (!context) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_CONTEXT;
  }

  iree_hal_streaming_stream_t* stream = NULL;
  iree_status_t status = iree_hal_streaming_stream_create(
      context, cuda_stream_flags_to_internal(Flags), 0, context->host_allocator,
      &stream);

  if (iree_status_is_ok(status)) {
    *phStream = (CUstream)stream;
  }

  CUresult result = iree_status_to_cu_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuStreamCreateWithPriority(CUstream* phStream,
                                            unsigned int flags, int priority) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!phStream) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  iree_hal_streaming_context_t* context = iree_hal_streaming_context_current();
  if (!context) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_CONTEXT;
  }

  iree_hal_streaming_stream_t* stream = NULL;
  iree_status_t status = iree_hal_streaming_stream_create(
      context, cuda_stream_flags_to_internal(flags), priority,
      context->host_allocator, &stream);

  if (iree_status_is_ok(status)) {
    *phStream = (CUstream)stream;
  }

  CUresult result = iree_status_to_cu_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuStreamGetPriority(CUstream hStream, int* priority) {
  if (!hStream || !priority) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  iree_hal_streaming_stream_t* stream = (iree_hal_streaming_stream_t*)hStream;
  *priority = stream->priority;
  return CUDA_SUCCESS;
}

CUDAAPI CUresult cuStreamGetFlags(CUstream hStream, unsigned int* flags) {
  if (!hStream || !flags) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  iree_hal_streaming_stream_t* stream = (iree_hal_streaming_stream_t*)hStream;
  *flags = stream->flags;
  return CUDA_SUCCESS;
}

CUDAAPI CUresult cuStreamGetCtx(CUstream hStream, CUcontext* pctx) {
  if (!hStream || !pctx) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  iree_hal_streaming_stream_t* stream = (iree_hal_streaming_stream_t*)hStream;
  *pctx = (CUcontext)stream->context;
  return CUDA_SUCCESS;
}

CUDAAPI CUresult cuStreamGetDevice(CUstream hStream, CUdevice* device) {
  if (!hStream || !device) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  iree_hal_streaming_stream_t* stream = (iree_hal_streaming_stream_t*)hStream;
  if (!stream->context) {
    return CUDA_ERROR_INVALID_HANDLE;
  }

  *device = stream->context->device_ordinal;
  return CUDA_SUCCESS;
}

CUDAAPI CUresult cuStreamWaitEvent(CUstream hStream, CUevent hEvent,
                                   unsigned int Flags) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Resolve NULL stream to default stream.
  if (!hStream) {
    iree_hal_streaming_context_t* context =
        iree_hal_streaming_context_current();
    if (!context) {
      IREE_TRACE_ZONE_END(z0);
      return CUDA_ERROR_INVALID_CONTEXT;
    }
    hStream = (CUstream)context->default_stream;
  }

  iree_status_t status = iree_hal_streaming_stream_wait_event(
      (iree_hal_streaming_stream_t*)hStream,
      (iree_hal_streaming_event_t*)hEvent);
  CUresult result = iree_status_to_cu_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuStreamQuery(CUstream hStream) {
  // Resolve NULL stream to default stream.
  if (!hStream) {
    iree_hal_streaming_context_t* context =
        iree_hal_streaming_context_current();
    if (!context) {
      return CUDA_ERROR_INVALID_CONTEXT;
    }
    hStream = (CUstream)context->default_stream;
  }

  int is_complete = 0;
  iree_status_t status = iree_hal_streaming_stream_query(
      (iree_hal_streaming_stream_t*)hStream, &is_complete);
  CUresult result =
      iree_status_is_ok(status)
          ? (is_complete == 1 ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY)
          : iree_status_to_cu_result(status);
  return result;
}

CUDAAPI CUresult cuStreamSynchronize(CUstream hStream) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Resolve NULL stream to default stream.
  if (!hStream) {
    iree_hal_streaming_context_t* context =
        iree_hal_streaming_context_current();
    if (!context) {
      IREE_TRACE_ZONE_END(z0);
      return CUDA_ERROR_INVALID_CONTEXT;
    }
    hStream = (CUstream)context->default_stream;
  }

  iree_status_t status = iree_hal_streaming_stream_synchronize(
      (iree_hal_streaming_stream_t*)hStream);
  CUresult result = iree_status_to_cu_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuStreamDestroy(CUstream hStream) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_streaming_stream_release((iree_hal_streaming_stream_t*)hStream);
  IREE_TRACE_ZONE_END(z0);
  return CUDA_SUCCESS;
}

CUDAAPI CUresult cuStreamCopyAttributes(CUstream dst, CUstream src) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!dst || !src) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  iree_hal_streaming_stream_t* dst_stream = (iree_hal_streaming_stream_t*)dst;
  iree_hal_streaming_stream_t* src_stream = (iree_hal_streaming_stream_t*)src;

  // Copy stream attributes: flags and priority.
  // Note: We don't copy the command buffer, semaphores, or other runtime state.
  dst_stream->flags = src_stream->flags;
  dst_stream->priority = src_stream->priority;
  dst_stream->queue_affinity = src_stream->queue_affinity;

  // If source stream has capture mode settings, copy those too.
  dst_stream->capture_mode = src_stream->capture_mode;

  IREE_TRACE_ZONE_END(z0);
  return CUDA_SUCCESS;
}

//===----------------------------------------------------------------------===//
// Unified memory management
//===----------------------------------------------------------------------===//

CUDAAPI CUresult cuMemAdvise(CUdeviceptr devPtr, size_t count,
                             CUmem_advise advice, CUdevice device) {
  IREE_TRACE_ZONE_BEGIN(z0);
  // TODO: Implement unified memory advice when we have managed memory support.
  IREE_TRACE_ZONE_END(z0);
  return CUDA_SUCCESS;  // to unblock
}

CUDAAPI CUresult cuMemPrefetchAsync(CUdeviceptr devPtr, size_t count,
                                    CUdevice dstDevice, CUstream hStream) {
  IREE_TRACE_ZONE_BEGIN(z0);
  // TODO: Implement memory prefetching when we have managed memory support.
  IREE_TRACE_ZONE_END(z0);
  return CUDA_SUCCESS;  // to unblock
}

CUDAAPI CUresult cuPointerGetAttribute(void* data,
                                       CUpointer_attribute attribute,
                                       CUdeviceptr ptr) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!data) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  // Get the current context.
  iree_hal_streaming_context_t* context = iree_hal_streaming_context_current();
  if (!context) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_CONTEXT;
  }

  // Look up buffer from pointer.
  iree_hal_streaming_buffer_ref_t buffer_ref;
  iree_status_t status = iree_hal_streaming_memory_lookup(
      context, (iree_hal_streaming_deviceptr_t)ptr, &buffer_ref);
  if (!iree_status_is_ok(status)) {
    // If lookup fails, the pointer might not be a valid allocation.
    iree_status_free(status);
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  CUresult result = CUDA_SUCCESS;

  switch (attribute) {
    case CU_POINTER_ATTRIBUTE_CONTEXT: {
      // Return the context handle.
      *(CUcontext*)data = (CUcontext)buffer_ref.buffer->context;
      break;
    }
    case CU_POINTER_ATTRIBUTE_MEMORY_TYPE: {
      // Determine memory type based on buffer properties.
      CUmemorytype* memType = (CUmemorytype*)data;
      if (buffer_ref.buffer->host_ptr) {
        if (buffer_ref.buffer->memory_type == 2) {
          // Host-registered memory.
          *memType = CU_MEMORYTYPE_HOST;
        } else {
          // Host allocated memory.
          *memType = CU_MEMORYTYPE_HOST;
        }
      } else {
        // Device memory.
        *memType = CU_MEMORYTYPE_DEVICE;
      }
      break;
    }
    case CU_POINTER_ATTRIBUTE_DEVICE_POINTER: {
      // Return the device pointer.
      *(CUdeviceptr*)data =
          (CUdeviceptr)((iree_device_size_t)buffer_ref.buffer->device_ptr +
                        buffer_ref.offset);
      break;
    }
    case CU_POINTER_ATTRIBUTE_HOST_POINTER: {
      // Return the host pointer if available.
      *(void**)data = (void*)((iree_host_size_t)buffer_ref.buffer->host_ptr +
                              buffer_ref.offset);
      break;
    }
    case CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL: {
      // Return the device ordinal.
      *(int*)data = (int)context->device_ordinal;
      break;
    }
    case CU_POINTER_ATTRIBUTE_IS_MANAGED: {
      // We don't support managed memory yet.
      *(unsigned int*)data = 0;
      break;
    }
    case CU_POINTER_ATTRIBUTE_RANGE_START_ADDR: {
      // Return the base address of the allocation.
      if (buffer_ref.buffer->host_ptr) {
        *(CUdeviceptr*)data = (CUdeviceptr)buffer_ref.buffer->host_ptr;
      } else {
        *(CUdeviceptr*)data = (CUdeviceptr)buffer_ref.buffer->device_ptr;
      }
      break;
    }
    case CU_POINTER_ATTRIBUTE_RANGE_SIZE: {
      // Return the size of the allocation.
      *(size_t*)data = buffer_ref.buffer->size;
      break;
    }
    case CU_POINTER_ATTRIBUTE_MAPPED: {
      // Check if memory is mapped (host-visible).
      unsigned int is_mapped = (buffer_ref.buffer->host_ptr != NULL) ? 1 : 0;
      *(unsigned int*)data = is_mapped;
      break;
    }
    case CU_POINTER_ATTRIBUTE_SYNC_MEMOPS: {
      // Synchronous memory operations flag.
      *(unsigned int*)data = 1;  // Default to synchronous.
      break;
    }
    case CU_POINTER_ATTRIBUTE_BUFFER_ID: {
      // Return a unique buffer ID (use pointer as ID).
      *(unsigned long long*)data = (unsigned long long)buffer_ref.buffer;
      break;
    }
    case CU_POINTER_ATTRIBUTE_P2P_TOKENS:
    case CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE:
    case CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES:
    case CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE:
    case CU_POINTER_ATTRIBUTE_ACCESS_FLAGS:
    case CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE:
      // These attributes are not supported yet.
      result = CUDA_ERROR_NOT_SUPPORTED;
      break;
    default:
      result = CUDA_ERROR_INVALID_VALUE;
      break;
  }

  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuPointerSetAttribute(const void* value,
                                       CUpointer_attribute attribute,
                                       CUdeviceptr ptr) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!value) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  // Get the current context.
  iree_hal_streaming_context_t* context = iree_hal_streaming_context_current();
  if (!context) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_CONTEXT;
  }

  // Look up buffer from pointer.
  iree_hal_streaming_buffer_ref_t buffer_ref;
  iree_status_t status = iree_hal_streaming_memory_lookup(
      context, (iree_hal_streaming_deviceptr_t)ptr, &buffer_ref);

  // If lookup fails, the pointer might not be a valid allocation.
  if (!iree_status_is_ok(status)) {
    iree_status_free(status);
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  CUresult result = CUDA_SUCCESS;

  switch (attribute) {
    case CU_POINTER_ATTRIBUTE_SYNC_MEMOPS: {
      // Set synchronous memory operations flag.
      // Note: We accept the value but don't store it since all our operations
      // are currently synchronous by default.
      unsigned int sync_value = *(const unsigned int*)value;
      (void)sync_value;  // Suppress unused variable warning.
      break;
    }
    case CU_POINTER_ATTRIBUTE_ACCESS_FLAGS: {
      // Set memory access permissions (read-only, read-write, etc.).
      // Note: This is typically used for texture memory and may not be
      // applicable to our current buffer model. We accept the value but don't
      // enforce it.
      unsigned int access_flags = *(const unsigned int*)value;
      (void)access_flags;  // Suppress unused variable warning.
      break;
    }
    case CU_POINTER_ATTRIBUTE_CONTEXT:
    case CU_POINTER_ATTRIBUTE_MEMORY_TYPE:
    case CU_POINTER_ATTRIBUTE_DEVICE_POINTER:
    case CU_POINTER_ATTRIBUTE_HOST_POINTER:
    case CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL:
    case CU_POINTER_ATTRIBUTE_IS_MANAGED:
    case CU_POINTER_ATTRIBUTE_RANGE_START_ADDR:
    case CU_POINTER_ATTRIBUTE_RANGE_SIZE:
    case CU_POINTER_ATTRIBUTE_MAPPED:
    case CU_POINTER_ATTRIBUTE_BUFFER_ID:
    case CU_POINTER_ATTRIBUTE_P2P_TOKENS:
    case CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE:
    case CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES:
    case CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE:
    case CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE:
      // These attributes are read-only and cannot be set.
      result = CUDA_ERROR_NOT_SUPPORTED;
      break;
    default:
      result = CUDA_ERROR_INVALID_VALUE;
      break;
  }

  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuPointerGetAttributes(unsigned int numAttributes,
                                        CUpointer_attribute* attributes,
                                        void** data, CUdeviceptr ptr) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!attributes || !data || numAttributes == 0) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  // Query each attribute individually using cuPointerGetAttribute.
  CUresult result = CUDA_SUCCESS;
  for (unsigned int i = 0; i < numAttributes; i++) {
    CUresult attr_result = cuPointerGetAttribute(data[i], attributes[i], ptr);
    if (attr_result != CUDA_SUCCESS) {
      // Return the first error encountered.
      if (result == CUDA_SUCCESS) {
        result = attr_result;
      }
      // Continue to try other attributes even if one fails.
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuMemRangeGetAttribute(void* data, size_t dataSize,
                                        CUmem_range_attribute attribute,
                                        CUdeviceptr devPtr, size_t count) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!data || dataSize == 0) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  // Get the current context.
  iree_hal_streaming_context_t* context = iree_hal_streaming_context_current();
  if (!context) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_CONTEXT;
  }

  // Look up the buffer containing this range.
  iree_hal_streaming_buffer_ref_t buffer_ref;
  iree_status_t status = iree_hal_streaming_memory_lookup_range(
      context, devPtr, count, &buffer_ref);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  // Map the attribute and return the appropriate value.
  CUresult result = CUDA_SUCCESS;
  switch (attribute) {
    case CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY:
      // Return 1 if all pages have read-duplication enabled.
      if (dataSize < sizeof(int)) {
        result = CUDA_ERROR_INVALID_VALUE;
      } else {
        *(int*)data = buffer_ref.buffer->read_mostly_hint ? 1 : 0;
      }
      break;

    case CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION:
      // Return the preferred device ID or CU_DEVICE_CPU (-1).
      if (dataSize < sizeof(CUdevice)) {
        result = CUDA_ERROR_INVALID_VALUE;
      } else {
        *(CUdevice*)data = (CUdevice)buffer_ref.buffer->preferred_location;
      }
      break;

    case CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY:
      // Not currently supported - would require tracking device access.
      result = CUDA_ERROR_NOT_SUPPORTED;
      break;

    case CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION:
      // Return the last prefetch location.
      if (dataSize < sizeof(CUdevice)) {
        result = CUDA_ERROR_INVALID_VALUE;
      } else {
        *(CUdevice*)data = (CUdevice)buffer_ref.buffer->last_prefetch_location;
      }
      break;

    default:
      result = CUDA_ERROR_INVALID_VALUE;
      break;
  }

  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuMemRangeGetAttributes(void** data, size_t* dataSizes,
                                         CUmem_range_attribute* attributes,
                                         size_t numAttributes,
                                         CUdeviceptr devPtr, size_t count) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!data || !dataSizes || !attributes || numAttributes == 0) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  // Query each attribute individually using cuMemRangeGetAttribute.
  CUresult result = CUDA_SUCCESS;
  for (size_t i = 0; i < numAttributes; i++) {
    CUresult attr_result = cuMemRangeGetAttribute(data[i], dataSizes[i],
                                                  attributes[i], devPtr, count);
    if (attr_result != CUDA_SUCCESS) {
      // Return the first error encountered.
      if (result == CUDA_SUCCESS) {
        result = attr_result;
      }
      // Continue to try other attributes even if one fails.
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return result;
}

//===----------------------------------------------------------------------===//
// CUDA graphs
//===----------------------------------------------------------------------===//

CUDAAPI CUresult cuGraphCreate(CUgraph* phGraph, unsigned int flags) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!phGraph) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  // Get current context.
  iree_hal_streaming_context_t* context = iree_hal_streaming_context_current();
  if (!context) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_CONTEXT;
  }

  // Create graph.
  iree_hal_streaming_graph_t* graph = NULL;
  iree_status_t status = iree_hal_streaming_graph_create(
      context, IREE_HAL_STREAMING_GRAPH_FLAG_NONE, context->host_allocator,
      &graph);

  if (iree_status_is_ok(status)) {
    *phGraph = (CUgraph)graph;
  }

  CUresult result = iree_status_to_cu_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuGraphDestroy(CUgraph hGraph) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!hGraph) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  iree_hal_streaming_graph_t* graph = (iree_hal_streaming_graph_t*)hGraph;
  iree_hal_streaming_graph_release(graph);

  IREE_TRACE_ZONE_END(z0);
  return CUDA_SUCCESS;
}

CUDAAPI CUresult cuGraphInstantiate(CUgraphExec* phGraphExec, CUgraph hGraph,
                                    CUgraphNode* phErrorNode, char* logBuffer,
                                    size_t bufferSize) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!phGraphExec || !hGraph) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  iree_hal_streaming_graph_t* graph = (iree_hal_streaming_graph_t*)hGraph;
  iree_hal_streaming_graph_exec_t* exec = NULL;
  iree_status_t status = iree_hal_streaming_graph_instantiate(
      graph, IREE_HAL_STREAMING_GRAPH_INSTANTIATE_FLAG_NONE, &exec);

  if (iree_status_is_ok(status)) {
    *phGraphExec = (CUgraphExec)exec;
  } else {
    // Clear error outputs.
    if (phErrorNode) *phErrorNode = NULL;
    if (logBuffer && bufferSize > 0) {
      logBuffer[0] = '\0';
    }
  }

  CUresult result = iree_status_to_cu_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuGraphInstantiateWithFlags(CUgraphExec* phGraphExec,
                                             CUgraph hGraph,
                                             unsigned long long flags) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!phGraphExec || !hGraph) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  iree_hal_streaming_graph_t* graph = (iree_hal_streaming_graph_t*)hGraph;
  iree_hal_streaming_graph_exec_t* exec = NULL;
  iree_status_t status = iree_hal_streaming_graph_instantiate(
      graph, cuda_graph_instantiate_flags_to_internal(flags), &exec);

  if (iree_status_is_ok(status)) {
    *phGraphExec = (CUgraphExec)exec;
  }

  CUresult result = iree_status_to_cu_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuGraphExecDestroy(CUgraphExec hGraphExec) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!hGraphExec) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  iree_hal_streaming_graph_exec_t* exec =
      (iree_hal_streaming_graph_exec_t*)hGraphExec;
  iree_hal_streaming_graph_exec_release(exec);

  IREE_TRACE_ZONE_END(z0);
  return CUDA_SUCCESS;
}

CUDAAPI CUresult cuGraphLaunch(CUgraphExec hGraphExec, CUstream hStream) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!hGraphExec) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  iree_hal_streaming_graph_exec_t* exec =
      (iree_hal_streaming_graph_exec_t*)hGraphExec;
  iree_hal_streaming_stream_t* stream = (iree_hal_streaming_stream_t*)hStream;

  // Use default stream if not specified.
  if (!stream) {
    iree_hal_streaming_context_t* context =
        iree_hal_streaming_context_current();
    if (!context) {
      IREE_TRACE_ZONE_END(z0);
      return CUDA_ERROR_INVALID_CONTEXT;
    }
    stream = context->default_stream;
  }

  iree_status_t status = iree_hal_streaming_graph_exec_launch(exec, stream);

  CUresult result = iree_status_to_cu_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuGraphExecUpdate(CUgraphExec hGraphExec, CUgraph hGraph,
                                   CUgraphNode* hErrorNode_out,
                                   unsigned int flags) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!hGraphExec || !hGraph) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  iree_hal_streaming_graph_exec_t* exec =
      (iree_hal_streaming_graph_exec_t*)hGraphExec;
  iree_hal_streaming_graph_t* graph = (iree_hal_streaming_graph_t*)hGraph;

  iree_status_t status = iree_hal_streaming_graph_exec_update(exec, graph);
  if (!iree_status_is_ok(status) && hErrorNode_out) {
    *hErrorNode_out = NULL;  // We don't track specific error nodes yet.
  }

  CUresult result = iree_status_to_cu_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuGraphAddKernelNode(CUgraphNode* phGraphNode, CUgraph hGraph,
                                      const CUgraphNode* dependencies,
                                      size_t numDependencies,
                                      const void* nodeParams) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!phGraphNode || !hGraph || !nodeParams) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  iree_hal_streaming_graph_t* graph = (iree_hal_streaming_graph_t*)hGraph;
  const CUDA_KERNEL_NODE_PARAMS* params =
      (const CUDA_KERNEL_NODE_PARAMS*)nodeParams;

  // Convert dependencies.
  iree_hal_streaming_graph_node_t** deps =
      (numDependencies > 0 && dependencies)
          ? (iree_hal_streaming_graph_node_t**)dependencies
          : NULL;

  // Create dispatch params from kernel node params.
  // Extract params pointer from CUDA's parameter format.
  void* params_ptr = NULL;
  if (params->extra) {
    // Extra format: {CU_LAUNCH_PARAM_BUFFER_POINTER, &buffer,
    //                CU_LAUNCH_PARAM_BUFFER_SIZE, &size, CU_LAUNCH_PARAM_END}
    if (params->extra[0] == CU_LAUNCH_PARAM_BUFFER_POINTER) {
      params_ptr = *(void**)params->extra[1];
    }
  } else if (params->kernelParams) {
    // kernelParams is an array of pointers to the actual parameters.
    params_ptr = params->kernelParams;
  }

  iree_hal_streaming_dispatch_params_t dispatch_params = {
      .grid_dim = {params->gridDimX, params->gridDimY, params->gridDimZ},
      .block_dim = {params->blockDimX, params->blockDimY, params->blockDimZ},
      .shared_memory_bytes = params->sharedMemBytes,
      .buffer = params_ptr,
  };

  // Add kernel node to graph.
  iree_hal_streaming_graph_node_t* node = NULL;
  iree_status_t status = iree_hal_streaming_graph_add_kernel_node(
      graph, deps, numDependencies, (iree_hal_streaming_symbol_t*)params->func,
      &dispatch_params, &node);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  *phGraphNode = (CUgraphNode)node;
  IREE_TRACE_ZONE_END(z0);
  return CUDA_SUCCESS;
}

CUDAAPI CUresult cuGraphAddMemcpyNode(CUgraphNode* phGraphNode, CUgraph hGraph,
                                      const CUgraphNode* dependencies,
                                      size_t numDependencies,
                                      const void* copyParams, CUcontext ctx) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!phGraphNode || !hGraph || !copyParams) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  iree_hal_streaming_graph_t* graph = (iree_hal_streaming_graph_t*)hGraph;
  const CUDA_MEMCPY3D* params = (const CUDA_MEMCPY3D*)copyParams;

  // Convert dependencies.
  iree_hal_streaming_graph_node_t** deps =
      (numDependencies > 0 && dependencies)
          ? (iree_hal_streaming_graph_node_t**)dependencies
          : NULL;

  // For simplicity, handle basic device-to-device copy.
  // Full implementation would handle all memory types.
  iree_hal_streaming_deviceptr_t src =
      (iree_hal_streaming_deviceptr_t)params->srcDevice;
  iree_hal_streaming_deviceptr_t dst =
      (iree_hal_streaming_deviceptr_t)params->dstDevice;
  size_t size = params->WidthInBytes * params->Height * params->Depth;

  // Add memcpy node to graph.
  iree_hal_streaming_graph_node_t* node = NULL;
  iree_status_t status = iree_hal_streaming_graph_add_memcpy_node(
      graph, deps, numDependencies, dst, src, size, &node);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  *phGraphNode = (CUgraphNode)node;
  IREE_TRACE_ZONE_END(z0);
  return CUDA_SUCCESS;
}

CUDAAPI CUresult cuGraphAddMemsetNode(CUgraphNode* phGraphNode, CUgraph hGraph,
                                      const CUgraphNode* dependencies,
                                      size_t numDependencies,
                                      const void* memsetParams, CUcontext ctx) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!phGraphNode || !hGraph || !memsetParams) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  iree_hal_streaming_graph_t* graph = (iree_hal_streaming_graph_t*)hGraph;
  const CUDA_MEMSET_NODE_PARAMS* params =
      (const CUDA_MEMSET_NODE_PARAMS*)memsetParams;

  // Convert dependencies.
  iree_hal_streaming_graph_node_t** deps =
      (numDependencies > 0 && dependencies)
          ? (iree_hal_streaming_graph_node_t**)dependencies
          : NULL;

  // Add memset node to graph.
  iree_hal_streaming_graph_node_t* node = NULL;
  iree_status_t status = iree_hal_streaming_graph_add_memset_node(
      graph, deps, numDependencies, (iree_hal_streaming_deviceptr_t)params->dst,
      params->value, params->elementSize, params->width * params->height,
      &node);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  *phGraphNode = (CUgraphNode)node;
  IREE_TRACE_ZONE_END(z0);
  return CUDA_SUCCESS;
}

CUDAAPI CUresult cuGraphAddHostNode(CUgraphNode* phGraphNode, CUgraph hGraph,
                                    const CUgraphNode* dependencies,
                                    size_t numDependencies,
                                    const void* hostParams) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!phGraphNode || !hGraph || !hostParams) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  iree_hal_streaming_graph_t* graph = (iree_hal_streaming_graph_t*)hGraph;
  const CUDA_HOST_NODE_PARAMS* params =
      (const CUDA_HOST_NODE_PARAMS*)hostParams;

  // Convert dependencies.
  iree_hal_streaming_graph_node_t** deps =
      (numDependencies > 0 && dependencies)
          ? (iree_hal_streaming_graph_node_t**)dependencies
          : NULL;

  // Add host node to graph.
  iree_hal_streaming_graph_node_t* node = NULL;
  iree_status_t status = iree_hal_streaming_graph_add_host_call_node(
      graph, deps, numDependencies, (void (*)(void*))params->fn,
      params->userData, &node);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  *phGraphNode = (CUgraphNode)node;
  IREE_TRACE_ZONE_END(z0);
  return CUDA_SUCCESS;
}

CUDAAPI CUresult cuGraphAddEmptyNode(CUgraphNode* phGraphNode, CUgraph hGraph,
                                     const CUgraphNode* dependencies,
                                     size_t numDependencies) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!phGraphNode || !hGraph) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  iree_hal_streaming_graph_t* graph = (iree_hal_streaming_graph_t*)hGraph;

  // Convert dependencies.
  iree_hal_streaming_graph_node_t** deps =
      (numDependencies > 0 && dependencies)
          ? (iree_hal_streaming_graph_node_t**)dependencies
          : NULL;

  // Empty nodes are just synchronization points.
  iree_hal_streaming_graph_node_t* node = NULL;
  iree_status_t status = iree_hal_streaming_graph_add_empty_node(
      graph, deps, numDependencies, &node);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  *phGraphNode = (CUgraphNode)node;
  IREE_TRACE_ZONE_END(z0);
  return CUDA_SUCCESS;
}

//===----------------------------------------------------------------------===//
// Stream capture
//===----------------------------------------------------------------------===//

CUDAAPI CUresult cuStreamBeginCapture(CUstream hStream,
                                      CUstreamCaptureMode mode) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!hStream) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  iree_hal_streaming_stream_t* stream = (iree_hal_streaming_stream_t*)hStream;

  // Map CUDA capture mode to internal mode.
  iree_hal_streaming_capture_mode_t capture_mode;
  switch (mode) {
    case CU_STREAM_CAPTURE_MODE_GLOBAL:
      capture_mode = IREE_HAL_STREAMING_CAPTURE_MODE_GLOBAL;
      break;
    case CU_STREAM_CAPTURE_MODE_THREAD_LOCAL:
      capture_mode = IREE_HAL_STREAMING_CAPTURE_MODE_THREAD_LOCAL;
      break;
    case CU_STREAM_CAPTURE_MODE_RELAXED:
      capture_mode = IREE_HAL_STREAMING_CAPTURE_MODE_RELAXED;
      break;
    default:
      IREE_TRACE_ZONE_END(z0);
      return CUDA_ERROR_INVALID_VALUE;
  }

  iree_status_t status = iree_hal_streaming_begin_capture(stream, capture_mode);

  CUresult result = iree_status_to_cu_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuStreamEndCapture(CUstream hStream, CUgraph* phGraph) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!hStream || !phGraph) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  iree_hal_streaming_stream_t* stream = (iree_hal_streaming_stream_t*)hStream;
  iree_hal_streaming_graph_t* graph = NULL;
  iree_status_t status = iree_hal_streaming_end_capture(stream, &graph);

  if (iree_status_is_ok(status)) {
    *phGraph = (CUgraph)graph;
  }

  CUresult result = iree_status_to_cu_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuStreamIsCapturing(CUstream hStream,
                                     CUstreamCaptureStatus* captureStatus) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!captureStatus) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  if (!hStream) {
    // Use default stream.
    iree_hal_streaming_context_t* context =
        iree_hal_streaming_context_current();
    if (!context) {
      *captureStatus = CU_STREAM_CAPTURE_STATUS_NONE;
      IREE_TRACE_ZONE_END(z0);
      return CUDA_SUCCESS;
    }
    hStream = (CUstream)context->default_stream;
  }

  iree_hal_streaming_stream_t* stream = (iree_hal_streaming_stream_t*)hStream;
  bool is_capturing = false;
  iree_status_t status = iree_hal_streaming_is_capturing(stream, &is_capturing);

  if (iree_status_is_ok(status)) {
    *captureStatus = is_capturing ? CU_STREAM_CAPTURE_STATUS_ACTIVE
                                  : CU_STREAM_CAPTURE_STATUS_NONE;
  } else {
    *captureStatus = CU_STREAM_CAPTURE_STATUS_NONE;
  }

  CUresult result = iree_status_to_cu_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuStreamGetCaptureInfo(CUstream hStream,
                                        CUstreamCaptureStatus* captureStatus,
                                        unsigned long long* id) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!captureStatus) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  if (!hStream) {
    // Use default stream.
    iree_hal_streaming_context_t* context =
        iree_hal_streaming_context_current();
    if (!context) {
      *captureStatus = CU_STREAM_CAPTURE_STATUS_NONE;
      if (id) *id = 0;
      IREE_TRACE_ZONE_END(z0);
      return CUDA_SUCCESS;
    }
    hStream = (CUstream)context->default_stream;
  }

  iree_hal_streaming_stream_t* stream = (iree_hal_streaming_stream_t*)hStream;
  iree_hal_streaming_capture_status_t status_internal;
  unsigned long long capture_id;
  iree_status_t status =
      iree_hal_streaming_capture_status(stream, &status_internal, &capture_id);

  if (iree_status_is_ok(status)) {
    // Map internal status to CUDA status.
    switch (status_internal) {
      case IREE_HAL_STREAMING_CAPTURE_STATUS_NONE:
        *captureStatus = CU_STREAM_CAPTURE_STATUS_NONE;
        break;
      case IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE:
        *captureStatus = CU_STREAM_CAPTURE_STATUS_ACTIVE;
        break;
      case IREE_HAL_STREAMING_CAPTURE_STATUS_INVALIDATED:
        *captureStatus = CU_STREAM_CAPTURE_STATUS_INVALIDATED;
        break;
      default:
        *captureStatus = CU_STREAM_CAPTURE_STATUS_NONE;
        break;
    }
    if (id) {
      *id = capture_id;
    }
  } else {
    *captureStatus = CU_STREAM_CAPTURE_STATUS_NONE;
    if (id) *id = 0;
  }

  CUresult result = iree_status_to_cu_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuStreamUpdateCaptureDependencies(CUstream hStream,
                                                   CUgraphNode* dependencies,
                                                   size_t numDependencies,
                                                   unsigned int flags) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!hStream) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  iree_hal_streaming_stream_t* stream = (iree_hal_streaming_stream_t*)hStream;

  // Map CUDA flags to internal mode.
  iree_hal_streaming_capture_dependencies_mode_t mode;
  if (flags & CU_STREAM_ADD_CAPTURE_DEPENDENCIES) {
    mode = IREE_HAL_STREAMING_CAPTURE_DEPENDENCIES_ADD;
  } else if (flags & CU_STREAM_SET_CAPTURE_DEPENDENCIES) {
    mode = IREE_HAL_STREAMING_CAPTURE_DEPENDENCIES_SET;
  } else {
    // Default to SET if no specific flag.
    mode = IREE_HAL_STREAMING_CAPTURE_DEPENDENCIES_SET;
  }

  iree_status_t status = iree_hal_streaming_update_capture_dependencies(
      stream, (iree_hal_streaming_graph_node_t**)dependencies, numDependencies,
      mode);

  CUresult result = iree_status_to_cu_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

//===----------------------------------------------------------------------===//
// Memory pool management
//===----------------------------------------------------------------------===//

CUDAAPI CUresult cuMemPoolCreate(CUmemoryPool* pool,
                                 const CUmemPoolProps* poolProps) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!pool || !poolProps) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  iree_hal_streaming_context_t* context = iree_hal_streaming_context_current();
  if (!context) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_NOT_INITIALIZED;
  }

  // Convert CUDA pool props to internal props.
  iree_hal_streaming_mem_pool_props_t props = {
      .alloc_handle_type =
          cuda_mem_handle_type_to_internal(poolProps->handleTypes),
      .location_type =
          cuda_mem_location_type_to_internal(poolProps->location.type),
      .location_id = poolProps->location.id,
  };

  iree_hal_streaming_mem_pool_t* mem_pool = NULL;
  iree_status_t status = iree_hal_streaming_mem_pool_create(
      context, &props, context->host_allocator, &mem_pool);

  if (iree_status_is_ok(status)) {
    *pool = (CUmemoryPool)mem_pool;
  }

  CUresult result = iree_status_to_cu_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuMemPoolDestroy(CUmemoryPool pool) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!pool) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  iree_hal_streaming_mem_pool_release((iree_hal_streaming_mem_pool_t*)pool);
  IREE_TRACE_ZONE_END(z0);
  return CUDA_SUCCESS;
}

CUDAAPI CUresult cuMemPoolSetAttribute(CUmemoryPool pool,
                                       CUmemPool_attribute attr, void* value) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!pool || !value) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  uint64_t attr_value = 0;
  switch (attr) {
    case CU_MEMPOOL_ATTR_RELEASE_THRESHOLD:
      attr_value = *(size_t*)value;
      break;
    case CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES:
    case CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC:
    case CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES:
      attr_value = *(int*)value;
      break;
    default:
      IREE_TRACE_ZONE_END(z0);
      return CUDA_ERROR_INVALID_VALUE;
  }

  iree_hal_streaming_mem_pool_attr_t internal_attr =
      cuda_mempool_attr_to_internal(attr);
  iree_status_t status = iree_hal_streaming_mem_pool_set_attribute(
      (iree_hal_streaming_mem_pool_t*)pool, internal_attr, attr_value);

  CUresult result = iree_status_to_cu_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuMemPoolGetAttribute(CUmemoryPool pool,
                                       CUmemPool_attribute attr, void* value) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!pool || !value) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  uint64_t attr_value = 0;
  iree_hal_streaming_mem_pool_attr_t internal_attr =
      cuda_mempool_attr_to_internal(attr);
  iree_status_t status = iree_hal_streaming_mem_pool_get_attribute(
      (iree_hal_streaming_mem_pool_t*)pool, internal_attr, &attr_value);

  if (iree_status_is_ok(status)) {
    switch (attr) {
      case CU_MEMPOOL_ATTR_RELEASE_THRESHOLD:
      case CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT:
      case CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH:
      case CU_MEMPOOL_ATTR_USED_MEM_CURRENT:
      case CU_MEMPOOL_ATTR_USED_MEM_HIGH:
        *(size_t*)value = (size_t)attr_value;
        break;
      case CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES:
      case CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC:
      case CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES:
        *(int*)value = (int)attr_value;
        break;
      default:
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
        break;
    }
  }

  CUresult result = iree_status_to_cu_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuMemPoolSetAccess(CUmemoryPool pool,
                                    const CUmemAccessDesc* map, size_t count) {
  IREE_TRACE_ZONE_BEGIN(z0);
  // Not implemented yet.
  IREE_TRACE_ZONE_END(z0);
  return CUDA_ERROR_NOT_SUPPORTED;
}

CUDAAPI CUresult cuMemPoolGetAccess(CUmemAccess_flags* flags, CUmemoryPool pool,
                                    CUmemLocation* location) {
  IREE_TRACE_ZONE_BEGIN(z0);
  // Not implemented yet.
  IREE_TRACE_ZONE_END(z0);
  return CUDA_ERROR_NOT_SUPPORTED;
}

CUDAAPI CUresult cuMemPoolTrimTo(CUmemoryPool pool, size_t minBytesToKeep) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!pool) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  iree_status_t status = iree_hal_streaming_mem_pool_trim_to(
      (iree_hal_streaming_mem_pool_t*)pool, minBytesToKeep);

  CUresult result = iree_status_to_cu_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuMemPoolExportToShareableHandle(
    void* handle_out, CUmemoryPool pool, CUmemAllocationHandleType handleType,
    unsigned long long flags) {
  IREE_TRACE_ZONE_BEGIN(z0);
  // Not implemented yet - IPC support.
  IREE_TRACE_ZONE_END(z0);
  return CUDA_ERROR_NOT_SUPPORTED;
}

CUDAAPI CUresult cuMemPoolImportFromShareableHandle(
    CUmemoryPool* pool_out, void* handle, CUmemAllocationHandleType handleType,
    unsigned long long flags) {
  IREE_TRACE_ZONE_BEGIN(z0);
  // Not implemented yet - IPC support.
  IREE_TRACE_ZONE_END(z0);
  return CUDA_ERROR_NOT_SUPPORTED;
}

CUDAAPI CUresult cuMemPoolExportPointer(CUmemPoolPtrExportData* shareData_out,
                                        CUdeviceptr ptr) {
  IREE_TRACE_ZONE_BEGIN(z0);
  // Not implemented yet - IPC support.
  IREE_TRACE_ZONE_END(z0);
  return CUDA_ERROR_NOT_SUPPORTED;
}

CUDAAPI CUresult cuMemPoolImportPointer(CUdeviceptr* ptr_out, CUmemoryPool pool,
                                        CUmemPoolPtrExportData* shareData) {
  IREE_TRACE_ZONE_BEGIN(z0);
  // Not implemented yet - IPC support.
  IREE_TRACE_ZONE_END(z0);
  return CUDA_ERROR_NOT_SUPPORTED;
}

CUDAAPI CUresult cuDeviceSetMemPool(CUdevice dev, CUmemoryPool pool) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_streaming_device_t* device = iree_hal_streaming_device_entry(dev);
  if (!device) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_DEVICE;
  }

  iree_status_t status = iree_hal_streaming_device_set_mem_pool(
      device, (iree_hal_streaming_mem_pool_t*)pool);

  CUresult result = iree_status_to_cu_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuDeviceGetMemPool(CUmemoryPool* pool, CUdevice dev) {
  if (!pool) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  iree_hal_streaming_device_t* device = iree_hal_streaming_device_entry(dev);
  if (!device) {
    return CUDA_ERROR_INVALID_DEVICE;
  }

  *pool = (CUmemoryPool)iree_hal_streaming_device_mem_pool(device);

  return CUDA_SUCCESS;
}

CUDAAPI CUresult cuDeviceGetDefaultMemPool(CUmemoryPool* pool_out,
                                           CUdevice dev) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!pool_out) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  iree_hal_streaming_device_t* device = iree_hal_streaming_device_entry(dev);
  if (!device) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_DEVICE;
  }

  *pool_out = (CUmemoryPool)iree_hal_streaming_device_default_mem_pool(device);

  IREE_TRACE_ZONE_END(z0);
  return CUDA_SUCCESS;
}

CUDAAPI CUresult cuMemAllocAsync(CUdeviceptr* dptr, size_t bytesize,
                                 CUstream hStream) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!dptr) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  iree_hal_streaming_context_t* context = iree_hal_streaming_context_current();
  if (!context) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_NOT_INITIALIZED;
  }

  iree_hal_streaming_deviceptr_t device_ptr = 0;
  iree_status_t status = iree_hal_streaming_memory_allocate_async(
      context, bytesize, (iree_hal_streaming_stream_t*)hStream, &device_ptr);

  if (iree_status_is_ok(status)) {
    *dptr = (CUdeviceptr)device_ptr;
  }

  CUresult result = iree_status_to_cu_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuMemAllocFromPoolAsync(CUdeviceptr* dptr, size_t bytesize,
                                         CUmemoryPool pool, CUstream hStream) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!dptr || !pool) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_INVALID_VALUE;
  }

  iree_hal_streaming_deviceptr_t device_ptr = 0;
  iree_status_t status = iree_hal_streaming_memory_allocate_from_pool_async(
      (iree_hal_streaming_mem_pool_t*)pool, bytesize,
      (iree_hal_streaming_stream_t*)hStream, &device_ptr);

  if (iree_status_is_ok(status)) {
    *dptr = (CUdeviceptr)device_ptr;
  }

  CUresult result = iree_status_to_cu_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI CUresult cuMemFreeAsync(CUdeviceptr dptr, CUstream hStream) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_streaming_context_t* context = iree_hal_streaming_context_current();
  if (!context) {
    IREE_TRACE_ZONE_END(z0);
    return CUDA_ERROR_NOT_INITIALIZED;
  }

  iree_status_t status = iree_hal_streaming_memory_free_async(
      context, (iree_hal_streaming_deviceptr_t)dptr,
      (iree_hal_streaming_stream_t*)hStream);

  CUresult result = iree_status_to_cu_result(status);
  IREE_TRACE_ZONE_END(z0);
  return result;
}

CUDAAPI const char* cuGetErrorName(CUresult error) {
  switch (error) {
    case CUDA_SUCCESS:
      return "CUDA_SUCCESS";
    case CUDA_ERROR_INVALID_VALUE:
      return "CUDA_ERROR_INVALID_VALUE";
    case CUDA_ERROR_OUT_OF_MEMORY:
      return "CUDA_ERROR_OUT_OF_MEMORY";
    case CUDA_ERROR_NOT_INITIALIZED:
      return "CUDA_ERROR_NOT_INITIALIZED";
    case CUDA_ERROR_INVALID_DEVICE:
      return "CUDA_ERROR_INVALID_DEVICE";
    case CUDA_ERROR_INVALID_CONTEXT:
      return "CUDA_ERROR_INVALID_CONTEXT";
    case CUDA_ERROR_NOT_SUPPORTED:
      return "CUDA_ERROR_NOT_SUPPORTED";
    default:
      return "CUDA_ERROR_UNKNOWN";
  }
}

// Thread-local error state.
static iree_thread_local CUresult cuda_thread_error = CUDA_SUCCESS;

static void cuda_thread_error_set(CUresult error) { cuda_thread_error = error; }

static CUresult cuda_thread_error_get_and_clear(void) {
  CUresult error = cuda_thread_error;
  cuda_thread_error = CUDA_SUCCESS;
  return error;
}

static CUresult cuda_thread_error_peek(void) { return cuda_thread_error; }

CUDAAPI CUresult cuGetLastError(void) {
  return cuda_thread_error_get_and_clear();
}

CUDAAPI CUresult cuPeekAtLastError(void) { return cuda_thread_error_peek(); }

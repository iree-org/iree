// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/cuda2/memory_pools.h"

#include "experimental/cuda2/cuda_buffer.h"
#include "experimental/cuda2/cuda_dynamic_symbols.h"
#include "experimental/cuda2/cuda_status_util.h"
#include "iree/base/tracing.h"

// NOTE: these are currently global for all devices; we could make
// device-specific ones by malloc() and leaking (with LSAN note) unique string
// values instead.
#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_TRACKING
static const char* IREE_HAL_CUDA_DEVICE_LOCAL_POOL_RESERVED_ID =
    "CUDA pool: device-local reserved";
static const char* IREE_HAL_CUDA_OTHER_POOL_RESERVED_ID =
    "CUDA pool: other reserved";
#endif  // IREE_TRACING_FEATURE_ALLOCATION_TRACKING

static iree_status_t iree_hal_cuda2_create_memory_pool(
    const iree_hal_cuda2_dynamic_symbols_t* cuda_symbols, CUdevice cu_device,
    iree_hal_cuda2_memory_pool_params_t params,
    CUmemoryPool* IREE_RESTRICT out_pool) {
  *out_pool = NULL;

  CUmemPoolProps pool_props = {
      .allocType = CU_MEM_ALLOCATION_TYPE_PINNED,
      // TODO: allow sharing of certain pool memory types by fd/HANDLE.
      .handleTypes = CU_MEM_HANDLE_TYPE_NONE,
      .location =
          {
              .type = CU_MEM_LOCATION_TYPE_DEVICE,
              .id = cu_device,
          },
      .win32SecurityAttributes = NULL,
      .reserved = {0},
  };

  CUmemoryPool pool = NULL;
  IREE_CUDA_RETURN_IF_ERROR(cuda_symbols, cuMemPoolCreate(&pool, &pool_props),
                            "cuMemPoolCreate");

  iree_status_t status = IREE_CURESULT_TO_STATUS(
      cuda_symbols,
      cuMemPoolSetAttribute(pool, CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                            &params.release_threshold),
      "cuMemPoolSetAttribute");

  if (iree_status_is_ok(status)) {
    *out_pool = pool;
  } else {
    IREE_CUDA_IGNORE_ERROR(cuda_symbols, cuMemPoolDestroy(pool));
  }
  return status;
}

iree_status_t iree_hal_cuda2_memory_pools_initialize(
    iree_allocator_t host_allocator,
    const iree_hal_cuda2_dynamic_symbols_t* cuda_symbols, CUdevice cu_device,
    const iree_hal_cuda2_memory_pooling_params_t* pooling_params,
    iree_hal_cuda2_memory_pools_t* IREE_RESTRICT out_pools) {
  IREE_ASSERT_ARGUMENT(cuda_symbols);
  IREE_ASSERT_ARGUMENT(pooling_params);
  IREE_ASSERT_ARGUMENT(out_pools);
  IREE_TRACE_ZONE_BEGIN(z0);

  memset(out_pools, 0, sizeof(*out_pools));
  out_pools->cuda_symbols = cuda_symbols;
  out_pools->host_allocator = host_allocator;

  iree_status_t status = iree_ok_status();

  if (iree_status_is_ok(status)) {
    status = iree_hal_cuda2_create_memory_pool(cuda_symbols, cu_device,
                                               pooling_params->device_local,
                                               &out_pools->device_local);
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_cuda2_create_memory_pool(
        cuda_symbols, cu_device, pooling_params->other, &out_pools->other);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_cuda2_memory_pools_deinitialize(
    iree_hal_cuda2_memory_pools_t* pools) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (pools->device_local) {
    IREE_CUDA_IGNORE_ERROR(pools->cuda_symbols,
                           cuMemPoolDestroy(pools->device_local));
    pools->device_local = NULL;
  }

  if (pools->other) {
    IREE_CUDA_IGNORE_ERROR(pools->cuda_symbols, cuMemPoolDestroy(pools->other));
    pools->other = NULL;
  }

  IREE_TRACE_ZONE_END(z0);
}

static void iree_hal_cuda2_memory_pool_track_alloc(
    iree_hal_cuda2_memory_pools_t* pools, iree_hal_buffer_t* buffer) {
  bool is_device_local = iree_all_bits_set(iree_hal_buffer_memory_type(buffer),
                                           IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL);
  (void)is_device_local;
  iree_device_size_t allocation_size = iree_hal_buffer_allocation_size(buffer);
  (void)allocation_size;
  IREE_TRACE_ALLOC_NAMED(
      is_device_local ? IREE_HAL_CUDA_DEVICE_LOCAL_POOL_RESERVED_ID
                      : IREE_HAL_CUDA_OTHER_POOL_RESERVED_ID,
      (void*)iree_hal_cuda2_buffer_device_pointer(buffer), allocation_size);
  IREE_STATISTICS({
    iree_atomic_int64_t* bytes_allocated =
        is_device_local ? &pools->statistics.device_bytes_allocated
                        : &pools->statistics.host_bytes_allocated;
    iree_atomic_fetch_add_int64(bytes_allocated, allocation_size,
                                iree_memory_order_relaxed);
  });
}

static void iree_hal_cuda2_memory_pool_track_free(
    iree_hal_cuda2_memory_pools_t* pools, iree_hal_buffer_t* buffer) {
  bool is_device_local = iree_all_bits_set(iree_hal_buffer_memory_type(buffer),
                                           IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL);
  (void)is_device_local;
  IREE_TRACE_FREE_NAMED(is_device_local
                            ? IREE_HAL_CUDA_DEVICE_LOCAL_POOL_RESERVED_ID
                            : IREE_HAL_CUDA_OTHER_POOL_RESERVED_ID,
                        (void*)iree_hal_cuda2_buffer_device_pointer(buffer));
  IREE_STATISTICS({
    iree_atomic_int64_t* bytes_freed =
        is_device_local ? &pools->statistics.device_bytes_freed
                        : &pools->statistics.host_bytes_freed;
    iree_device_size_t allocation_size =
        iree_hal_buffer_allocation_size(buffer);
    iree_atomic_fetch_add_int64(bytes_freed, allocation_size,
                                iree_memory_order_relaxed);
  });
}

void iree_hal_cuda2_memory_pools_merge_statistics(
    iree_hal_cuda2_memory_pools_t* pools,
    iree_hal_allocator_statistics_t* statistics) {
  IREE_STATISTICS({
    statistics->device_bytes_allocated = iree_atomic_load_int64(
        &pools->statistics.device_bytes_allocated, iree_memory_order_relaxed);
    statistics->host_bytes_allocated = iree_atomic_load_int64(
        &pools->statistics.host_bytes_allocated, iree_memory_order_relaxed);
    statistics->device_bytes_freed = iree_atomic_load_int64(
        &pools->statistics.device_bytes_freed, iree_memory_order_relaxed);
    statistics->host_bytes_freed = iree_atomic_load_int64(
        &pools->statistics.host_bytes_freed, iree_memory_order_relaxed);
    if (pools->device_local) {
      cuuint64_t pool_peak = 0;
      IREE_CUDA_IGNORE_ERROR(
          pools->cuda_symbols,
          cuMemPoolGetAttribute(pools->device_local,
                                CU_MEMPOOL_ATTR_USED_MEM_HIGH, &pool_peak));
      statistics->device_bytes_peak += (iree_device_size_t)pool_peak;
    }
    if (pools->other) {
      cuuint64_t pool_peak = 0;
      IREE_CUDA_IGNORE_ERROR(
          pools->cuda_symbols,
          cuMemPoolGetAttribute(pools->other, CU_MEMPOOL_ATTR_USED_MEM_HIGH,
                                &pool_peak));
      statistics->host_bytes_peak += (iree_device_size_t)pool_peak;
    }
  });
}

iree_status_t iree_hal_cuda2_memory_pools_trim(
    iree_hal_cuda2_memory_pools_t* pools,
    const iree_hal_cuda2_memory_pooling_params_t* pooling_params) {
  IREE_CUDA_RETURN_IF_ERROR(
      pools->cuda_symbols,
      cuMemPoolTrimTo(pools->device_local,
                      pooling_params->device_local.minimum_capacity),
      "cuMemPoolTrimTo");
  IREE_CUDA_RETURN_IF_ERROR(
      pools->cuda_symbols,
      cuMemPoolTrimTo(pools->other, pooling_params->other.minimum_capacity),
      "cuMemPoolTrimTo");
  return iree_ok_status();
}

// NOTE: this is only issued if the buffer is destroyed without having had been
// scheduled for deallocation asynchronously. When a buffer is scheduled we drop
// the release callback so that this isn't called and we don't double-free.
static void iree_hal_cuda2_async_buffer_release_callback(
    void* user_data, iree_hal_buffer_t* buffer) {
  iree_hal_cuda2_memory_pools_t* pools =
      (iree_hal_cuda2_memory_pools_t*)user_data;
  IREE_TRACE_ZONE_BEGIN(z0);

  CUdeviceptr device_ptr = iree_hal_cuda2_buffer_device_pointer(buffer);
  IREE_CUDA_IGNORE_ERROR(pools->cuda_symbols, cuMemFree(device_ptr));
  iree_hal_cuda2_memory_pool_track_free(pools, buffer);

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_cuda2_memory_pools_alloca(
    iree_hal_cuda2_memory_pools_t* pools, CUstream stream,
    iree_hal_allocator_pool_t pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE(z0, (int64_t)allocation_size);

  iree_hal_buffer_params_canonicalize(&params);

  // TODO: more pools and better selection; this is coarsely deciding between
  // only device local (variables, constants, transients) and other (staging,
  // external) but could use more buffer properties (including usage/export
  // flags) to better isolate the different usage patterns and keep the pools
  // operating with reasonable limits. We should be using the |pool| arg.
  CUmemoryPool memory_pool =
      iree_all_bits_set(params.type, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL)
          ? pools->device_local
          : pools->other;

  CUdeviceptr device_ptr = 0;
  iree_status_t status = IREE_CURESULT_TO_STATUS(
      pools->cuda_symbols,
      cuMemAllocFromPoolAsync(&device_ptr, (size_t)allocation_size, memory_pool,
                              stream),
      "cuMemAllocFromPoolAsync");

  // Wrap the allocated CUDA buffer in a HAL buffer.
  // NOTE: we don't provide a device allocator because we didn't allocate from
  // one and instead we use a release callback to perform the free if the user
  // doesn't dealloca the buffer.
  iree_hal_buffer_t* buffer = NULL;
  if (iree_status_is_ok(status)) {
    iree_hal_buffer_release_callback_t release_callback = {
        .fn = iree_hal_cuda2_async_buffer_release_callback,
        .user_data = pools,
    };
    status = iree_hal_cuda2_buffer_wrap(
        /*device_allocator=*/NULL, params.type, params.access, params.usage,
        allocation_size, /*byte_offset=*/0,
        /*byte_length=*/allocation_size, IREE_HAL_CUDA_BUFFER_TYPE_ASYNC,
        device_ptr, /*host_ptr=*/NULL, release_callback, pools->host_allocator,
        &buffer);
  }

  if (iree_status_is_ok(status)) {
    // Update statistics (note that it may not yet be accurate).
    iree_hal_cuda2_memory_pool_track_alloc(pools, buffer);
    *out_buffer = buffer;
  } else if (buffer) {
    iree_hal_buffer_release(buffer);
  } else {
    IREE_CUDA_IGNORE_ERROR(pools->cuda_symbols,
                           cuMemFreeAsync(device_ptr, stream));
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_cuda2_memory_pools_dealloca(
    iree_hal_cuda2_memory_pools_t* pools, CUstream stream,
    iree_hal_buffer_t* buffer) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE(
      z0, (int64_t)iree_hal_buffer_allocation_size(buffer));

  // Try to schedule the buffer for freeing.
  CUdeviceptr device_ptr = iree_hal_cuda2_buffer_device_pointer(buffer);
  iree_status_t status = IREE_CURESULT_TO_STATUS(
      pools->cuda_symbols, cuMemFreeAsync(device_ptr, stream),
      "cuMemFreeAsync");

  // Drop the release callback so that we don't try to double-free the buffer.
  iree_hal_cuda2_buffer_drop_release_callback(buffer);

  // Update statistics (note that it may not yet be accurate).
  iree_hal_cuda2_memory_pool_track_free(pools, buffer);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

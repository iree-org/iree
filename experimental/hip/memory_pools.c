// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/hip/memory_pools.h"

#include "experimental/hip/dynamic_symbols.h"
#include "experimental/hip/hip_buffer.h"
#include "experimental/hip/status_util.h"

// NOTE: these are currently global for all devices; we could make
// device-specific ones by malloc() and leaking (with LSAN note) unique string
// values instead.
#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_TRACKING
static const char* IREE_HAL_HIP_DEVICE_LOCAL_POOL_RESERVED_ID =
    "HIP pool: device-local reserved";
static const char* IREE_HAL_HIP_OTHER_POOL_RESERVED_ID =
    "HIP pool: other reserved";
#endif  // IREE_TRACING_FEATURE_ALLOCATION_TRACKING

static iree_status_t iree_hal_hip_create_memory_pool(
    const iree_hal_hip_dynamic_symbols_t* hip_symbols, hipDevice_t hip_device,
    iree_hal_hip_memory_pool_params_t params,
    hipMemPool_t* IREE_RESTRICT out_pool) {
  *out_pool = NULL;

  hipMemPoolProps pool_props = {
      .allocType = hipMemAllocationTypePinned,
      // TODO: allow sharing of certain pool memory types by fd/HANDLE.
      .handleTypes = hipMemHandleTypeNone,
      .location =
          {
              .type = hipMemLocationTypeDevice,
              .id = hip_device,
          },
      .reserved = {0},
      .win32SecurityAttributes = NULL,
  };

  hipMemPool_t pool = NULL;

  // WARNING: hipMemPoolCreate() API is marked as beta in HIP library meaning
  // that while the feature is complete, it is still open to changes and may
  // have outstanding issues.
  IREE_HIP_RETURN_IF_ERROR(hip_symbols, hipMemPoolCreate(&pool, &pool_props),
                           "hipMemPoolCreate");

  // WARNING: hipMemPoolSetAttribute() API is marked as beta in HIP library
  // meaning that while the feature is complete, it is still open to changes and
  // may have outstanding issues.
  iree_status_t status = IREE_HIP_RESULT_TO_STATUS(
      hip_symbols,
      hipMemPoolSetAttribute(pool, hipMemPoolAttrReleaseThreshold,
                             &params.release_threshold),
      "hipMemPoolSetAttribute");

  if (iree_status_is_ok(status)) {
    *out_pool = pool;
  } else {
    // WARNING: hipMemPoolDestroy() API is marked as beta in HIP library meaning
    // that while the feature is complete, it is still open to changes and may
    // have outstanding issues.
    IREE_HIP_IGNORE_ERROR(hip_symbols, hipMemPoolDestroy(pool));
  }
  return status;
}

iree_status_t iree_hal_hip_memory_pools_initialize(
    const iree_hal_hip_dynamic_symbols_t* hip_symbols, hipDevice_t hip_device,
    const iree_hal_hip_memory_pooling_params_t* pooling_params,
    iree_allocator_t host_allocator,
    iree_hal_hip_memory_pools_t* IREE_RESTRICT out_pools) {
  IREE_ASSERT_ARGUMENT(hip_symbols);
  IREE_ASSERT_ARGUMENT(pooling_params);
  IREE_ASSERT_ARGUMENT(out_pools);
  IREE_TRACE_ZONE_BEGIN(z0);

  memset(out_pools, 0, sizeof(*out_pools));
  out_pools->hip_symbols = hip_symbols;
  out_pools->host_allocator = host_allocator;

  iree_status_t status = iree_ok_status();

  if (iree_status_is_ok(status)) {
    status = iree_hal_hip_create_memory_pool(hip_symbols, hip_device,
                                             pooling_params->device_local,
                                             &out_pools->device_local);
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_hip_create_memory_pool(
        hip_symbols, hip_device, pooling_params->other, &out_pools->other);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_hip_memory_pools_deinitialize(
    iree_hal_hip_memory_pools_t* pools) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // WARNING: hipMemPoolDestroy() API is marked as beta in HIP library meaning
  // that while the feature is complete, it is still open to changes and may
  // have outstanding issues.
  if (pools->device_local) {
    IREE_HIP_IGNORE_ERROR(pools->hip_symbols,
                          hipMemPoolDestroy(pools->device_local));
    pools->device_local = NULL;
  }

  if (pools->other) {
    IREE_HIP_IGNORE_ERROR(pools->hip_symbols, hipMemPoolDestroy(pools->other));
    pools->other = NULL;
  }

  IREE_TRACE_ZONE_END(z0);
}

static void iree_hal_hip_memory_pool_track_alloc(
    iree_hal_hip_memory_pools_t* pools, iree_hal_buffer_t* buffer) {
  bool is_device_local = iree_all_bits_set(iree_hal_buffer_memory_type(buffer),
                                           IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL);
  (void)is_device_local;
  iree_device_size_t allocation_size = iree_hal_buffer_allocation_size(buffer);
  (void)allocation_size;
  IREE_TRACE_ALLOC_NAMED(
      is_device_local ? IREE_HAL_HIP_DEVICE_LOCAL_POOL_RESERVED_ID
                      : IREE_HAL_HIP_OTHER_POOL_RESERVED_ID,
      (void*)iree_hal_hip_buffer_device_pointer(buffer), allocation_size);
  IREE_STATISTICS({
    iree_atomic_int64_t* bytes_allocated =
        is_device_local ? &pools->statistics.device_bytes_allocated
                        : &pools->statistics.host_bytes_allocated;
    iree_atomic_fetch_add_int64(bytes_allocated, allocation_size,
                                iree_memory_order_relaxed);
  });
}

static void iree_hal_hip_memory_pool_track_free(
    iree_hal_hip_memory_pools_t* pools, iree_hal_buffer_t* buffer) {
  bool is_device_local = iree_all_bits_set(iree_hal_buffer_memory_type(buffer),
                                           IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL);
  (void)is_device_local;
  IREE_TRACE_FREE_NAMED(is_device_local
                            ? IREE_HAL_HIP_DEVICE_LOCAL_POOL_RESERVED_ID
                            : IREE_HAL_HIP_OTHER_POOL_RESERVED_ID,
                        (void*)iree_hal_hip_buffer_device_pointer(buffer));
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

void iree_hal_hip_memory_pools_merge_statistics(
    iree_hal_hip_memory_pools_t* pools,
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

    // WARNING: hipMemPoolGetAttribute() API is marked as beta in HIP library
    // meaning that while the feature is complete, it is still open to changes
    // and may have outstanding issues.
    if (pools->device_local) {
      uint64_t pool_peak = 0;
      IREE_HIP_IGNORE_ERROR(
          pools->hip_symbols,
          hipMemPoolGetAttribute(pools->device_local,
                                 hipMemPoolAttrReservedMemHigh, &pool_peak));
      statistics->device_bytes_peak += (iree_device_size_t)pool_peak;
    }
    if (pools->other) {
      uint64_t pool_peak = 0;
      IREE_HIP_IGNORE_ERROR(
          pools->hip_symbols,
          hipMemPoolGetAttribute(pools->other, hipMemPoolAttrReservedMemHigh,
                                 &pool_peak));
      statistics->host_bytes_peak += (iree_device_size_t)pool_peak;
    }
  });
}

iree_status_t iree_hal_hip_memory_pools_trim(
    iree_hal_hip_memory_pools_t* pools,
    const iree_hal_hip_memory_pooling_params_t* pooling_params) {
  // WARNING: hipMemPoolTrimTo() API is marked as beta in HIP library meaning
  // that while the feature is complete, it is still open to changes and may
  // have outstanding issues.
  IREE_HIP_RETURN_IF_ERROR(
      pools->hip_symbols,
      hipMemPoolTrimTo(pools->device_local,
                       pooling_params->device_local.minimum_capacity),
      "hipMemPoolTrimTo");
  IREE_HIP_RETURN_IF_ERROR(
      pools->hip_symbols,
      hipMemPoolTrimTo(pools->other, pooling_params->other.minimum_capacity),
      "hipMemPoolTrimTo");
  return iree_ok_status();
}

// NOTE: this is only issued if the buffer is destroyed without having had been
// scheduled for deallocation asynchronously. When a buffer is scheduled we drop
// the release callback so that this isn't called and we don't double-free.
static void iree_hal_hip_async_buffer_release_callback(
    void* user_data, iree_hal_buffer_t* buffer) {
  iree_hal_hip_memory_pools_t* pools = (iree_hal_hip_memory_pools_t*)user_data;
  IREE_TRACE_ZONE_BEGIN(z0);

  hipDeviceptr_t device_ptr = iree_hal_hip_buffer_device_pointer(buffer);
  IREE_HIP_IGNORE_ERROR(pools->hip_symbols, hipFree(device_ptr));
  iree_hal_hip_memory_pool_track_free(pools, buffer);

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_hip_memory_pools_alloca(
    iree_hal_hip_memory_pools_t* pools, hipStream_t stream,
    iree_hal_allocator_pool_t pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)allocation_size);

  iree_hal_buffer_params_canonicalize(&params);

  // TODO: more pools and better selection; this is coarsely deciding between
  // only device local (variables, constants, transients) and other (staging,
  // external) but could use more buffer properties (including usage/export
  // flags) to better isolate the different usage patterns and keep the pools
  // operating with reasonable limits. We should be using the |pool| arg.
  hipMemPool_t memory_pool =
      iree_all_bits_set(params.type, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL)
          ? pools->device_local
          : pools->other;

  hipDeviceptr_t device_ptr = 0;
  // WARNING: hipMallocFromPoolAsync() API is marked as beta in HIP library
  // meaning that while the feature is complete, it is still open to changes and
  // may have outstanding issues.
  iree_status_t status = IREE_HIP_RESULT_TO_STATUS(
      pools->hip_symbols,
      hipMallocFromPoolAsync(&device_ptr, (size_t)allocation_size, memory_pool,
                             stream),
      "hipMallocFromPoolAsync");

  // Wrap the allocated HIP buffer in a HAL buffer.
  // NOTE: we don't provide a device allocator because we didn't allocate from
  // one and instead we use a release callback to perform the free if the user
  // doesn't dealloca the buffer.
  iree_hal_buffer_t* buffer = NULL;
  if (iree_status_is_ok(status)) {
    iree_hal_buffer_release_callback_t release_callback = {
        .fn = iree_hal_hip_async_buffer_release_callback,
        .user_data = pools,
    };
    status = iree_hal_hip_buffer_wrap(
        /*device_allocator=*/NULL, params.type, params.access, params.usage,
        allocation_size, /*byte_offset=*/0,
        /*byte_length=*/allocation_size, IREE_HAL_HIP_BUFFER_TYPE_ASYNC,
        device_ptr, /*host_ptr=*/NULL, release_callback, pools->host_allocator,
        &buffer);
  }

  if (iree_status_is_ok(status)) {
    // Update statistics (note that it may not yet be accurate).
    iree_hal_hip_memory_pool_track_alloc(pools, buffer);
    *out_buffer = buffer;
  } else if (buffer) {
    iree_hal_buffer_release(buffer);
  } else {
    IREE_HIP_IGNORE_ERROR(pools->hip_symbols, hipFreeAsync(device_ptr, stream));
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_hip_memory_pools_dealloca(
    iree_hal_hip_memory_pools_t* pools, hipStream_t stream,
    iree_hal_buffer_t* buffer) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(
      z0, (int64_t)iree_hal_buffer_allocation_size(buffer));

  // Only process the request if the buffer came from an async pool.
  // We may get requests for deallocations on ones that didn't if one part of
  // the application allocated the buffer synchronously and another deallocated
  // it asynchronously.
  iree_status_t status = iree_ok_status();
  if (iree_hal_hip_buffer_type(buffer) == IREE_HAL_HIP_BUFFER_TYPE_ASYNC) {
    // Try to schedule the buffer for freeing.
    hipDeviceptr_t device_ptr = iree_hal_hip_buffer_device_pointer(buffer);
    status = IREE_HIP_RESULT_TO_STATUS(
        pools->hip_symbols, hipFreeAsync(device_ptr, stream), "hipFreeAsync");
    if (iree_status_is_ok(status)) {
      // Drop the release callback so that we don't try to double-free the
      // buffer. Note that we only do this if the HIP free succeeded as
      // otherwise we still need to synchronously deallocate the buffer when it
      // is destroyed.
      iree_hal_hip_buffer_drop_release_callback(buffer);

      // Update statistics (note that it may not yet be accurate).
      iree_hal_hip_memory_pool_track_free(pools, buffer);
    }
  } else {
    // Not allocated via alloca, ignore.
    IREE_TRACE_ZONE_APPEND_TEXT(z0, "ignored sync allocation");
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

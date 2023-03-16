// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/utils/caching_allocator.h"

#include "iree/base/internal/synchronization.h"
#include "iree/base/tracing.h"

// Default capacity of a pool free list when not specified by the user.
#define IREE_HAL_CACHING_ALLOCATOR_DEFAULT_FREE_LIST_CAPACITY 64

//===----------------------------------------------------------------------===//
// iree_hal_caching_allocator_pool_t
//===----------------------------------------------------------------------===//

IREE_TRACE(
    static const char* IREE_HAL_CACHING_ALLOCATOR_ID = "Free Cached Memory");

void iree_hal_caching_allocator_pool_params_initialize(
    iree_hal_allocator_memory_heap_t heap,
    iree_hal_caching_allocator_pool_params_t* out_params) {
  IREE_ASSERT_ARGUMENT(out_params);
  memset(out_params, 0, sizeof(*out_params));
  out_params->heap = heap;
  out_params->max_allocation_size = heap.max_allocation_size;
  out_params->max_allocation_capacity = IREE_DEVICE_SIZE_MAX;
  out_params->max_free_allocation_count =
      IREE_HAL_CACHING_ALLOCATOR_DEFAULT_FREE_LIST_CAPACITY;
}

// Pool of arbitrarily-sized device allocations for a particular heap.
// This maintains a free list of blocks available for use but does not track
// outstanding allocations.
//
// Thread-safe. Pools can service requests from multiple threads concurrently by
// way of a pool-specific mutex. The mutex will not be held during underlying
// allocator operations such as when acquiring a new allocation as these can be
// extremely slow and the underlying allocator is also assumed thread-safe.
typedef iree_alignas(
    iree_max_align_t) struct iree_hal_caching_allocator_pool_t {
  // Defines which heap this pool allocates from and the pool limits.
  iree_hal_caching_allocator_pool_params_t params;

  // Underlying device allocator used to allocate storage buffers.
  // Unretained as the parent allocator retains it for us.
  iree_hal_allocator_t* device_allocator;

  // Guards access to the pool data structures as buffers can be
  // acquired/released from multiple threads if shared across user-visible
  // devices.
  //
  // Note that we keep the mutex per-pool so that if we do need to allocate or
  // free we can do so without holding the lock.
  iree_slim_mutex_t mutex;

  // Total size, in bytes, of all outstanding allocations made from this pool.
  // This only includes allocations we are able to pool as we otherwise cannot
  // observe imported/exported buffers.
  iree_device_size_t total_allocated_size;

  // Total size, in bytes, of all free buffers currently in this pool.
  iree_device_size_t free_allocated_size;

  // Flat MRU list of available buffers with max_free_allocation_count slots.
  // Sorted by ascending recency (the higher the index the more recent).
  // If we really cared about optimizing the interior removal then we'd want
  // a linked/skip list or some bucketing but that's really for the higher
  // level allocators to do.
  iree_host_size_t free_count;
  iree_hal_buffer_t* free_buffers[];
} iree_hal_caching_allocator_pool_t;

static void iree_hal_caching_allocator_pool_trim(
    iree_hal_caching_allocator_pool_t* pool);

// Initializes a buffer pool in |out_pool|.
// Buffer device storage will be allocated from |device_allocator|.
static void iree_hal_caching_allocator_pool_initialize(
    iree_hal_caching_allocator_pool_params_t params,
    iree_hal_allocator_t* device_allocator,
    iree_hal_caching_allocator_pool_t* out_pool) {
  IREE_TRACE_ZONE_BEGIN(z0);

  out_pool->params = params;
  out_pool->device_allocator = device_allocator;
  iree_slim_mutex_initialize(&out_pool->mutex);
  out_pool->total_allocated_size = 0;
  out_pool->free_allocated_size = 0;
  out_pool->free_count = 0;

  IREE_TRACE_SET_PLOT_TYPE(IREE_HAL_CACHING_ALLOCATOR_ID,
                           IREE_TRACING_PLOT_TYPE_MEMORY, /*step=*/true,
                           /*fill=*/true, /*color=*/0);
  IREE_TRACE_PLOT_VALUE_I64(IREE_HAL_CACHING_ALLOCATOR_ID,
                            out_pool->free_allocated_size);

  IREE_TRACE_ZONE_END(z0);
}

// Deinitializes |pool|; all allocated buffers must have been released.
static void iree_hal_caching_allocator_pool_deinitialize(
    iree_hal_caching_allocator_pool_t* pool) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Trim first to release all the buffers. There shouldn't be any live
  // allocations by the time we are deinitializing.
  iree_hal_caching_allocator_pool_trim(pool);
  IREE_ASSERT_EQ(pool->total_allocated_size, 0,
                 "must have released all allocations prior to deinit");
  IREE_ASSERT_EQ(pool->free_allocated_size, 0,
                 "must have released all allocations prior to deinit");
  IREE_ASSERT_EQ(pool->free_count, 0,
                 "must have released all allocations prior to deinit");

  iree_slim_mutex_deinitialize(&pool->mutex);

  IREE_TRACE_ZONE_END(z0);
}

// Pushes |buffer| on to the pool free list as the most recently used.
// The buffer will be retained in the list.
//
// Must be called with the pool mutex held.
static void iree_hal_caching_allocator_pool_push_buffer(
    iree_hal_caching_allocator_pool_t* pool, iree_hal_buffer_t* buffer) {
  // Retain the buffer; the caller must release it to complete the ownership
  // transfer.
  iree_hal_buffer_retain(buffer);

  IREE_ASSERT_LT(pool->free_count, pool->params.max_free_allocation_count);

  // Add to the end of the list (the most recent).
  iree_host_size_t i = pool->free_count++;
  pool->free_buffers[i] = buffer;

  // Track that we're now retaining unused memory.
  pool->free_allocated_size += buffer->allocation_size;
  IREE_TRACE_PLOT_VALUE_I64(IREE_HAL_CACHING_ALLOCATOR_ID,
                            pool->free_allocated_size);
}

// Takes the buffer in the |pool| free list at index |i| and returns ownership.
// If the list is large this is bad but in most cases it's just a few dozen
// elements.
//
// Must be called with the pool mutex held.
static iree_hal_buffer_t* iree_hal_caching_allocator_pool_take_buffer_at(
    iree_hal_caching_allocator_pool_t* pool, iree_host_size_t i) {
  iree_hal_buffer_t* buffer = pool->free_buffers[i];
  if (i < pool->free_count - 1) {
    // Shift the list down to keep it dense and in ascending recency order.
    memmove(&pool->free_buffers[i], &pool->free_buffers[i + 1],
            (pool->free_count - i - 1) * sizeof(pool->free_buffers[0]));
  }
  --pool->free_count;
  pool->free_allocated_size -= buffer->allocation_size;
  IREE_TRACE_PLOT_VALUE_I64(IREE_HAL_CACHING_ALLOCATOR_ID,
                            pool->free_allocated_size);
  return buffer;
}

// Scans the |pool| free list for a buffer matching the given requirements and
// returns ownership.
//
// Must be called with the pool mutex held.
static iree_hal_buffer_t* iree_hal_caching_allocator_pool_find_and_take_buffer(
    iree_hal_caching_allocator_pool_t* pool,
    const iree_hal_buffer_params_t* params,
    iree_device_size_t allocation_size) {
  // Walk backwards so that we check the most recently released buffers first.
  for (int i = (int)pool->free_count - 1; i >= 0; --i) {
    // NOTE: we are not currently checking alignment as we don't really have it.
    // We assume programs will use consistent alignments for a particular heap
    // (as the heap has a min alignment).
    iree_hal_buffer_t* buffer = pool->free_buffers[i];
    if (iree_all_bits_set(iree_hal_buffer_memory_type(buffer), params->type) &&
        iree_all_bits_set(iree_hal_buffer_allowed_usage(buffer),
                          params->usage) &&
        iree_hal_buffer_allocation_size(buffer) == allocation_size) {
      return iree_hal_caching_allocator_pool_take_buffer_at(pool, i);
    }
  }
  return NULL;  // nothing found
}

// Trims |pool| down to at most |target_size| of available allocations.
// The oldest allocations will be trimmed first.
//
// Thread-safe; multiple threads may concurrently access the |pool|.
static void iree_hal_caching_allocator_pool_trim_to_size(
    iree_hal_caching_allocator_pool_t* pool, iree_device_size_t target_size) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE(z0, (int64_t)target_size);

  iree_slim_mutex_lock(&pool->mutex);

  while (pool->free_count > 0 && pool->total_allocated_size > target_size) {
    // Take the oldest buffer in the list.
    iree_hal_buffer_t* dead_buffer =
        iree_hal_caching_allocator_pool_take_buffer_at(pool,
                                                       pool->free_count - 1);

    // NOTE: we've removed the buffer but have not subtracted the size from
    // the total yet - we want to do that only after releasing the buffer.
    // If we didn't it's possible for another thread to start an allocation
    // thinking that we've already released the buffer.
    iree_device_size_t allocation_size =
        iree_hal_buffer_allocation_size(dead_buffer);

    // Release the buffer without holding the lock as deallocation can be slow.
    iree_slim_mutex_unlock(&pool->mutex);
    iree_hal_allocator_deallocate_buffer(pool->device_allocator, dead_buffer);
    iree_slim_mutex_lock(&pool->mutex);

    // Update accounting to represent that we've released the buffer.
    IREE_ASSERT_GE(pool->total_allocated_size, allocation_size);
    pool->total_allocated_size -= allocation_size;
  }

  iree_slim_mutex_unlock(&pool->mutex);

  IREE_TRACE_ZONE_END(z0);
}

// Releases all unused buffers in |pool| to the underlying device allocator.
//
// The pool mutex must not be held by the caller.
static void iree_hal_caching_allocator_pool_trim(
    iree_hal_caching_allocator_pool_t* pool) {
  iree_hal_caching_allocator_pool_trim_to_size(pool, 0);
}

// Acquires a buffer of |allocation_size| from the |pool|.
// The buffer will have a memory type and usage compatible with the given types.
// Fails if the pool is empty and the underlying device fails the allocation.
//
// Thread-safe; multiple threads may concurrently access the |pool|.
static iree_status_t iree_hal_caching_allocator_pool_acquire(
    iree_hal_caching_allocator_pool_t* pool,
    const iree_hal_buffer_params_t* params, iree_device_size_t allocation_size,
    iree_const_byte_span_t initial_data, iree_hal_buffer_t** out_buffer) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE(z0, (int64_t)allocation_size);

  // Scan the free list to find an appropriate block.
  // If found we pop it off the list and return it without needing to allocate.
  iree_slim_mutex_lock(&pool->mutex);
  iree_hal_buffer_t* existing_buffer =
      iree_hal_caching_allocator_pool_find_and_take_buffer(pool, params,
                                                           allocation_size);
  if (!existing_buffer) {
    // We'll need to allocate so we add the size such that it'll be accounted
    // for by other threads allocating at the same time.
    pool->total_allocated_size += allocation_size;
  }
  iree_slim_mutex_unlock(&pool->mutex);
  if (existing_buffer) {
    // Found a buffer - return it after writing in initial_data (if any).
    // We can only do this if the buffer supports mapping and expect unmappable
    // buffers to have been filtered out earlier up.
    iree_status_t status = iree_ok_status();
    if (!iree_const_byte_span_is_empty(initial_data)) {
      status = iree_hal_buffer_map_write(existing_buffer, 0, initial_data.data,
                                         initial_data.data_length);
    }
    if (iree_status_is_ok(status)) {
      // Return the initialized buffer.
      *out_buffer = existing_buffer;
    } else {
      // Release the buffer back to the pool.
      iree_hal_buffer_release(existing_buffer);
    }
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Trim first before allocating so that we don't go over peak.
  iree_hal_caching_allocator_pool_trim_to_size(
      pool, pool->params.max_allocation_capacity);

  // No existing buffer was found that could be used and we'll need to allocate
  // one. Note that we do this without holding the lock as the underlying
  // device allocator can be very slow. It's possible for buffers to be released
  // to the pool by another thread while we're allocating here but that's OK.
  iree_hal_buffer_t* buffer = NULL;
  iree_status_t status = iree_hal_allocator_allocate_buffer(
      pool->device_allocator, *params, allocation_size, initial_data, &buffer);

  // If the allocation failed then remove the size from the total.
  if (iree_status_is_ok(status)) {
    *out_buffer = buffer;
  } else {
    if (buffer) iree_hal_buffer_release(buffer);
    iree_slim_mutex_lock(&pool->mutex);
    pool->total_allocated_size -= allocation_size;
    iree_slim_mutex_unlock(&pool->mutex);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Releases a |buffer| to the |pool| if there is capacity remaining.
//
// Thread-safe; multiple threads may concurrently access the |pool|.
static void iree_hal_caching_allocator_pool_release(
    iree_hal_caching_allocator_pool_t* pool, iree_hal_buffer_t* buffer) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE(
      z0, (int64_t)iree_hal_buffer_allocation_size(buffer));

  // Try to add the buffer to the pool. If the pool is at capacity we'll just
  // release it back to the allocator.
  iree_slim_mutex_lock(&pool->mutex);

  const iree_device_size_t allocation_size =
      iree_hal_buffer_allocation_size(buffer);
  const bool under_capacity = pool->total_allocated_size - allocation_size <=
                              pool->params.max_allocation_capacity;
  const bool under_count =
      pool->free_count + 1 <= pool->params.max_free_allocation_count;
  if (under_capacity && under_count) {
    iree_hal_caching_allocator_pool_push_buffer(pool, buffer);
    buffer = NULL;
  }

  // If the buffer didn't fit in the pool we drop it here while we don't hold
  // the lock as deallocations can be very expensive.
  if (buffer) {
    iree_slim_mutex_unlock(&pool->mutex);
    iree_hal_allocator_deallocate_buffer(pool->device_allocator, buffer);
    iree_slim_mutex_lock(&pool->mutex);
    pool->total_allocated_size -= allocation_size;
  }

  iree_slim_mutex_unlock(&pool->mutex);

  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// iree_hal_caching_allocator_t
//===----------------------------------------------------------------------===//

struct iree_hal_caching_allocator_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;

  // Underlying device allocator used to allocate storage blocks.
  // We also route down to it for things we don't support (import/export/etc).
  iree_hal_allocator_t* device_allocator;

  // Total number of pools.
  iree_host_size_t pool_count;

  // Pointers to pool storage.
  // The count and layout of pools is immutable while each pool has a mutex to
  // guard the pool state.
  iree_hal_caching_allocator_pool_t* pools[];
};

static const iree_hal_allocator_vtable_t iree_hal_caching_allocator_vtable;

iree_hal_caching_allocator_t* iree_hal_caching_allocator_cast(
    iree_hal_allocator_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_caching_allocator_vtable);
  return (iree_hal_caching_allocator_t*)base_value;
}

iree_status_t iree_hal_caching_allocator_create_unbounded(
    iree_hal_allocator_t* device_allocator, iree_allocator_t host_allocator,
    iree_hal_allocator_t** out_allocator) {
  IREE_ASSERT_ARGUMENT(device_allocator);
  IREE_ASSERT_ARGUMENT(out_allocator);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Query heaps from the underlying allocator.
  iree_hal_allocator_memory_heap_t heaps[8];
  iree_host_size_t heap_count = 0;
  IREE_RETURN_IF_ERROR(iree_hal_allocator_query_memory_heaps(
      device_allocator, IREE_ARRAYSIZE(heaps), heaps, &heap_count));

  // Setup pool parameters for each heap.
  iree_hal_caching_allocator_pool_params_t* heap_pool_params =
      (iree_hal_caching_allocator_pool_params_t*)iree_alloca(
          sizeof(iree_hal_caching_allocator_pool_params_t) * heap_count);
  for (iree_host_size_t i = 0; i < heap_count; ++i) {
    iree_hal_caching_allocator_pool_params_initialize(heaps[i],
                                                      &heap_pool_params[i]);
  }

  iree_status_t status = iree_hal_caching_allocator_create_with_pools(
      heap_count, heap_pool_params, device_allocator, host_allocator,
      out_allocator);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_caching_allocator_create_with_pools(
    iree_host_size_t pool_count,
    const iree_hal_caching_allocator_pool_params_t* pool_params,
    iree_hal_allocator_t* device_allocator, iree_allocator_t host_allocator,
    iree_hal_allocator_t** out_allocator) {
  IREE_ASSERT_ARGUMENT(!pool_count || pool_params);
  IREE_ASSERT_ARGUMENT(device_allocator);
  IREE_ASSERT_ARGUMENT(out_allocator);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Allocate the allocator itself and then a trailing list of variable-length
  // pools based on their free list sizes.
  iree_hal_caching_allocator_t* allocator = NULL;
  iree_host_size_t pool_list_size = pool_count * sizeof(allocator->pools[0]);
  iree_host_size_t total_size = iree_host_align(
      iree_sizeof_struct(*allocator) + pool_list_size, iree_max_align_t);
  iree_host_size_t pool_offset = total_size;
  for (iree_host_size_t i = 0; i < pool_count; ++i) {
    iree_hal_caching_allocator_pool_t* pool = NULL;
    total_size += iree_host_align(
        sizeof(*pool) + sizeof(pool->free_buffers[0]) *
                            pool_params[i].max_free_allocation_count,
        iree_max_align_t);
  }
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, total_size, (void**)&allocator));

  // Initialize the allocator.
  iree_hal_resource_initialize(&iree_hal_caching_allocator_vtable,
                               &allocator->resource);
  allocator->host_allocator = host_allocator;
  allocator->device_allocator = device_allocator;
  iree_hal_allocator_retain(allocator->device_allocator);
  allocator->pool_count = pool_count;

  // Initialize each pool.
  uint8_t* pool_ptr = (uint8_t*)allocator + pool_offset;
  for (iree_host_size_t i = 0; i < pool_count; ++i) {
    iree_hal_caching_allocator_pool_t* pool =
        (iree_hal_caching_allocator_pool_t*)pool_ptr;
    pool_ptr += iree_host_align(
        sizeof(*pool) + sizeof(pool->free_buffers[0]) *
                            pool_params[i].max_free_allocation_count,
        iree_max_align_t);
    allocator->pools[i] = pool;
    iree_hal_caching_allocator_pool_initialize(pool_params[i], device_allocator,
                                               pool);
  }

  *out_allocator = (iree_hal_allocator_t*)allocator;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// Selects a heap from |heaps| matching the given |heap_key|.
// Fails if no heap matches the given key. Optionally a buffer usage bitfield
// can be provided. Wildcards can be used with either to match the first heap
// that meets either requirement.
//
// Examples:
//   *: first heap
//   *;transfer: first heap with transfer usage
//   device_local: first heap with the IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL bit
//   device_local|host_visible
//   device_local;transfer|dispatch_storage
static iree_status_t iree_hal_select_heap(
    iree_string_view_t heap_key, iree_host_size_t heap_count,
    const iree_hal_allocator_memory_heap_t* heaps,
    const iree_hal_allocator_memory_heap_t** out_heap) {
  iree_string_view_t memory_type_str = iree_string_view_empty();
  iree_string_view_t buffer_usage_str = iree_string_view_empty();
  iree_string_view_split(heap_key, ';', &memory_type_str, &buffer_usage_str);

  // Parse the provided filters, if any.
  iree_hal_memory_type_t memory_type = IREE_HAL_MEMORY_TYPE_NONE;
  iree_hal_buffer_usage_t buffer_usage = IREE_HAL_BUFFER_USAGE_NONE;
  if (!iree_string_view_is_empty(memory_type_str) &&
      !iree_string_view_equal(memory_type_str, IREE_SV("*"))) {
    IREE_RETURN_IF_ERROR(
        iree_hal_memory_type_parse(memory_type_str, &memory_type));
  }
  if (!iree_string_view_is_empty(buffer_usage_str) &&
      !iree_string_view_equal(buffer_usage_str, IREE_SV("*"))) {
    IREE_RETURN_IF_ERROR(
        iree_hal_buffer_usage_parse(buffer_usage_str, &buffer_usage));
  }

  // Return the first heap satisfying all filters.
  for (iree_host_size_t i = 0; i < heap_count; ++i) {
    if ((!memory_type || iree_all_bits_set(heaps[i].type, memory_type)) &&
        (!buffer_usage ||
         iree_all_bits_set(heaps[i].allowed_usage, buffer_usage))) {
      *out_heap = &heaps[i];
      return iree_ok_status();
    }
  }

  // No matching heap found; can happen if the device doesn't have the kind of
  // heaps the user was expecting with the configuration.
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "no heap matching requested config params "
                          "memory_type='%.*s', buffer_usage='%.*s'",
                          (int)memory_type_str.size, memory_type_str.data,
                          (int)buffer_usage_str.size, buffer_usage_str.data);
}

iree_status_t iree_hal_caching_allocator_create_from_spec(
    iree_string_view_t config_pairs, iree_hal_allocator_t* device_allocator,
    iree_allocator_t host_allocator, iree_hal_allocator_t** out_allocator) {
  if (iree_string_view_is_empty(config_pairs)) {
    // No parameters implies unbounded; we'll hang on to all memory forever.
    // This is only useful in very specific usage patterns such as statically
    // shaped and deterministic benchmarks that always allocate the same amounts
    // of memory per invocation.
    return iree_hal_caching_allocator_create_unbounded(
        device_allocator, host_allocator, out_allocator);
  }

  // Query all heaps from the base allocator. We'll use this list to match the
  // user-provided pool parameters to heaps. It's likely that not all heaps
  // will be selected by the user.
  iree_host_size_t heap_count = 0;
  iree_hal_allocator_memory_heap_t heaps[16];
  IREE_RETURN_IF_ERROR(iree_hal_allocator_query_memory_heaps(
      device_allocator, IREE_ARRAYSIZE(heaps), heaps, &heap_count));

  // Build a list of pools based on user specification.
  iree_host_size_t pool_count = 0;
  iree_hal_caching_allocator_pool_params_t pool_params_storage[16];
  do {
    if (pool_count + 1 > IREE_ARRAYSIZE(pool_params_storage)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "too many pools specified");
    }

    // Pop the key=value config pair from the list.
    iree_string_view_t config_pair = iree_string_view_empty();
    iree_string_view_split(config_pairs, ',', &config_pair, &config_pairs);
    iree_string_view_t heap_key = iree_string_view_empty();
    iree_string_view_t pool_config = iree_string_view_empty();
    iree_string_view_split(config_pair, '=', &heap_key, &pool_config);
    heap_key = iree_string_view_trim(heap_key);
    if (iree_string_view_is_empty(heap_key)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "heap key must specified in pool params");
    }

    // Select the heap based on the key.
    const iree_hal_allocator_memory_heap_t* heap = NULL;
    IREE_RETURN_IF_ERROR(
        iree_hal_select_heap(heap_key, heap_count, heaps, &heap));

    // Configure the pool based on the provided parameters.
    iree_hal_caching_allocator_pool_params_t* pool_params =
        &pool_params_storage[pool_count++];
    iree_hal_caching_allocator_pool_params_initialize(*heap, pool_params);

    iree_string_view_t max_allocation_size_str = iree_string_view_empty();
    iree_string_view_t max_allocation_capacity_str = iree_string_view_empty();
    iree_string_view_t max_free_allocation_count_str = iree_string_view_empty();
    iree_string_view_split(pool_config, ';', &max_allocation_size_str,
                           &pool_config);
    iree_string_view_split(pool_config, ';', &max_allocation_capacity_str,
                           &pool_config);
    iree_string_view_split(pool_config, ';', &max_free_allocation_count_str,
                           &pool_config);
    max_allocation_size_str = iree_string_view_trim(max_allocation_size_str);
    if (!iree_string_view_is_empty(max_allocation_size_str) &&
        !iree_string_view_equal(max_allocation_size_str, IREE_SV("*"))) {
      IREE_RETURN_IF_ERROR(
          iree_string_view_parse_device_size(max_allocation_size_str,
                                             &pool_params->max_allocation_size),
          "parsing max_allocation_size");
    }
    max_allocation_capacity_str =
        iree_string_view_trim(max_allocation_capacity_str);
    if (!iree_string_view_is_empty(max_allocation_capacity_str) &&
        !iree_string_view_equal(max_allocation_capacity_str, IREE_SV("*"))) {
      IREE_RETURN_IF_ERROR(iree_string_view_parse_device_size(
                               max_allocation_capacity_str,
                               &pool_params->max_allocation_capacity),
                           "parsing max_allocation_capacity");
    }
    max_free_allocation_count_str =
        iree_string_view_trim(max_free_allocation_count_str);
    if (!iree_string_view_is_empty(max_free_allocation_count_str) &&
        !iree_string_view_equal(max_free_allocation_count_str, IREE_SV("*"))) {
      uint32_t max_free_allocation_count = 0;
      if (!iree_string_view_atoi_uint32(max_free_allocation_count_str,
                                        &max_free_allocation_count)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "invalid count '%.*s'",
                                (int)max_free_allocation_count_str.size,
                                max_free_allocation_count_str.data);
      }
      pool_params->max_free_allocation_count = max_free_allocation_count;
    }
  } while (!iree_string_view_is_empty(config_pairs));
  return iree_hal_caching_allocator_create_with_pools(
      pool_count, pool_params_storage, device_allocator, host_allocator,
      out_allocator);
}

static void iree_hal_caching_allocator_destroy(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  iree_hal_caching_allocator_t* allocator =
      iree_hal_caching_allocator_cast(base_allocator);
  iree_allocator_t host_allocator = allocator->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Deinitialize each pool, returning any available resources to the underlying
  // device allocator.
  for (iree_host_size_t i = 0; i < allocator->pool_count; ++i) {
    iree_hal_caching_allocator_pool_deinitialize(allocator->pools[i]);
  }

  iree_hal_allocator_release(allocator->device_allocator);
  iree_allocator_free(host_allocator, allocator);

  IREE_TRACE_ZONE_END(z0);
}

static iree_allocator_t iree_hal_caching_allocator_host_allocator(
    const iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  iree_hal_caching_allocator_t* allocator =
      (iree_hal_caching_allocator_t*)base_allocator;
  return allocator->host_allocator;
}

static iree_status_t iree_hal_caching_allocator_trim(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  iree_hal_caching_allocator_t* allocator =
      iree_hal_caching_allocator_cast(base_allocator);
  for (iree_host_size_t i = 0; i < allocator->pool_count; ++i) {
    iree_hal_caching_allocator_pool_trim(allocator->pools[i]);
  }
  return iree_hal_allocator_trim(allocator->device_allocator);
}

static void iree_hal_caching_allocator_query_statistics(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_allocator_statistics_t* IREE_RESTRICT out_statistics) {
  iree_hal_caching_allocator_t* allocator =
      iree_hal_caching_allocator_cast(base_allocator);
  iree_hal_allocator_query_statistics(allocator->device_allocator,
                                      out_statistics);
}

static iree_status_t iree_hal_caching_allocator_query_memory_heaps(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_host_size_t capacity,
    iree_hal_allocator_memory_heap_t* IREE_RESTRICT heaps,
    iree_host_size_t* IREE_RESTRICT out_count) {
  // We could expose just the heaps backing our pools but that would prevent the
  // use of any underlying heaps that we aren't pooling.
  iree_hal_caching_allocator_t* allocator =
      iree_hal_caching_allocator_cast(base_allocator);
  return iree_hal_allocator_query_memory_heaps(allocator->device_allocator,
                                               capacity, heaps, out_count);
}

static iree_hal_buffer_compatibility_t
iree_hal_caching_allocator_query_buffer_compatibility(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_device_size_t* IREE_RESTRICT allocation_size) {
  // Defer to the base allocator.
  iree_hal_caching_allocator_t* allocator =
      iree_hal_caching_allocator_cast(base_allocator);
  return iree_hal_allocator_query_buffer_compatibility(
      allocator->device_allocator, *params, *allocation_size, params,
      allocation_size);
}

static iree_hal_caching_allocator_pool_t* iree_hal_caching_allocator_find_pool(
    iree_hal_caching_allocator_t* allocator, iree_hal_memory_type_t type,
    iree_hal_buffer_usage_t allowed_usage) {
  // Scan in order; the preferred pools are first.
  for (iree_host_size_t i = 0; i < allocator->pool_count; ++i) {
    iree_hal_allocator_memory_heap_t heap = allocator->pools[i]->params.heap;
    if (iree_all_bits_set(heap.type, type) &&
        iree_all_bits_set(heap.allowed_usage, allowed_usage)) {
      return allocator->pools[i];
    }
  }
  return NULL;
}

static iree_status_t iree_hal_caching_allocator_allocate_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    const iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_device_size_t allocation_size, iree_const_byte_span_t initial_data,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  iree_hal_caching_allocator_t* allocator =
      iree_hal_caching_allocator_cast(base_allocator);

  bool can_pool = true;

  // If the buffer is constant/exported then we'll skip the pool.
  // Constant buffers are often mapped directly from files or host memory and
  // benefit from being routed down to the underlying allocator.
  if (iree_any_bit_set(params->usage,
                       IREE_HAL_BUFFER_USAGE_SHARING_EXPORT |
                           IREE_HAL_BUFFER_USAGE_SHARING_IMMUTABLE |
                           IREE_HAL_BUFFER_USAGE_SHARING_REPLICATE)) {
    can_pool = false;
  }

  // Performance warning: if the initial_data is non-empty and the memory type
  // does not support mapping we bypass the pool and go straight to the
  // underlying allocator. This is because initial data uploads require transfer
  // operations and we don't want to require a device as that would prevent us
  // from sharing the same pool across several devices. We could pass in a
  // device per allocation request if it becomes a heavily used pattern but in
  // most cases where initial_data is used it's for a CONSTANT buffer that we
  // aren't pooling anyway.
  if (!iree_const_byte_span_is_empty(initial_data) &&
      !iree_all_bits_set(params->usage, IREE_HAL_BUFFER_USAGE_MAPPING)) {
    can_pool = false;
  }

  // If we don't want to pool then early-exit to the backing allocator.
  if (!can_pool) {
    return iree_hal_allocator_allocate_buffer(allocator->device_allocator,
                                              *params, allocation_size,
                                              initial_data, out_buffer);
  }

  // We need to ensure we have the same parameters the allocator will use so
  // that when we reuse memory we're using the broadest set to select from that
  // can service each request.
  iree_hal_buffer_params_t compat_params;
  if (!iree_all_bits_set(iree_hal_allocator_query_buffer_compatibility(
                             allocator->device_allocator, *params,
                             allocation_size, &compat_params, &allocation_size),
                         IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "allocator cannot allocate a buffer with the given parameters");
  }

  // Try to find a pool for the buffer parameters.
  iree_hal_caching_allocator_pool_t* pool =
      iree_hal_caching_allocator_find_pool(allocator, compat_params.type,
                                           compat_params.usage);
  if (!pool) {
    // Fallback to the underlying allocator.
    return iree_hal_allocator_allocate_buffer(allocator->device_allocator,
                                              compat_params, allocation_size,
                                              initial_data, out_buffer);
  }

  // Acquire the buffer from the pool.
  IREE_RETURN_IF_ERROR(iree_hal_caching_allocator_pool_acquire(
      pool, &compat_params, allocation_size, initial_data, out_buffer));

  // Point the buffer back to us for deallocation.
  (*out_buffer)->device_allocator = base_allocator;

  return iree_ok_status();
}

static void iree_hal_caching_allocator_deallocate_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT buffer) {
  iree_hal_caching_allocator_t* allocator =
      iree_hal_caching_allocator_cast(base_allocator);

  // Try to find the pool we would want to release the buffer into.
  // Note that we are only going to get called if we had successfully placed the
  // buffer into a pool.
  iree_hal_caching_allocator_pool_t* pool =
      iree_hal_caching_allocator_find_pool(
          allocator, iree_hal_buffer_memory_type(buffer),
          iree_hal_buffer_allowed_usage(buffer));
  IREE_ASSERT(pool, "pool to return cached buffer to not found");
  if (!pool) return;

  // Release back to pool (which may deallocate).
  iree_hal_caching_allocator_pool_release(pool, buffer);
}

static iree_status_t iree_hal_caching_allocator_import_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    const iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_hal_external_buffer_t* IREE_RESTRICT external_buffer,
    iree_hal_buffer_release_callback_t release_callback,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  // Bypass the caching allocator and directly ask the backing implementation.
  iree_hal_caching_allocator_t* allocator =
      iree_hal_caching_allocator_cast(base_allocator);
  return iree_hal_allocator_import_buffer(allocator->device_allocator, *params,
                                          external_buffer, release_callback,
                                          out_buffer);
}

static iree_status_t iree_hal_caching_allocator_export_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT buffer,
    iree_hal_external_buffer_type_t requested_type,
    iree_hal_external_buffer_flags_t requested_flags,
    iree_hal_external_buffer_t* IREE_RESTRICT out_external_buffer) {
  // Bypass the caching allocator and directly ask the backing implementation.
  //
  // TODO(benvanik): confirm we want to allow export? it can arbitrarily extend
  // lifetime and we want to ensure we aren't going to reuse buffers with
  // outstanding exports.
  iree_hal_caching_allocator_t* allocator =
      iree_hal_caching_allocator_cast(base_allocator);
  return iree_hal_allocator_export_buffer(allocator->device_allocator, buffer,
                                          requested_type, requested_flags,
                                          out_external_buffer);
}

static const iree_hal_allocator_vtable_t iree_hal_caching_allocator_vtable = {
    .destroy = iree_hal_caching_allocator_destroy,
    .host_allocator = iree_hal_caching_allocator_host_allocator,
    .trim = iree_hal_caching_allocator_trim,
    .query_statistics = iree_hal_caching_allocator_query_statistics,
    .query_memory_heaps = iree_hal_caching_allocator_query_memory_heaps,
    .query_buffer_compatibility =
        iree_hal_caching_allocator_query_buffer_compatibility,
    .allocate_buffer = iree_hal_caching_allocator_allocate_buffer,
    .deallocate_buffer = iree_hal_caching_allocator_deallocate_buffer,
    .import_buffer = iree_hal_caching_allocator_import_buffer,
    .export_buffer = iree_hal_caching_allocator_export_buffer,
};

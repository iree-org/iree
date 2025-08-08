// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/streaming/internal.h"

//===----------------------------------------------------------------------===//
// Memory pool management
//===----------------------------------------------------------------------===//

static void iree_hal_streaming_mem_pool_destroy(
    iree_hal_streaming_mem_pool_t* pool);

iree_status_t iree_hal_streaming_mem_pool_create(
    iree_hal_streaming_context_t* context,
    const iree_hal_streaming_mem_pool_props_t* props,
    iree_allocator_t host_allocator, iree_hal_streaming_mem_pool_t** out_pool) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(props);
  IREE_ASSERT_ARGUMENT(out_pool);
  *out_pool = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Allocate memory pool structure.
  iree_hal_streaming_mem_pool_t* pool = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*pool), (void**)&pool));

  // Initialize pool.
  iree_atomic_ref_count_init(&pool->ref_count);
  pool->context = context;
  iree_hal_streaming_context_retain(context);
  pool->props = *props;
  pool->release_threshold = 0;
  pool->reuse_allow_internal_dependencies = false;
  pool->reuse_follow_event_dependencies = true;
  pool->reuse_allow_opportunistic = false;
  pool->reserved_mem_current = 0;
  pool->reserved_mem_high = 0;
  pool->used_mem_current = 0;
  pool->used_mem_high = 0;
  pool->platform_handle = NULL;
  iree_slim_mutex_initialize(&pool->mutex);
  pool->host_allocator = host_allocator;

  *out_pool = pool;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_streaming_mem_pool_destroy(
    iree_hal_streaming_mem_pool_t* pool) {
  IREE_ASSERT_ARGUMENT(pool);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Clean up platform handle if present.
  if (pool->platform_handle) {
    // TODO(benvanik): platform-specific cleanup.
  }

  iree_slim_mutex_deinitialize(&pool->mutex);
  iree_hal_streaming_context_release(pool->context);
  iree_allocator_free(pool->host_allocator, pool);

  IREE_TRACE_ZONE_END(z0);
}

void iree_hal_streaming_mem_pool_retain(iree_hal_streaming_mem_pool_t* pool) {
  if (pool) {
    iree_atomic_ref_count_inc(&pool->ref_count);
  }
}

void iree_hal_streaming_mem_pool_release(iree_hal_streaming_mem_pool_t* pool) {
  if (pool && iree_atomic_ref_count_dec(&pool->ref_count) == 1) {
    iree_hal_streaming_mem_pool_destroy(pool);
  }
}

iree_status_t iree_hal_streaming_mem_pool_get_attribute(
    iree_hal_streaming_mem_pool_t* pool,
    iree_hal_streaming_mem_pool_attr_t attr, uint64_t* out_value) {
  IREE_ASSERT_ARGUMENT(pool);
  IREE_ASSERT_ARGUMENT(out_value);
  *out_value = 0;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_slim_mutex_lock(&pool->mutex);

  iree_status_t status = iree_ok_status();
  switch (attr) {
    case IREE_HAL_STREAMING_MEM_POOL_ATTR_RELEASE_THRESHOLD:
      *out_value = pool->release_threshold;
      break;
    case IREE_HAL_STREAMING_MEM_POOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES:
      *out_value = pool->reuse_allow_internal_dependencies ? 1 : 0;
      break;
    case IREE_HAL_STREAMING_MEM_POOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES:
      *out_value = pool->reuse_follow_event_dependencies ? 1 : 0;
      break;
    case IREE_HAL_STREAMING_MEM_POOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC:
      *out_value = pool->reuse_allow_opportunistic ? 1 : 0;
      break;
    case IREE_HAL_STREAMING_MEM_POOL_ATTR_RESERVED_MEM_CURRENT:
      *out_value = pool->reserved_mem_current;
      break;
    case IREE_HAL_STREAMING_MEM_POOL_ATTR_RESERVED_MEM_HIGH:
      *out_value = pool->reserved_mem_high;
      break;
    case IREE_HAL_STREAMING_MEM_POOL_ATTR_USED_MEM_CURRENT:
      *out_value = pool->used_mem_current;
      break;
    case IREE_HAL_STREAMING_MEM_POOL_ATTR_USED_MEM_HIGH:
      *out_value = pool->used_mem_high;
      break;
    default:
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "invalid memory pool attribute");
      break;
  }

  iree_slim_mutex_unlock(&pool->mutex);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_mem_pool_set_attribute(
    iree_hal_streaming_mem_pool_t* pool,
    iree_hal_streaming_mem_pool_attr_t attr, uint64_t value) {
  IREE_ASSERT_ARGUMENT(pool);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_slim_mutex_lock(&pool->mutex);

  iree_status_t status = iree_ok_status();
  switch (attr) {
    case IREE_HAL_STREAMING_MEM_POOL_ATTR_RELEASE_THRESHOLD:
      pool->release_threshold = value;
      break;
    case IREE_HAL_STREAMING_MEM_POOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES:
      pool->reuse_allow_internal_dependencies = value != 0;
      break;
    case IREE_HAL_STREAMING_MEM_POOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES:
      pool->reuse_follow_event_dependencies = value != 0;
      break;
    case IREE_HAL_STREAMING_MEM_POOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC:
      pool->reuse_allow_opportunistic = value != 0;
      break;
    default:
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "invalid memory pool attribute");
      break;
  }

  iree_slim_mutex_unlock(&pool->mutex);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_mem_pool_trim_to(
    iree_hal_streaming_mem_pool_t* pool, iree_device_size_t min_bytes_to_keep) {
  IREE_ASSERT_ARGUMENT(pool);
  IREE_TRACE_ZONE_BEGIN(z0);

  // TODO(benvanik): implement memory pool trimming.
  // Ignored for now as it's just a hint.

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_hal_streaming_mem_pool_t* iree_hal_streaming_device_default_mem_pool(
    iree_hal_streaming_device_t* device) {
  IREE_ASSERT_ARGUMENT(device);
  return device->default_mem_pool;
}

iree_hal_streaming_mem_pool_t* iree_hal_streaming_device_mem_pool(
    iree_hal_streaming_device_t* device) {
  IREE_ASSERT_ARGUMENT(device);
  return device->current_mem_pool;
}

iree_status_t iree_hal_streaming_device_set_mem_pool(
    iree_hal_streaming_device_t* device, iree_hal_streaming_mem_pool_t* pool) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Release old pool if present.
  if (device->current_mem_pool) {
    iree_hal_streaming_mem_pool_release(device->current_mem_pool);
  }

  // Set new pool.
  device->current_mem_pool = pool;
  if (pool) {
    iree_hal_streaming_mem_pool_retain(pool);
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_memory_allocate_async(
    iree_hal_streaming_context_t* context, iree_device_size_t size,
    iree_hal_streaming_stream_t* stream,
    iree_hal_streaming_deviceptr_t* out_ptr) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_ptr);
  *out_ptr = 0;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Get the default memory pool from the device.
  iree_hal_streaming_mem_pool_t* default_pool =
      iree_hal_streaming_device_default_mem_pool(context->device_entry);

  if (!default_pool) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "no default memory pool available");
  }

  // Allocate from the default pool.
  iree_status_t status = iree_hal_streaming_memory_allocate_from_pool_async(
      default_pool, size, stream, out_ptr);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_memory_allocate_from_pool_async(
    iree_hal_streaming_mem_pool_t* pool, iree_device_size_t size,
    iree_hal_streaming_stream_t* stream,
    iree_hal_streaming_deviceptr_t* out_ptr) {
  IREE_ASSERT_ARGUMENT(pool);
  IREE_ASSERT_ARGUMENT(out_ptr);
  *out_ptr = 0;
  IREE_TRACE_ZONE_BEGIN(z0);

  // TODO(benvanik): implement async memory allocation from pool.

  IREE_TRACE_ZONE_END(z0);
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "async memory allocation from pool not yet implemented");
}

iree_status_t iree_hal_streaming_memory_free_async(
    iree_hal_streaming_context_t* context, iree_hal_streaming_deviceptr_t ptr,
    iree_hal_streaming_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_TRACE_ZONE_BEGIN(z0);

  // TODO(benvanik): implement async memory free.

  IREE_TRACE_ZONE_END(z0);
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "async memory free not yet implemented");
}

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

  // Check if allocator supports virtual memory.
  pool->supports_virtual_memory = false;
  pool->vm_page_size_min = 0;
  pool->vm_page_size_recommended = 0;
  pool->pending_allocations = NULL;
  pool->pending_count = 0;
  pool->free_physical_blocks = NULL;

  if (iree_hal_allocator_supports_virtual_memory(context->device_allocator)) {
    pool->supports_virtual_memory = true;

    // Query granularity for device-local memory.
    iree_hal_buffer_params_t params = {
        .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
        .access = IREE_HAL_MEMORY_ACCESS_ALL,
        .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
        .queue_affinity = context->queue_affinity,
    };
    iree_status_t vm_status =
        iree_hal_allocator_virtual_memory_query_granularity(
            context->device_allocator, params, &pool->vm_page_size_min,
            &pool->vm_page_size_recommended);
    if (!iree_status_is_ok(vm_status)) {
      // If granularity query fails, disable virtual memory support.
      pool->supports_virtual_memory = false;
      iree_status_ignore(vm_status);
    }
  }

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

//===----------------------------------------------------------------------===//
// Async allocation helpers
//===----------------------------------------------------------------------===//

// Cleanup async allocation after decommit.
static void iree_hal_streaming_async_allocation_cleanup(
    iree_hal_streaming_mem_pool_t* pool,
    iree_hal_streaming_async_allocation_t* alloc) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Remove from pending list.
  iree_slim_mutex_lock(&pool->mutex);
  iree_hal_streaming_async_allocation_t** prev = &pool->pending_allocations;
  while (*prev) {
    if (*prev == alloc) {
      *prev = alloc->next;
      pool->pending_count--;
      break;
    }
    prev = &(*prev)->next;
  }
  iree_slim_mutex_unlock(&pool->mutex);

  // Release virtual buffer.
  if (alloc->virtual_buffer) {
    iree_hal_allocator_virtual_memory_release(pool->context->device_allocator,
                                               alloc->virtual_buffer);
    iree_hal_buffer_release(alloc->virtual_buffer);
    alloc->virtual_buffer = NULL;
  }

  // Free tracking structure.
  iree_allocator_free(alloc->host_allocator, alloc);

  IREE_TRACE_ZONE_END(z0);
}

// Host callback: Commit physical memory to virtual range or decommit.
void iree_hal_streaming_async_commit_callback(void* user_data) {
  iree_hal_streaming_async_commit_context_t* ctx =
      (iree_hal_streaming_async_commit_context_t*)user_data;

  iree_hal_streaming_async_allocation_t* alloc = ctx->allocation;
  iree_hal_streaming_context_t* context = ctx->context;

  if (ctx->is_commit) {
    // COMMIT: Allocate physical memory and map.
    if (alloc->state == IREE_HAL_STREAMING_ASYNC_ALLOC_RESERVED) {
      // 1. Allocate physical memory.
      iree_hal_buffer_params_t params = {
          .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
          .access = IREE_HAL_MEMORY_ACCESS_ALL,
          .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
          .queue_affinity = context->queue_affinity,
      };

      iree_status_t status = iree_hal_allocator_physical_memory_allocate(
          context->device_allocator, params, alloc->size, alloc->host_allocator,
          &alloc->physical_memory);

      if (iree_status_is_ok(status)) {
        // 2. Map physical memory to virtual buffer.
        status = iree_hal_allocator_virtual_memory_map(
            context->device_allocator, alloc->virtual_buffer, 0,
            alloc->physical_memory, 0, alloc->size);

        if (!iree_status_is_ok(status)) {
          iree_status_ignore(status);
          iree_hal_allocator_physical_memory_free(context->device_allocator,
                                                   alloc->physical_memory);
          alloc->physical_memory = NULL;
        } else {
          // 3. Set memory permissions (READ|WRITE).
          status = iree_hal_allocator_virtual_memory_protect(
              context->device_allocator, alloc->virtual_buffer, 0, alloc->size,
              context->queue_affinity,
              IREE_HAL_MEMORY_PROTECTION_READ |
                  IREE_HAL_MEMORY_PROTECTION_WRITE);

          if (!iree_status_is_ok(status)) {
            iree_status_ignore(status);
            // Continue anyway - some backends may not support protect.
          }

          // 4. Update state.
          alloc->state = IREE_HAL_STREAMING_ASYNC_ALLOC_COMMITTED;
        }
      } else {
        iree_status_ignore(status);
      }
    }
  } else {
    // DECOMMIT: Unmap and free physical memory.
    if (alloc->state == IREE_HAL_STREAMING_ASYNC_ALLOC_COMMITTED) {
      // 1. Unmap physical memory.
      iree_status_t status = iree_hal_allocator_virtual_memory_unmap(
          context->device_allocator, alloc->virtual_buffer, 0, alloc->size);
      iree_status_ignore(status);

      // 2. Free physical memory.
      if (alloc->physical_memory) {
        iree_hal_allocator_physical_memory_free(context->device_allocator,
                                                 alloc->physical_memory);
        alloc->physical_memory = NULL;
      }

      // 3. Update state.
      alloc->state = IREE_HAL_STREAMING_ASYNC_ALLOC_DECOMMITTING;

      // 4. Cleanup the allocation.
      iree_hal_streaming_async_allocation_cleanup(alloc->pool, alloc);
    }
  }

  iree_allocator_free(alloc->host_allocator, ctx);
}

//===----------------------------------------------------------------------===//
// Async memory allocation
//===----------------------------------------------------------------------===//

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
      context, default_pool, size, stream, out_ptr);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_memory_allocate_from_pool_async(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_mem_pool_t* pool, iree_device_size_t size,
    iree_hal_streaming_stream_t* stream,
    iree_hal_streaming_deviceptr_t* out_ptr) {
  IREE_ASSERT_ARGUMENT(pool);
  IREE_ASSERT_ARGUMENT(out_ptr);
  *out_ptr = 0;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Fast path: If VM not supported, fall back to blocking allocation.
  if (!pool->supports_virtual_memory) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_streaming_stream_synchronize(stream));
    iree_hal_streaming_buffer_t* buffer = NULL;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_streaming_memory_allocate_device(context, size, 0, &buffer));
    *out_ptr = (iree_hal_streaming_deviceptr_t)buffer->device_ptr;
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // NEW: Async path with virtual memory.

  // 1. Allocate tracking structure.
  iree_hal_streaming_async_allocation_t* async_alloc = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(pool->host_allocator, sizeof(*async_alloc),
                                (void**)&async_alloc));

  // 2. Round size up to page boundary.
  iree_device_size_t aligned_size =
      (size + pool->vm_page_size_recommended - 1) &
      ~(pool->vm_page_size_recommended - 1);

  // 3. Reserve virtual address space (NON-BLOCKING).
  iree_status_t status = iree_hal_allocator_virtual_memory_reserve(
      context->device_allocator, context->queue_affinity, aligned_size,
      &async_alloc->virtual_buffer);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(pool->host_allocator, async_alloc);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // 4. Extract device pointer from virtual buffer.
  iree_hal_external_buffer_t external_buffer;
  status = iree_hal_allocator_export_buffer(
      context->device_allocator, async_alloc->virtual_buffer,
      IREE_HAL_EXTERNAL_BUFFER_TYPE_DEVICE_ALLOCATION,
      IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE, &external_buffer);
  if (!iree_status_is_ok(status)) {
    iree_hal_allocator_virtual_memory_release(context->device_allocator,
                                               async_alloc->virtual_buffer);
    iree_hal_buffer_release(async_alloc->virtual_buffer);
    iree_allocator_free(pool->host_allocator, async_alloc);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  async_alloc->virtual_ptr =
      (iree_hal_streaming_deviceptr_t)external_buffer.handle.device_allocation
          .ptr;
  async_alloc->size = aligned_size;
  async_alloc->physical_memory = NULL;
  async_alloc->state = IREE_HAL_STREAMING_ASYNC_ALLOC_RESERVED;
  async_alloc->pool = pool;
  async_alloc->host_allocator = pool->host_allocator;

  // 5. Record timeline value for lifetime tracking.
  iree_slim_mutex_lock(&stream->mutex);
  async_alloc->alloc_timeline_value = stream->pending_value;
  async_alloc->first_use_value = UINT64_MAX;
  async_alloc->last_use_value = 0;
  iree_slim_mutex_unlock(&stream->mutex);

  // 6. Add to pool's pending list.
  iree_slim_mutex_lock(&pool->mutex);
  async_alloc->next = pool->pending_allocations;
  pool->pending_allocations = async_alloc;
  pool->pending_count++;
  iree_slim_mutex_unlock(&pool->mutex);

  // 7. Schedule commit callback to execute before next work submission.
  // Allocate callback context.
  iree_hal_streaming_async_commit_context_t* commit_ctx = NULL;
  status = iree_allocator_malloc(pool->host_allocator, sizeof(*commit_ctx),
                                  (void**)&commit_ctx);
  if (!iree_status_is_ok(status)) {
    // Remove from pending list on error.
    iree_slim_mutex_lock(&pool->mutex);
    iree_hal_streaming_async_allocation_t** prev = &pool->pending_allocations;
    while (*prev) {
      if (*prev == async_alloc) {
        *prev = async_alloc->next;
        pool->pending_count--;
        break;
      }
      prev = &(*prev)->next;
    }
    iree_slim_mutex_unlock(&pool->mutex);

    iree_hal_allocator_virtual_memory_release(context->device_allocator,
                                               async_alloc->virtual_buffer);
    iree_hal_buffer_release(async_alloc->virtual_buffer);
    iree_allocator_free(pool->host_allocator, async_alloc);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  commit_ctx->context = context;
  commit_ctx->allocation = async_alloc;
  commit_ctx->is_commit = true;

  // Launch host function to commit physical memory.
  status =
      iree_hal_streaming_launch_host_function(
          stream, iree_hal_streaming_async_commit_callback, commit_ctx);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(pool->host_allocator, commit_ctx);

    // Remove from pending list on error.
    iree_slim_mutex_lock(&pool->mutex);
    iree_hal_streaming_async_allocation_t** prev = &pool->pending_allocations;
    while (*prev) {
      if (*prev == async_alloc) {
        *prev = async_alloc->next;
        pool->pending_count--;
        break;
      }
      prev = &(*prev)->next;
    }
    iree_slim_mutex_unlock(&pool->mutex);

    iree_hal_allocator_virtual_memory_release(context->device_allocator,
                                               async_alloc->virtual_buffer);
    iree_hal_buffer_release(async_alloc->virtual_buffer);
    iree_allocator_free(pool->host_allocator, async_alloc);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  *out_ptr = async_alloc->virtual_ptr;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_memory_free_async(
    iree_hal_streaming_context_t* context, iree_hal_streaming_deviceptr_t ptr,
    iree_hal_streaming_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Look up allocation from device pointer.
  iree_hal_streaming_mem_pool_t* pool =
      iree_hal_streaming_device_mem_pool(context->device_entry);

  if (!pool) {
    pool = iree_hal_streaming_device_default_mem_pool(context->device_entry);
  }
  if (!pool || !pool->supports_virtual_memory) {
    // Fall back to blocking free.
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_streaming_stream_synchronize(stream));
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_streaming_memory_free_device(context, ptr));
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // Find allocation in pending list.
  iree_hal_streaming_async_allocation_t* alloc = NULL;
  iree_slim_mutex_lock(&pool->mutex);
  for (iree_hal_streaming_async_allocation_t* curr = pool->pending_allocations;
       curr != NULL; curr = curr->next) {
    if (curr->virtual_ptr == ptr) {
      alloc = curr;
      break;
    }
  }
  iree_slim_mutex_unlock(&pool->mutex);

  if (!alloc) {
    // Not an async allocation, fall back to blocking free.
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_streaming_stream_synchronize(stream));
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_streaming_memory_free_device(context, ptr));
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // Record free timeline value.
  iree_slim_mutex_lock(&stream->mutex);
  alloc->free_timeline_value = stream->pending_value;
  iree_slim_mutex_unlock(&stream->mutex);

  // Allocate decommit context.
  iree_hal_streaming_async_commit_context_t* decommit_ctx = NULL;
  iree_status_t status = iree_allocator_malloc(
      pool->host_allocator, sizeof(*decommit_ctx), (void**)&decommit_ctx);
  if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  decommit_ctx->context = context;
  decommit_ctx->allocation = alloc;
  decommit_ctx->is_commit = false;

  // Launch host function to decommit physical memory after work completes.
  status = iree_hal_streaming_launch_host_function(
      stream, iree_hal_streaming_async_commit_callback, decommit_ctx);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(pool->host_allocator, decommit_ctx);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Future optimization helpers (stubs for now)
//===----------------------------------------------------------------------===//

// Analyze if two async allocations have overlapping lifetimes.
// This is used for future physical memory reuse optimization.
static bool iree_hal_streaming_async_allocations_overlap(
    const iree_hal_streaming_async_allocation_t* a,
    const iree_hal_streaming_async_allocation_t* b) {
  // If either hasn't been used yet, assume overlap.
  if (a->first_use_value == UINT64_MAX || b->first_use_value == UINT64_MAX) {
    return true;
  }

  // Check timeline overlap.
  // a overlaps b if: a.last_use >= b.first_use AND a.first_use <= b.last_use
  bool overlaps = (a->last_use_value >= b->first_use_value) &&
                  (a->first_use_value <= b->last_use_value);

  return overlaps;
}

// Find non-overlapping allocations that can share physical memory.
// This is a stub for future physical memory reuse optimization.
static iree_hal_physical_memory_t*
iree_hal_streaming_mem_pool_find_reusable_physical(
    iree_hal_streaming_mem_pool_t* pool, iree_device_size_t size,
    uint64_t first_use_value) {
  // Scan free physical blocks.
  for (iree_hal_streaming_physical_memory_block_t* block =
           pool->free_physical_blocks;
       block; block = block->next) {
    // Check if block is available and large enough.
    if (block->size >= size &&
        block->available_after_value <= first_use_value) {
      return block->physical_memory;
    }
  }

  return NULL;  // No reusable physical memory found.
}

// Add freed physical memory to reuse pool.
// This is a stub for future physical memory reuse optimization.
static void iree_hal_streaming_mem_pool_recycle_physical(
    iree_hal_streaming_mem_pool_t* pool,
    iree_hal_physical_memory_t* physical_memory, iree_device_size_t size,
    uint64_t available_after_value) {
  // TODO: Implement in future optimization phase.
  // For now, just free immediately.
  iree_hal_allocator_physical_memory_free(pool->context->device_allocator,
                                           physical_memory);
}

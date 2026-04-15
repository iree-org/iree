// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/transient_buffer.h"

struct iree_hal_amdgpu_transient_buffer_t {
  // Base HAL buffer resource returned to callers.
  iree_hal_buffer_t base;

  // Pool this wrapper returns to when its HAL buffer refcount reaches zero.
  iree_hal_amdgpu_transient_buffer_pool_t* pool;

  // Next wrapper in either the pool return stack or acquire-side cache.
  iree_hal_amdgpu_transient_buffer_t* pool_next;

  // Provider-backed view staged for queue packet emission and future commit;
  // retained while non-NULL.
  iree_hal_buffer_t* staged_backing;

  // Host-visible committed backing. NULL before alloca commit and after
  // dealloca decommit.
  //
  // The queue's semaphore edges provide the real ordering contract; the
  // acquire/release atomics here make host-side wrapper state transitions
  // data-race-free and visible to TSAN.
  iree_atomic_intptr_t committed_backing;

  // Borrowed source pool for the queue-owned reservation token.
  iree_hal_pool_t* reservation_pool;

  // Queue-owned pool reservation token attached after acquire succeeds.
  iree_hal_pool_reservation_t reservation;

  // Non-zero while |reservation| is valid and must be released.
  iree_atomic_int32_t reservation_armed;

  // Set when one dealloca has been accepted for this wrapper. This is
  // single-owner bookkeeping for reservation release/decommit, not a queue-use
  // lifetime validator; queue operation order is expressed by semaphores.
  iree_atomic_int32_t dealloca_queued;
};

static const iree_hal_buffer_vtable_t iree_hal_amdgpu_transient_buffer_vtable;

static inline iree_hal_amdgpu_transient_buffer_t*
iree_hal_amdgpu_transient_buffer_cast(iree_hal_buffer_t* base_buffer) {
  IREE_HAL_ASSERT_TYPE(base_buffer, &iree_hal_amdgpu_transient_buffer_vtable);
  return (iree_hal_amdgpu_transient_buffer_t*)base_buffer;
}

static inline const iree_hal_buffer_vtable_t*
iree_hal_amdgpu_transient_buffer_backing_vtable(iree_hal_buffer_t* buffer) {
  return (const iree_hal_buffer_vtable_t*)((const iree_hal_resource_t*)buffer)
      ->vtable;
}

static inline iree_hal_buffer_t*
iree_hal_amdgpu_transient_buffer_load_committed_backing(
    iree_hal_amdgpu_transient_buffer_t* buffer) {
  return (iree_hal_buffer_t*)iree_atomic_load(&buffer->committed_backing,
                                              iree_memory_order_acquire);
}

//===----------------------------------------------------------------------===//
// Transient buffer pool
//===----------------------------------------------------------------------===//

static iree_host_size_t iree_hal_amdgpu_transient_buffer_pool_slot_size(void) {
  return iree_host_align(sizeof(iree_hal_amdgpu_transient_buffer_t),
                         iree_alignof(iree_hal_amdgpu_transient_buffer_t));
}

static iree_status_t iree_hal_amdgpu_transient_buffer_pool_grow_locked(
    iree_hal_amdgpu_transient_buffer_pool_t* pool) {
  IREE_TRACE_ZONE_BEGIN(z0);

  const iree_host_size_t slot_size =
      iree_hal_amdgpu_transient_buffer_pool_slot_size();
  const iree_host_size_t slot_count =
      pool->block_pool->usable_block_size / slot_size;
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)slot_count);

  iree_arena_block_t* block = NULL;
  void* block_ptr = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_arena_block_pool_acquire(pool->block_pool, &block, &block_ptr));

  if (pool->block_tail) {
    pool->block_tail->next = block;
  } else {
    pool->block_head = block;
  }
  pool->block_tail = block;

  uint8_t* slot_ptr = (uint8_t*)block_ptr;
  for (iree_host_size_t i = 0; i < slot_count; ++i) {
    iree_hal_amdgpu_transient_buffer_t* buffer =
        (iree_hal_amdgpu_transient_buffer_t*)slot_ptr;
    buffer->pool = pool;
    buffer->pool_next = pool->acquire_head;
    pool->acquire_head = buffer;
    slot_ptr += slot_size;
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_transient_buffer_pool_initialize(
    iree_arena_block_pool_t* block_pool,
    iree_hal_amdgpu_transient_buffer_pool_t* out_pool) {
  IREE_ASSERT_ARGUMENT(block_pool);
  IREE_ASSERT_ARGUMENT(out_pool);
  IREE_TRACE_ZONE_BEGIN(z0);

  memset(out_pool, 0, sizeof(*out_pool));
  const iree_host_size_t slot_size =
      iree_hal_amdgpu_transient_buffer_pool_slot_size();
  if (IREE_UNLIKELY(block_pool->usable_block_size < slot_size)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "transient buffer pool block usable size %" PRIhsz
                            " is smaller than wrapper slot size %" PRIhsz,
                            block_pool->usable_block_size, slot_size);
  }

  out_pool->block_pool = block_pool;
  iree_atomic_store(&out_pool->return_head, 0, iree_memory_order_relaxed);
  iree_slim_mutex_initialize(&out_pool->mutex);
#if !defined(NDEBUG)
  iree_atomic_store(&out_pool->live_count, 0, iree_memory_order_relaxed);
#endif  // !defined(NDEBUG)

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

void iree_hal_amdgpu_transient_buffer_pool_deinitialize(
    iree_hal_amdgpu_transient_buffer_pool_t* pool) {
  if (!pool || !pool->block_pool) return;
  IREE_TRACE_ZONE_BEGIN(z0);

#if !defined(NDEBUG)
  const int32_t live_count =
      iree_atomic_load(&pool->live_count, iree_memory_order_acquire);
  IREE_ASSERT(live_count == 0,
              "deinitializing transient buffer pool with %d live wrappers",
              live_count);
#endif  // !defined(NDEBUG)

  iree_atomic_store(&pool->return_head, 0, iree_memory_order_relaxed);
  pool->acquire_head = NULL;
  if (pool->block_head) {
    iree_arena_block_pool_release(pool->block_pool, pool->block_head,
                                  pool->block_tail);
  }
  pool->block_head = NULL;
  pool->block_tail = NULL;
  iree_slim_mutex_deinitialize(&pool->mutex);
  pool->block_pool = NULL;

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_amdgpu_transient_buffer_pool_acquire(
    iree_hal_amdgpu_transient_buffer_pool_t* pool,
    iree_hal_amdgpu_transient_buffer_t** out_buffer) {
  *out_buffer = NULL;

  iree_slim_mutex_lock(&pool->mutex);

  iree_status_t status = iree_ok_status();
  iree_hal_amdgpu_transient_buffer_t* buffer = pool->acquire_head;
  if (buffer) {
    pool->acquire_head = buffer->pool_next;
  } else {
    buffer = (iree_hal_amdgpu_transient_buffer_t*)iree_atomic_exchange(
        &pool->return_head, 0, iree_memory_order_acquire);
    if (buffer) {
      pool->acquire_head = buffer->pool_next;
    } else {
      status = iree_hal_amdgpu_transient_buffer_pool_grow_locked(pool);
      if (iree_status_is_ok(status)) {
        buffer = pool->acquire_head;
        pool->acquire_head = buffer->pool_next;
      }
    }
  }

  iree_slim_mutex_unlock(&pool->mutex);

  if (iree_status_is_ok(status)) {
    buffer->pool_next = NULL;
#if !defined(NDEBUG)
    iree_atomic_fetch_add(&pool->live_count, 1, iree_memory_order_acq_rel);
#endif  // !defined(NDEBUG)
    *out_buffer = buffer;
  }
  return status;
}

static void iree_hal_amdgpu_transient_buffer_pool_release(
    iree_hal_amdgpu_transient_buffer_pool_t* pool,
    iree_hal_amdgpu_transient_buffer_t* buffer) {
#if !defined(NDEBUG)
  const int32_t old_live_count =
      iree_atomic_fetch_sub(&pool->live_count, 1, iree_memory_order_acq_rel);
  IREE_ASSERT(old_live_count > 0,
              "releasing transient buffer wrapper with no live wrapper count");
#endif  // !defined(NDEBUG)

  intptr_t expected = 0;
  do {
    expected = iree_atomic_load(&pool->return_head, iree_memory_order_relaxed);
    buffer->pool_next = (iree_hal_amdgpu_transient_buffer_t*)expected;
  } while (!iree_atomic_compare_exchange_weak(
      &pool->return_head, &expected, (intptr_t)buffer,
      iree_memory_order_release, iree_memory_order_relaxed));
}

//===----------------------------------------------------------------------===//
// Transient buffer wrapper
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_amdgpu_transient_buffer_create(
    iree_hal_buffer_placement_t placement, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_device_size_t byte_length,
    iree_hal_amdgpu_transient_buffer_pool_t* pool,
    iree_hal_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(pool);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;
  if (IREE_UNLIKELY(byte_length > allocation_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "transient buffer byte length (%" PRIu64
                            ") exceeds allocation size (%" PRIu64 ")",
                            (uint64_t)byte_length, (uint64_t)allocation_size);
  }
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdgpu_transient_buffer_t* buffer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdgpu_transient_buffer_pool_acquire(pool, &buffer));

  iree_hal_buffer_initialize(
      placement, /*allocated_buffer=*/&buffer->base, allocation_size,
      /*byte_offset=*/0, byte_length, params.type, params.access, params.usage,
      &iree_hal_amdgpu_transient_buffer_vtable, &buffer->base);
  buffer->pool = pool;
  buffer->staged_backing = NULL;
  iree_atomic_store(&buffer->committed_backing, 0, iree_memory_order_relaxed);
  buffer->reservation_pool = NULL;
  memset(&buffer->reservation, 0, sizeof(buffer->reservation));
  iree_atomic_store(&buffer->reservation_armed, 0, iree_memory_order_relaxed);
  iree_atomic_store(&buffer->dealloca_queued, 0, iree_memory_order_relaxed);

  *out_buffer = &buffer->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

bool iree_hal_amdgpu_transient_buffer_isa(const iree_hal_buffer_t* buffer) {
  return iree_hal_resource_is(&buffer->resource,
                              &iree_hal_amdgpu_transient_buffer_vtable);
}

void iree_hal_amdgpu_transient_buffer_attach_reservation(
    iree_hal_buffer_t* base_buffer, iree_hal_pool_t* pool,
    const iree_hal_pool_reservation_t* reservation) {
  IREE_ASSERT_ARGUMENT(base_buffer);
  IREE_ASSERT_ARGUMENT(pool);
  IREE_ASSERT_ARGUMENT(reservation);
  iree_hal_amdgpu_transient_buffer_t* buffer =
      iree_hal_amdgpu_transient_buffer_cast(base_buffer);
  IREE_ASSERT_TRUE(buffer->reservation_pool == NULL);
  IREE_ASSERT_TRUE(iree_atomic_load(&buffer->reservation_armed,
                                    iree_memory_order_acquire) == 0);
  buffer->reservation_pool = pool;
  buffer->reservation = *reservation;
  iree_atomic_store(&buffer->reservation_armed, 1, iree_memory_order_release);
}

void iree_hal_amdgpu_transient_buffer_stage_backing(
    iree_hal_buffer_t* base_buffer, iree_hal_buffer_t* backing_buffer) {
  IREE_ASSERT_ARGUMENT(base_buffer);
  IREE_ASSERT_ARGUMENT(backing_buffer);
  iree_hal_amdgpu_transient_buffer_t* buffer =
      iree_hal_amdgpu_transient_buffer_cast(base_buffer);
  IREE_ASSERT_TRUE(buffer->staged_backing == NULL);
  IREE_ASSERT_TRUE(
      iree_hal_amdgpu_transient_buffer_load_committed_backing(buffer) == NULL);
  buffer->staged_backing = backing_buffer;
}

void iree_hal_amdgpu_transient_buffer_commit(iree_hal_buffer_t* base_buffer) {
  IREE_ASSERT_ARGUMENT(base_buffer);
  iree_hal_amdgpu_transient_buffer_t* buffer =
      iree_hal_amdgpu_transient_buffer_cast(base_buffer);
  IREE_ASSERT_TRUE(buffer->staged_backing != NULL);
  IREE_ASSERT_TRUE(
      iree_hal_amdgpu_transient_buffer_load_committed_backing(buffer) == NULL);
  iree_atomic_store(&buffer->committed_backing,
                    (intptr_t)buffer->staged_backing,
                    iree_memory_order_release);
}

void iree_hal_amdgpu_transient_buffer_decommit(iree_hal_buffer_t* base_buffer) {
  IREE_ASSERT_ARGUMENT(base_buffer);
  iree_hal_amdgpu_transient_buffer_t* buffer =
      iree_hal_amdgpu_transient_buffer_cast(base_buffer);
  iree_atomic_store(&buffer->committed_backing, 0, iree_memory_order_release);
  if (buffer->staged_backing) {
    iree_hal_buffer_release(buffer->staged_backing);
    buffer->staged_backing = NULL;
  }
}

bool iree_hal_amdgpu_transient_buffer_begin_dealloca(
    iree_hal_buffer_t* base_buffer) {
  IREE_ASSERT_ARGUMENT(base_buffer);
  iree_hal_amdgpu_transient_buffer_t* buffer =
      iree_hal_amdgpu_transient_buffer_cast(base_buffer);
  int32_t expected = 0;
  return iree_atomic_compare_exchange_strong(
      &buffer->dealloca_queued, &expected, 1, iree_memory_order_acq_rel,
      iree_memory_order_acquire);
}

void iree_hal_amdgpu_transient_buffer_abort_dealloca(
    iree_hal_buffer_t* base_buffer) {
  IREE_ASSERT_ARGUMENT(base_buffer);
  iree_hal_amdgpu_transient_buffer_t* buffer =
      iree_hal_amdgpu_transient_buffer_cast(base_buffer);
  iree_atomic_store(&buffer->dealloca_queued, 0, iree_memory_order_release);
}

void iree_hal_amdgpu_transient_buffer_release_reservation(
    iree_hal_buffer_t* base_buffer,
    const iree_async_frontier_t* death_frontier) {
  IREE_ASSERT_ARGUMENT(base_buffer);
  iree_hal_amdgpu_transient_buffer_t* buffer =
      iree_hal_amdgpu_transient_buffer_cast(base_buffer);
  iree_hal_pool_t* reservation_pool = buffer->reservation_pool;
  if (!reservation_pool) return;
  const int32_t was_armed = iree_atomic_exchange(&buffer->reservation_armed, 0,
                                                 iree_memory_order_acq_rel);
  if (was_armed) {
    iree_hal_pool_release_reservation(reservation_pool, &buffer->reservation,
                                      death_frontier);
  }
  buffer->reservation_pool = NULL;
  memset(&buffer->reservation, 0, sizeof(buffer->reservation));
}

iree_hal_buffer_t* iree_hal_amdgpu_transient_buffer_backing_buffer(
    iree_hal_buffer_t* base_buffer) {
  IREE_ASSERT_ARGUMENT(base_buffer);
  iree_hal_amdgpu_transient_buffer_t* buffer =
      iree_hal_amdgpu_transient_buffer_cast(base_buffer);
  return buffer->staged_backing;
}

static void iree_hal_amdgpu_transient_buffer_destroy(
    iree_hal_buffer_t* base_buffer) {
  iree_hal_amdgpu_transient_buffer_t* buffer =
      iree_hal_amdgpu_transient_buffer_cast(base_buffer);
  iree_hal_amdgpu_transient_buffer_pool_t* pool = buffer->pool;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdgpu_transient_buffer_decommit(base_buffer);
  iree_hal_amdgpu_transient_buffer_release_reservation(base_buffer,
                                                       /*death_frontier=*/NULL);

  iree_atomic_store(&buffer->dealloca_queued, 0, iree_memory_order_relaxed);
  iree_hal_amdgpu_transient_buffer_pool_release(pool, buffer);
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_amdgpu_transient_buffer_load_host_backing(
    iree_hal_amdgpu_transient_buffer_t* buffer,
    iree_hal_buffer_t** out_backing_buffer) {
  if (IREE_UNLIKELY(iree_atomic_load(&buffer->dealloca_queued,
                                     iree_memory_order_acquire) != 0)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "transient buffer has been queued for deallocation");
  }
  iree_hal_buffer_t* backing_buffer =
      iree_hal_amdgpu_transient_buffer_load_committed_backing(buffer);
  if (IREE_UNLIKELY(!backing_buffer)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "transient buffer has not been committed; wait on the alloca "
        "signal semaphores before accessing it");
  }
  *out_backing_buffer = backing_buffer;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_transient_buffer_map_range(
    iree_hal_buffer_t* base_buffer, iree_hal_mapping_mode_t mapping_mode,
    iree_hal_memory_access_t memory_access,
    iree_device_size_t local_byte_offset, iree_device_size_t local_byte_length,
    iree_hal_buffer_mapping_t* mapping) {
  iree_hal_amdgpu_transient_buffer_t* buffer =
      iree_hal_amdgpu_transient_buffer_cast(base_buffer);
  iree_hal_buffer_t* backing_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_transient_buffer_load_host_backing(
      buffer, &backing_buffer));
  return iree_hal_amdgpu_transient_buffer_backing_vtable(backing_buffer)
      ->map_range(backing_buffer, mapping_mode, memory_access,
                  local_byte_offset, local_byte_length, mapping);
}

static iree_status_t iree_hal_amdgpu_transient_buffer_unmap_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length, iree_hal_buffer_mapping_t* mapping) {
  iree_hal_amdgpu_transient_buffer_t* buffer =
      iree_hal_amdgpu_transient_buffer_cast(base_buffer);
  iree_hal_buffer_t* backing_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_transient_buffer_load_host_backing(
      buffer, &backing_buffer));
  return iree_hal_amdgpu_transient_buffer_backing_vtable(backing_buffer)
      ->unmap_range(backing_buffer, local_byte_offset, local_byte_length,
                    mapping);
}

static iree_status_t iree_hal_amdgpu_transient_buffer_invalidate_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  iree_hal_amdgpu_transient_buffer_t* buffer =
      iree_hal_amdgpu_transient_buffer_cast(base_buffer);
  iree_hal_buffer_t* backing_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_transient_buffer_load_host_backing(
      buffer, &backing_buffer));
  return iree_hal_amdgpu_transient_buffer_backing_vtable(backing_buffer)
      ->invalidate_range(backing_buffer, local_byte_offset, local_byte_length);
}

static iree_status_t iree_hal_amdgpu_transient_buffer_flush_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  iree_hal_amdgpu_transient_buffer_t* buffer =
      iree_hal_amdgpu_transient_buffer_cast(base_buffer);
  iree_hal_buffer_t* backing_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_transient_buffer_load_host_backing(
      buffer, &backing_buffer));
  return iree_hal_amdgpu_transient_buffer_backing_vtable(backing_buffer)
      ->flush_range(backing_buffer, local_byte_offset, local_byte_length);
}

static const iree_hal_buffer_vtable_t iree_hal_amdgpu_transient_buffer_vtable =
    {
        .recycle = iree_hal_buffer_recycle,
        .destroy = iree_hal_amdgpu_transient_buffer_destroy,
        .map_range = iree_hal_amdgpu_transient_buffer_map_range,
        .unmap_range = iree_hal_amdgpu_transient_buffer_unmap_range,
        .invalidate_range = iree_hal_amdgpu_transient_buffer_invalidate_range,
        .flush_range = iree_hal_amdgpu_transient_buffer_flush_range,
};

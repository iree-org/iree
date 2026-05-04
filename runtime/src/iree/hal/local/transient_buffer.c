// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/local/transient_buffer.h"

#include "iree/base/threading/mutex.h"

static iree_atomic_int64_t iree_hal_local_transient_buffer_next_profile_id =
    IREE_ATOMIC_VAR_INIT(1);

// Vtable dispatch for forwarding to the committed buffer's implementation.
// Equivalent to IREE_HAL_VTABLE_DISPATCH from detail.h but accessible from
// this HAL-local utility (detail.h is module-private to the HAL core).
static inline const iree_hal_buffer_vtable_t*
iree_hal_local_transient_buffer_committed_vtable(iree_hal_buffer_t* buffer) {
  return (const iree_hal_buffer_vtable_t*)((const iree_hal_resource_t*)buffer)
      ->vtable;
}

struct iree_hal_local_transient_buffer_t {
  // Base HAL buffer resource exposed to callers.
  iree_hal_buffer_t base;

  // Host allocator used for wrapper storage and teardown.
  iree_allocator_t host_allocator;

  // Stable nonzero id used to join profile rows for this wrapper lifetime.
  uint64_t profile_id;

  // Guards all staged backing, committed backing, and reservation state.
  iree_slim_mutex_t mutex;

  // Materialized backing buffer staged for a future commit. Retained by the
  // wrapper while non-NULL.
  iree_hal_buffer_t* staged_backing;

  // Materialized backing buffer visible to accessors after commit.
  iree_hal_buffer_t* committed_backing;

  // Borrowed pool that owns |reservation| when armed.
  iree_hal_pool_t* reservation_pool;

  // Optional queue-allocation reservation owned by this wrapper while armed.
  iree_hal_pool_reservation_t reservation;

  // Nonzero while the wrapper still owns |reservation|.
  int32_t reservation_armed;
};

static const iree_hal_buffer_vtable_t iree_hal_local_transient_buffer_vtable;

static iree_hal_local_transient_buffer_t* iree_hal_local_transient_buffer_cast(
    iree_hal_buffer_t* buffer) {
  return (iree_hal_local_transient_buffer_t*)buffer;
}

static iree_hal_buffer_t* iree_hal_local_transient_buffer_retain_committed(
    iree_hal_local_transient_buffer_t* buffer) {
  iree_slim_mutex_lock(&buffer->mutex);
  iree_hal_buffer_t* committed = buffer->committed_backing;
  if (committed) {
    iree_hal_buffer_retain(committed);
  }
  iree_slim_mutex_unlock(&buffer->mutex);
  return committed;
}

iree_status_t iree_hal_local_transient_buffer_create(
    iree_hal_buffer_placement_t placement, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_device_size_t byte_length,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;
  if (IREE_UNLIKELY(byte_length > allocation_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "transient buffer byte length (%" PRIu64
                            ") exceeds allocation size (%" PRIu64 ")",
                            (uint64_t)byte_length, (uint64_t)allocation_size);
  }
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_local_transient_buffer_t* buffer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, sizeof(*buffer), (void**)&buffer));

  iree_hal_buffer_initialize(
      placement, /*allocated_buffer=*/&buffer->base, allocation_size,
      /*byte_offset=*/0, byte_length, params.type, params.access, params.usage,
      &iree_hal_local_transient_buffer_vtable, &buffer->base);
  buffer->host_allocator = host_allocator;
  buffer->profile_id = (uint64_t)iree_atomic_fetch_add(
      &iree_hal_local_transient_buffer_next_profile_id, 1,
      iree_memory_order_relaxed);
  iree_slim_mutex_initialize(&buffer->mutex);
  buffer->staged_backing = NULL;
  buffer->committed_backing = NULL;
  buffer->reservation_pool = NULL;
  memset(&buffer->reservation, 0, sizeof(buffer->reservation));
  buffer->reservation_armed = 0;

  *out_buffer = &buffer->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

bool iree_hal_local_transient_buffer_isa(const iree_hal_buffer_t* buffer) {
  return iree_hal_resource_is(&buffer->resource,
                              &iree_hal_local_transient_buffer_vtable);
}

bool iree_hal_local_transient_buffer_is_committed(
    iree_hal_buffer_t* base_buffer) {
  iree_hal_local_transient_buffer_t* buffer =
      iree_hal_local_transient_buffer_cast(base_buffer);
  iree_slim_mutex_lock(&buffer->mutex);
  const bool is_committed = buffer->committed_backing != NULL;
  iree_slim_mutex_unlock(&buffer->mutex);
  return is_committed;
}

uint64_t iree_hal_local_transient_buffer_profile_id(
    iree_hal_buffer_t* base_buffer) {
  iree_hal_local_transient_buffer_t* buffer =
      iree_hal_local_transient_buffer_cast(base_buffer);
  return buffer->profile_id;
}

void iree_hal_local_transient_buffer_attach_reservation(
    iree_hal_buffer_t* base_buffer, iree_hal_pool_t* pool,
    const iree_hal_pool_reservation_t* reservation) {
  iree_hal_local_transient_buffer_t* buffer =
      iree_hal_local_transient_buffer_cast(base_buffer);
  IREE_ASSERT_ARGUMENT(pool);
  IREE_ASSERT_ARGUMENT(reservation);
  iree_slim_mutex_lock(&buffer->mutex);
  IREE_ASSERT_TRUE(buffer->reservation_pool == NULL);
  IREE_ASSERT_TRUE(buffer->reservation_armed == 0);
  buffer->reservation_pool = pool;
  buffer->reservation = *reservation;
  buffer->reservation_armed = 1;
  iree_slim_mutex_unlock(&buffer->mutex);
}

void iree_hal_local_transient_buffer_stage_backing(
    iree_hal_buffer_t* base_buffer, iree_hal_buffer_t* backing) {
  iree_hal_local_transient_buffer_t* buffer =
      iree_hal_local_transient_buffer_cast(base_buffer);
  IREE_ASSERT_ARGUMENT(backing);
  iree_slim_mutex_lock(&buffer->mutex);
  IREE_ASSERT_TRUE(buffer->staged_backing == NULL);
  IREE_ASSERT_TRUE(buffer->committed_backing == NULL);
  iree_hal_buffer_retain(backing);
  buffer->staged_backing = backing;
  iree_slim_mutex_unlock(&buffer->mutex);
}

void iree_hal_local_transient_buffer_commit(iree_hal_buffer_t* base_buffer) {
  iree_hal_local_transient_buffer_t* buffer =
      iree_hal_local_transient_buffer_cast(base_buffer);
  iree_slim_mutex_lock(&buffer->mutex);
  IREE_ASSERT_TRUE(buffer->staged_backing != NULL);
  IREE_ASSERT_TRUE(buffer->committed_backing == NULL);
  buffer->committed_backing = buffer->staged_backing;
  iree_slim_mutex_unlock(&buffer->mutex);
}

void iree_hal_local_transient_buffer_decommit(iree_hal_buffer_t* base_buffer) {
  iree_hal_local_transient_buffer_t* buffer =
      iree_hal_local_transient_buffer_cast(base_buffer);
  iree_slim_mutex_lock(&buffer->mutex);
  iree_hal_buffer_t* staged_backing = buffer->staged_backing;
  buffer->staged_backing = NULL;
  buffer->committed_backing = NULL;
  iree_slim_mutex_unlock(&buffer->mutex);
  if (staged_backing) {
    iree_hal_buffer_release(staged_backing);
  }
}

bool iree_hal_local_transient_buffer_query_reservation(
    iree_hal_buffer_t* base_buffer, iree_hal_pool_t** out_pool,
    iree_hal_pool_reservation_t* out_reservation) {
  iree_hal_local_transient_buffer_t* buffer =
      iree_hal_local_transient_buffer_cast(base_buffer);
  iree_slim_mutex_lock(&buffer->mutex);
  const bool has_reservation =
      buffer->reservation_pool != NULL && buffer->reservation_armed;
  if (has_reservation) {
    if (out_pool) *out_pool = buffer->reservation_pool;
    if (out_reservation) *out_reservation = buffer->reservation;
  }
  iree_slim_mutex_unlock(&buffer->mutex);
  return has_reservation;
}

void iree_hal_local_transient_buffer_release_reservation(
    iree_hal_buffer_t* base_buffer,
    const iree_async_frontier_t* death_frontier) {
  iree_hal_local_transient_buffer_t* buffer =
      iree_hal_local_transient_buffer_cast(base_buffer);
  iree_hal_pool_t* pool = NULL;
  iree_hal_pool_reservation_t reservation;
  iree_slim_mutex_lock(&buffer->mutex);
  const int32_t was_armed = buffer->reservation_armed;
  if (was_armed) {
    pool = buffer->reservation_pool;
    reservation = buffer->reservation;
    buffer->reservation_armed = 0;
  }
  iree_slim_mutex_unlock(&buffer->mutex);
  if (was_armed) {
    iree_hal_pool_release_reservation(pool, &reservation, death_frontier);
  }
}

static void iree_hal_local_transient_buffer_destroy(
    iree_hal_buffer_t* base_buffer) {
  iree_hal_local_transient_buffer_t* buffer =
      iree_hal_local_transient_buffer_cast(base_buffer);
  iree_allocator_t host_allocator = buffer->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_local_transient_buffer_decommit(base_buffer);
  iree_hal_local_transient_buffer_release_reservation(base_buffer,
                                                      /*death_frontier=*/NULL);

  iree_slim_mutex_deinitialize(&buffer->mutex);
  iree_allocator_free(host_allocator, buffer);
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_local_transient_buffer_map_range(
    iree_hal_buffer_t* base_buffer, iree_hal_mapping_mode_t mapping_mode,
    iree_hal_memory_access_t memory_access,
    iree_device_size_t local_byte_offset, iree_device_size_t local_byte_length,
    iree_hal_buffer_mapping_t* mapping) {
  iree_hal_local_transient_buffer_t* buffer =
      iree_hal_local_transient_buffer_cast(base_buffer);
  iree_hal_buffer_t* committed =
      iree_hal_local_transient_buffer_retain_committed(buffer);
  if (IREE_UNLIKELY(!committed)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "transient buffer has not been committed; ensure the alloca signal "
        "semaphores are satisfied before accessing it");
  }
  iree_status_t status =
      iree_hal_local_transient_buffer_committed_vtable(committed)->map_range(
          committed, mapping_mode, memory_access, local_byte_offset,
          local_byte_length, mapping);
  if (iree_status_is_ok(status)) {
    if (mapping->impl.is_persistent) {
      iree_hal_buffer_release(committed);
    } else {
      iree_hal_buffer_t* mapped_buffer = mapping->buffer;
      // Scoped maps own their mapped storage until unmap. Transfer that mapping
      // ownership from the transient wrapper to the committed backing buffer so
      // a queue-ordered decommit cannot invalidate the unmap path.
      mapping->buffer = committed;
      iree_hal_buffer_release(mapped_buffer);
    }
  } else {
    iree_hal_buffer_release(committed);
  }
  return status;
}

static iree_status_t iree_hal_local_transient_buffer_unmap_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length, iree_hal_buffer_mapping_t* mapping) {
  iree_hal_local_transient_buffer_t* buffer =
      iree_hal_local_transient_buffer_cast(base_buffer);
  iree_hal_buffer_t* committed =
      iree_hal_local_transient_buffer_retain_committed(buffer);
  if (IREE_UNLIKELY(!committed)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "transient buffer has been decommitted");
  }
  iree_status_t status =
      iree_hal_local_transient_buffer_committed_vtable(committed)->unmap_range(
          committed, local_byte_offset, local_byte_length, mapping);
  iree_hal_buffer_release(committed);
  return status;
}

static iree_status_t iree_hal_local_transient_buffer_invalidate_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  iree_hal_local_transient_buffer_t* buffer =
      iree_hal_local_transient_buffer_cast(base_buffer);
  iree_hal_buffer_t* committed =
      iree_hal_local_transient_buffer_retain_committed(buffer);
  if (IREE_UNLIKELY(!committed)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "transient buffer has been decommitted");
  }
  iree_status_t status =
      iree_hal_local_transient_buffer_committed_vtable(committed)
          ->invalidate_range(committed, local_byte_offset, local_byte_length);
  iree_hal_buffer_release(committed);
  return status;
}

static iree_status_t iree_hal_local_transient_buffer_flush_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  iree_hal_local_transient_buffer_t* buffer =
      iree_hal_local_transient_buffer_cast(base_buffer);
  iree_hal_buffer_t* committed =
      iree_hal_local_transient_buffer_retain_committed(buffer);
  if (IREE_UNLIKELY(!committed)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "transient buffer has been decommitted");
  }
  iree_status_t status =
      iree_hal_local_transient_buffer_committed_vtable(committed)->flush_range(
          committed, local_byte_offset, local_byte_length);
  iree_hal_buffer_release(committed);
  return status;
}

static const iree_hal_buffer_vtable_t iree_hal_local_transient_buffer_vtable = {
    .recycle = iree_hal_buffer_recycle,
    .destroy = iree_hal_local_transient_buffer_destroy,
    .map_range = iree_hal_local_transient_buffer_map_range,
    .unmap_range = iree_hal_local_transient_buffer_unmap_range,
    .invalidate_range = iree_hal_local_transient_buffer_invalidate_range,
    .flush_range = iree_hal_local_transient_buffer_flush_range,
};

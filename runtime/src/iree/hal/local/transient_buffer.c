// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/local/transient_buffer.h"

// Vtable dispatch for forwarding to the committed buffer's implementation.
// Equivalent to IREE_HAL_VTABLE_DISPATCH from detail.h but accessible from
// this HAL-local utility (detail.h is module-private to the HAL core).
static inline const iree_hal_buffer_vtable_t*
iree_hal_local_transient_buffer_committed_vtable(iree_hal_buffer_t* buffer) {
  return (const iree_hal_buffer_vtable_t*)((const iree_hal_resource_t*)buffer)
      ->vtable;
}

struct iree_hal_local_transient_buffer_t {
  iree_hal_buffer_t base;
  iree_allocator_t host_allocator;

  // Materialized backing buffer staged for a future commit. Retained by the
  // wrapper while non-NULL.
  iree_hal_buffer_t* staged_backing;

  // The committed backing buffer. NULL before commit, non-NULL after.
  //
  // Semaphore waits provide the real queue-ordering edge between commit and
  // use, but acquire/release atomics make the transition visible to TSAN and
  // keep concurrent host access to the wrapper's state data-race-free.
  iree_atomic_intptr_t committed;

  // Optional queue-allocation reservation owned by this wrapper.
  //
  // |reservation_pool| is borrowed; pool owners must outlive every transient
  // buffer sourced from them. |reservation_armed| tracks whether the wrapper
  // still owns |reservation| so submit-time dealloca and destroy can race
  // without double-releasing.
  iree_hal_pool_t* reservation_pool;
  iree_hal_pool_reservation_t reservation;
  iree_atomic_int32_t reservation_armed;
};

static const iree_hal_buffer_vtable_t iree_hal_local_transient_buffer_vtable;

static iree_hal_local_transient_buffer_t* iree_hal_local_transient_buffer_cast(
    iree_hal_buffer_t* buffer) {
  return (iree_hal_local_transient_buffer_t*)buffer;
}

static iree_hal_buffer_t* iree_hal_local_transient_buffer_load_committed(
    iree_hal_local_transient_buffer_t* buffer) {
  return (iree_hal_buffer_t*)iree_atomic_load(&buffer->committed,
                                              iree_memory_order_acquire);
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
  buffer->staged_backing = NULL;
  iree_atomic_store(&buffer->committed, 0, iree_memory_order_relaxed);
  buffer->reservation_pool = NULL;
  memset(&buffer->reservation, 0, sizeof(buffer->reservation));
  iree_atomic_store(&buffer->reservation_armed, 0, iree_memory_order_relaxed);

  *out_buffer = &buffer->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

bool iree_hal_local_transient_buffer_isa(const iree_hal_buffer_t* buffer) {
  return iree_hal_resource_is(&buffer->resource,
                              &iree_hal_local_transient_buffer_vtable);
}

void iree_hal_local_transient_buffer_attach_reservation(
    iree_hal_buffer_t* base_buffer, iree_hal_pool_t* pool,
    const iree_hal_pool_reservation_t* reservation) {
  iree_hal_local_transient_buffer_t* buffer =
      iree_hal_local_transient_buffer_cast(base_buffer);
  IREE_ASSERT_ARGUMENT(pool);
  IREE_ASSERT_ARGUMENT(reservation);
  IREE_ASSERT_TRUE(buffer->reservation_pool == NULL);
  IREE_ASSERT_TRUE(iree_atomic_load(&buffer->reservation_armed,
                                    iree_memory_order_acquire) == 0);
  buffer->reservation_pool = pool;
  buffer->reservation = *reservation;
  iree_atomic_store(&buffer->reservation_armed, 1, iree_memory_order_release);
}

void iree_hal_local_transient_buffer_stage_backing(
    iree_hal_buffer_t* base_buffer, iree_hal_buffer_t* backing) {
  iree_hal_local_transient_buffer_t* buffer =
      iree_hal_local_transient_buffer_cast(base_buffer);
  IREE_ASSERT_ARGUMENT(backing);
  IREE_ASSERT_TRUE(buffer->staged_backing == NULL);
  IREE_ASSERT_TRUE(iree_hal_local_transient_buffer_load_committed(buffer) ==
                   NULL);
  iree_hal_buffer_retain(backing);
  buffer->staged_backing = backing;
}

void iree_hal_local_transient_buffer_commit(iree_hal_buffer_t* base_buffer) {
  iree_hal_local_transient_buffer_t* buffer =
      iree_hal_local_transient_buffer_cast(base_buffer);
  IREE_ASSERT_TRUE(buffer->staged_backing != NULL);
  IREE_ASSERT_TRUE(iree_hal_local_transient_buffer_load_committed(buffer) ==
                   NULL);
  iree_atomic_store(&buffer->committed, (intptr_t)buffer->staged_backing,
                    iree_memory_order_release);
}

void iree_hal_local_transient_buffer_decommit(iree_hal_buffer_t* base_buffer) {
  iree_hal_local_transient_buffer_t* buffer =
      iree_hal_local_transient_buffer_cast(base_buffer);
  iree_atomic_exchange(&buffer->committed, 0, iree_memory_order_acq_rel);
  if (buffer->staged_backing) {
    iree_hal_buffer_release(buffer->staged_backing);
    buffer->staged_backing = NULL;
  }
}

void iree_hal_local_transient_buffer_release_reservation(
    iree_hal_buffer_t* base_buffer,
    const iree_async_frontier_t* death_frontier) {
  iree_hal_local_transient_buffer_t* buffer =
      iree_hal_local_transient_buffer_cast(base_buffer);
  if (!buffer->reservation_pool) return;
  const int32_t was_armed = iree_atomic_exchange(&buffer->reservation_armed, 0,
                                                 iree_memory_order_acq_rel);
  if (was_armed) {
    iree_hal_pool_release_reservation(buffer->reservation_pool,
                                      &buffer->reservation, death_frontier);
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
      iree_hal_local_transient_buffer_load_committed(buffer);
  if (IREE_UNLIKELY(!committed)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "transient buffer has not been committed; ensure the alloca signal "
        "semaphores are satisfied before accessing it");
  }
  return iree_hal_local_transient_buffer_committed_vtable(committed)->map_range(
      committed, mapping_mode, memory_access, local_byte_offset,
      local_byte_length, mapping);
}

static iree_status_t iree_hal_local_transient_buffer_unmap_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length, iree_hal_buffer_mapping_t* mapping) {
  iree_hal_local_transient_buffer_t* buffer =
      iree_hal_local_transient_buffer_cast(base_buffer);
  iree_hal_buffer_t* committed =
      iree_hal_local_transient_buffer_load_committed(buffer);
  if (IREE_UNLIKELY(!committed)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "transient buffer has been decommitted");
  }
  return iree_hal_local_transient_buffer_committed_vtable(committed)
      ->unmap_range(committed, local_byte_offset, local_byte_length, mapping);
}

static iree_status_t iree_hal_local_transient_buffer_invalidate_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  iree_hal_local_transient_buffer_t* buffer =
      iree_hal_local_transient_buffer_cast(base_buffer);
  iree_hal_buffer_t* committed =
      iree_hal_local_transient_buffer_load_committed(buffer);
  if (IREE_UNLIKELY(!committed)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "transient buffer has been decommitted");
  }
  return iree_hal_local_transient_buffer_committed_vtable(committed)
      ->invalidate_range(committed, local_byte_offset, local_byte_length);
}

static iree_status_t iree_hal_local_transient_buffer_flush_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  iree_hal_local_transient_buffer_t* buffer =
      iree_hal_local_transient_buffer_cast(base_buffer);
  iree_hal_buffer_t* committed =
      iree_hal_local_transient_buffer_load_committed(buffer);
  if (IREE_UNLIKELY(!committed)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "transient buffer has been decommitted");
  }
  return iree_hal_local_transient_buffer_committed_vtable(committed)
      ->flush_range(committed, local_byte_offset, local_byte_length);
}

static const iree_hal_buffer_vtable_t iree_hal_local_transient_buffer_vtable = {
    .recycle = iree_hal_buffer_recycle,
    .destroy = iree_hal_local_transient_buffer_destroy,
    .map_range = iree_hal_local_transient_buffer_map_range,
    .unmap_range = iree_hal_local_transient_buffer_unmap_range,
    .invalidate_range = iree_hal_local_transient_buffer_invalidate_range,
    .flush_range = iree_hal_local_transient_buffer_flush_range,
};

// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/transient_buffer.h"

typedef struct iree_hal_amdgpu_transient_buffer_t {
  iree_hal_buffer_t base;
  iree_allocator_t host_allocator;

  // Provider-backed view staged for queue packet emission and future commit.
  // Retained while non-NULL.
  iree_hal_buffer_t* staged_backing;

  // Host-visible committed backing. NULL before alloca commit and after
  // dealloca decommit.
  //
  // The queue's semaphore edges provide the real ordering contract; the
  // acquire/release atomics here make host-side wrapper state transitions
  // data-race-free and visible to TSAN.
  iree_atomic_intptr_t committed_backing;

  // Borrowed source pool and one queue-owned reservation token.
  iree_hal_pool_t* reservation_pool;
  iree_hal_pool_reservation_t reservation;
  iree_atomic_int32_t reservation_armed;

  // Set when one dealloca has been accepted for this wrapper. This is
  // single-owner bookkeeping for reservation release/decommit, not a queue-use
  // lifetime validator; queue operation order is expressed by semaphores.
  iree_atomic_int32_t dealloca_queued;
} iree_hal_amdgpu_transient_buffer_t;

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

iree_status_t iree_hal_amdgpu_transient_buffer_create(
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

  iree_hal_amdgpu_transient_buffer_t* buffer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, sizeof(*buffer), (void**)&buffer));

  iree_hal_buffer_initialize(
      placement, /*allocated_buffer=*/&buffer->base, allocation_size,
      /*byte_offset=*/0, byte_length, params.type, params.access, params.usage,
      &iree_hal_amdgpu_transient_buffer_vtable, &buffer->base);
  buffer->host_allocator = host_allocator;
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
  iree_hal_buffer_retain(backing_buffer);
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
  if (!buffer->reservation_pool) return;
  const int32_t was_armed = iree_atomic_exchange(&buffer->reservation_armed, 0,
                                                 iree_memory_order_acq_rel);
  if (was_armed) {
    iree_hal_pool_release_reservation(buffer->reservation_pool,
                                      &buffer->reservation, death_frontier);
  }
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
  iree_allocator_t host_allocator = buffer->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdgpu_transient_buffer_decommit(base_buffer);
  iree_hal_amdgpu_transient_buffer_release_reservation(base_buffer,
                                                       /*death_frontier=*/NULL);

  iree_allocator_free(host_allocator, buffer);
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

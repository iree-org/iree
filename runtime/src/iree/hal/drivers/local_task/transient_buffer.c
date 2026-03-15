// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/local_task/transient_buffer.h"

// Vtable dispatch for forwarding to the committed buffer's implementation.
// Equivalent to IREE_HAL_VTABLE_DISPATCH from detail.h but accessible from
// driver code (detail.h is module-private to the HAL).
static inline const iree_hal_buffer_vtable_t*
iree_hal_task_transient_buffer_committed_vtable(iree_hal_buffer_t* buffer) {
  return (const iree_hal_buffer_vtable_t*)((const iree_hal_resource_t*)buffer)
      ->vtable;
}

//===----------------------------------------------------------------------===//
// iree_hal_task_transient_buffer_t
//===----------------------------------------------------------------------===//

struct iree_hal_task_transient_buffer_t {
  iree_hal_buffer_t base;
  iree_allocator_t host_allocator;

  // The committed backing buffer. NULL before commit, non-NULL after.
  //
  // Atomic for TSAN visibility: the commit happens on the queue drain thread
  // and the reads happen on worker threads processing downstream commands.
  // The semaphore signal between commit and use provides the real
  // happens-before ordering, but TSAN cannot observe cross-semaphore
  // relationships, so we use acquire/release atomics to make the ordering
  // explicit in a way TSAN can verify.
  iree_atomic_intptr_t committed;
};

static const iree_hal_buffer_vtable_t iree_hal_task_transient_buffer_vtable;

static iree_hal_task_transient_buffer_t* iree_hal_task_transient_buffer_cast(
    iree_hal_buffer_t* buffer) {
  return (iree_hal_task_transient_buffer_t*)buffer;
}

// Loads the committed backing buffer with acquire semantics.
// Returns NULL if the buffer has not been committed (or has been decommitted).
static iree_hal_buffer_t* iree_hal_task_transient_buffer_load_committed(
    iree_hal_task_transient_buffer_t* buffer) {
  return (iree_hal_buffer_t*)iree_atomic_load(&buffer->committed,
                                              iree_memory_order_acquire);
}

iree_status_t iree_hal_task_transient_buffer_create(
    iree_hal_buffer_placement_t placement, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_allocator_t host_allocator,
    iree_hal_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_task_transient_buffer_t* buffer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, sizeof(*buffer), (void**)&buffer));

  iree_hal_buffer_initialize(
      placement, /*allocated_buffer=*/&buffer->base, allocation_size,
      /*byte_offset=*/0,
      /*byte_length=*/allocation_size, params.type, params.access, params.usage,
      &iree_hal_task_transient_buffer_vtable, &buffer->base);
  buffer->host_allocator = host_allocator;
  iree_atomic_store(&buffer->committed, 0, iree_memory_order_relaxed);

  *out_buffer = &buffer->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

bool iree_hal_task_transient_buffer_isa(const iree_hal_buffer_t* buffer) {
  return iree_hal_resource_is(&buffer->resource,
                              &iree_hal_task_transient_buffer_vtable);
}

void iree_hal_task_transient_buffer_commit(iree_hal_buffer_t* base_buffer,
                                           iree_hal_buffer_t* backing) {
  iree_hal_task_transient_buffer_t* buffer =
      iree_hal_task_transient_buffer_cast(base_buffer);
  IREE_ASSERT_ARGUMENT(backing);

  // The buffer must not already be committed.
  IREE_ASSERT_TRUE(iree_hal_task_transient_buffer_load_committed(buffer) ==
                   NULL);

  // Sync the wrapper's metadata to match the actual allocated buffer. The
  // allocator may have adjusted memory type (e.g. added HOST_VISIBLE), access,
  // or usage beyond what the caller originally requested.
  base_buffer->memory_type = backing->memory_type;
  base_buffer->allowed_access = backing->allowed_access;
  base_buffer->allowed_usage = backing->allowed_usage;

  // Retain the backing buffer and store with release semantics.
  iree_hal_buffer_retain(backing);
  iree_atomic_store(&buffer->committed, (intptr_t)backing,
                    iree_memory_order_release);
}

void iree_hal_task_transient_buffer_decommit(iree_hal_buffer_t* base_buffer) {
  iree_hal_task_transient_buffer_t* buffer =
      iree_hal_task_transient_buffer_cast(base_buffer);
  iree_hal_buffer_t* committed = (iree_hal_buffer_t*)iree_atomic_exchange(
      &buffer->committed, 0, iree_memory_order_acq_rel);
  if (committed) {
    iree_hal_buffer_release(committed);
  }
}

//===----------------------------------------------------------------------===//
// iree_hal_buffer_t vtable
//===----------------------------------------------------------------------===//

static void iree_hal_task_transient_buffer_destroy(
    iree_hal_buffer_t* base_buffer) {
  iree_hal_task_transient_buffer_t* buffer =
      iree_hal_task_transient_buffer_cast(base_buffer);
  iree_allocator_t host_allocator = buffer->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Safety net: decommit if still committed. Normally dealloca handles this,
  // but if the buffer is released without a prior dealloca (e.g., error paths
  // or synchronous allocation fallback) we must not leak the backing.
  iree_hal_task_transient_buffer_decommit(base_buffer);

  iree_allocator_free(host_allocator, buffer);
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_task_transient_buffer_map_range(
    iree_hal_buffer_t* base_buffer, iree_hal_mapping_mode_t mapping_mode,
    iree_hal_memory_access_t memory_access,
    iree_device_size_t local_byte_offset, iree_device_size_t local_byte_length,
    iree_hal_buffer_mapping_t* mapping) {
  iree_hal_task_transient_buffer_t* buffer =
      iree_hal_task_transient_buffer_cast(base_buffer);
  iree_hal_buffer_t* committed =
      iree_hal_task_transient_buffer_load_committed(buffer);
  if (IREE_UNLIKELY(!committed)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "transient buffer has not been committed; ensure all alloca signal "
        "semaphores have been waited on before accessing the buffer");
  }
  // Forward directly to the committed buffer's vtable. Our base has already
  // validated the range and access so we skip redundant checks.
  return iree_hal_task_transient_buffer_committed_vtable(committed)->map_range(
      committed, mapping_mode, memory_access, local_byte_offset,
      local_byte_length, mapping);
}

static iree_status_t iree_hal_task_transient_buffer_unmap_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length, iree_hal_buffer_mapping_t* mapping) {
  iree_hal_task_transient_buffer_t* buffer =
      iree_hal_task_transient_buffer_cast(base_buffer);
  iree_hal_buffer_t* committed =
      iree_hal_task_transient_buffer_load_committed(buffer);
  if (IREE_UNLIKELY(!committed)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "transient buffer has been decommitted");
  }
  return iree_hal_task_transient_buffer_committed_vtable(committed)
      ->unmap_range(committed, local_byte_offset, local_byte_length, mapping);
}

static iree_status_t iree_hal_task_transient_buffer_invalidate_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  iree_hal_task_transient_buffer_t* buffer =
      iree_hal_task_transient_buffer_cast(base_buffer);
  iree_hal_buffer_t* committed =
      iree_hal_task_transient_buffer_load_committed(buffer);
  if (IREE_UNLIKELY(!committed)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "transient buffer has been decommitted");
  }
  return iree_hal_task_transient_buffer_committed_vtable(committed)
      ->invalidate_range(committed, local_byte_offset, local_byte_length);
}

static iree_status_t iree_hal_task_transient_buffer_flush_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  iree_hal_task_transient_buffer_t* buffer =
      iree_hal_task_transient_buffer_cast(base_buffer);
  iree_hal_buffer_t* committed =
      iree_hal_task_transient_buffer_load_committed(buffer);
  if (IREE_UNLIKELY(!committed)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "transient buffer has been decommitted");
  }
  return iree_hal_task_transient_buffer_committed_vtable(committed)
      ->flush_range(committed, local_byte_offset, local_byte_length);
}

static const iree_hal_buffer_vtable_t iree_hal_task_transient_buffer_vtable = {
    .recycle = iree_hal_buffer_recycle,
    .destroy = iree_hal_task_transient_buffer_destroy,
    .map_range = iree_hal_task_transient_buffer_map_range,
    .unmap_range = iree_hal_task_transient_buffer_unmap_range,
    .invalidate_range = iree_hal_task_transient_buffer_invalidate_range,
    .flush_range = iree_hal_task_transient_buffer_flush_range,
};

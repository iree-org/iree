// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/buffer.h"

#include "iree/hal/drivers/amdgpu/logical_device.h"
#include "iree/hal/drivers/amdgpu/transient_buffer.h"

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_buffer_t
//===----------------------------------------------------------------------===//

// A buffer backed by an HSA memory pool allocation.
// Device-local buffers are usually coarse-grained and unmappable; explicit
// host-visible/host-local buffers use fine-grained pools and can be mapped.
struct iree_hal_amdgpu_buffer_t {
  // Base HAL buffer resource returned to callers.
  iree_hal_buffer_t base;

  // Host allocator used to free unpooled wrapper storage.
  iree_allocator_t host_allocator;

  // Pool this wrapper returns to when its final reference is released.
  iree_hal_amdgpu_buffer_pool_t* pool;

  // Next wrapper in either the pool return stack or acquire-side cache.
  iree_hal_amdgpu_buffer_t* pool_next;

  // Unowned libhsa handle for freeing the allocation on destroy.
  const iree_hal_amdgpu_libhsa_t* libhsa;

  // HSA-allocated pointer. Accessible from both host and device when allocated
  // from a fine-grained pool, or device-only from a coarse-grained pool.
  void* host_ptr;

  // Optional callback for provider/pool-owned buffer storage.
  // When present the callback owns release of |host_ptr| and any backing pool
  // bookkeeping. When null this buffer frees |host_ptr| directly with HSA.
  iree_hal_buffer_release_callback_t release_callback;

  // Session-local profiling allocation id for direct allocator buffers.
  uint64_t profile_allocation_id;

  // Profiling session id owning |profile_allocation_id|.
  uint64_t profile_session_id;

  // Producer-defined memory pool id used for profiling events.
  uint64_t profile_pool_id;

  // Physical device ordinal used for profiling allocation/free events.
  uint32_t profile_physical_device_ordinal;

  // Byte alignment used for profiling allocation/free events.
  iree_device_size_t profile_alignment;
};

static const iree_hal_buffer_vtable_t iree_hal_amdgpu_buffer_vtable;

static iree_hal_amdgpu_buffer_t* iree_hal_amdgpu_buffer_cast(
    iree_hal_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_amdgpu_buffer_vtable);
  return (iree_hal_amdgpu_buffer_t*)base_value;
}

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_buffer_pool_t
//===----------------------------------------------------------------------===//

static iree_host_size_t iree_hal_amdgpu_buffer_pool_slot_size(void) {
  return iree_host_align(sizeof(iree_hal_amdgpu_buffer_t),
                         iree_alignof(iree_hal_amdgpu_buffer_t));
}

static iree_status_t iree_hal_amdgpu_buffer_pool_grow_locked(
    iree_hal_amdgpu_buffer_pool_t* pool) {
  IREE_TRACE_ZONE_BEGIN(z0);

  const iree_host_size_t slot_size = iree_hal_amdgpu_buffer_pool_slot_size();
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
    iree_hal_amdgpu_buffer_t* buffer = (iree_hal_amdgpu_buffer_t*)slot_ptr;
    buffer->pool = pool;
    buffer->pool_next = pool->acquire_head;
    pool->acquire_head = buffer;
    slot_ptr += slot_size;
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_buffer_pool_initialize(
    iree_arena_block_pool_t* block_pool,
    iree_hal_amdgpu_buffer_pool_t* out_pool) {
  IREE_ASSERT_ARGUMENT(block_pool);
  IREE_ASSERT_ARGUMENT(out_pool);
  IREE_TRACE_ZONE_BEGIN(z0);

  memset(out_pool, 0, sizeof(*out_pool));
  const iree_host_size_t slot_size = iree_hal_amdgpu_buffer_pool_slot_size();
  if (IREE_UNLIKELY(block_pool->usable_block_size < slot_size)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU buffer pool block usable size %" PRIhsz
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

void iree_hal_amdgpu_buffer_pool_deinitialize(
    iree_hal_amdgpu_buffer_pool_t* pool) {
  if (!pool || !pool->block_pool) return;
  IREE_TRACE_ZONE_BEGIN(z0);

#if !defined(NDEBUG)
  const int32_t live_count =
      iree_atomic_load(&pool->live_count, iree_memory_order_acquire);
  IREE_ASSERT(live_count == 0,
              "deinitializing AMDGPU buffer pool with %d live wrappers",
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

static iree_status_t iree_hal_amdgpu_buffer_pool_acquire(
    iree_hal_amdgpu_buffer_pool_t* pool,
    iree_hal_amdgpu_buffer_t** out_buffer) {
  *out_buffer = NULL;

  iree_slim_mutex_lock(&pool->mutex);

  iree_status_t status = iree_ok_status();
  iree_hal_amdgpu_buffer_t* buffer = pool->acquire_head;
  if (buffer) {
    pool->acquire_head = buffer->pool_next;
  } else {
    buffer = (iree_hal_amdgpu_buffer_t*)iree_atomic_exchange(
        &pool->return_head, 0, iree_memory_order_acquire);
    if (buffer) {
      pool->acquire_head = buffer->pool_next;
    } else {
      status = iree_hal_amdgpu_buffer_pool_grow_locked(pool);
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

static void iree_hal_amdgpu_buffer_pool_release(
    iree_hal_amdgpu_buffer_pool_t* pool, iree_hal_amdgpu_buffer_t* buffer) {
#if !defined(NDEBUG)
  const int32_t old_live_count =
      iree_atomic_fetch_sub(&pool->live_count, 1, iree_memory_order_acq_rel);
  IREE_ASSERT(old_live_count > 0,
              "releasing AMDGPU buffer wrapper with no live wrapper count");
#endif  // !defined(NDEBUG)

  intptr_t expected = 0;
  do {
    expected = iree_atomic_load(&pool->return_head, iree_memory_order_relaxed);
    buffer->pool_next = (iree_hal_amdgpu_buffer_t*)expected;
  } while (!iree_atomic_compare_exchange_weak(
      &pool->return_head, &expected, (intptr_t)buffer,
      iree_memory_order_release, iree_memory_order_relaxed));
}

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_buffer_t
//===----------------------------------------------------------------------===//

void* iree_hal_amdgpu_buffer_device_pointer(iree_hal_buffer_t* base_buffer) {
  if (!iree_hal_resource_is((const iree_hal_resource_t*)base_buffer,
                            &iree_hal_amdgpu_buffer_vtable)) {
    if (iree_hal_amdgpu_transient_buffer_isa(base_buffer)) {
      iree_hal_buffer_t* backing_buffer =
          iree_hal_amdgpu_transient_buffer_backing_buffer(base_buffer);
      if (!backing_buffer) return NULL;
      return iree_hal_amdgpu_buffer_device_pointer(backing_buffer);
    }
    return NULL;
  }
  return ((iree_hal_amdgpu_buffer_t*)base_buffer)->host_ptr;
}

static void iree_hal_amdgpu_buffer_initialize(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    iree_hal_buffer_placement_t placement, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_device_size_t allocation_size,
    iree_device_size_t byte_length, void* host_ptr,
    iree_hal_buffer_release_callback_t release_callback,
    iree_hal_amdgpu_buffer_pool_t* pool, iree_allocator_t host_allocator,
    iree_hal_amdgpu_buffer_t* out_buffer) {
  iree_hal_buffer_initialize(placement, &out_buffer->base, allocation_size,
                             /*byte_offset=*/0, byte_length, memory_type,
                             allowed_access, allowed_usage,
                             &iree_hal_amdgpu_buffer_vtable, &out_buffer->base);
  out_buffer->host_allocator = host_allocator;
  out_buffer->pool = pool;
  out_buffer->pool_next = NULL;
  out_buffer->libhsa = libhsa;
  out_buffer->host_ptr = host_ptr;
  out_buffer->release_callback = release_callback;
  out_buffer->profile_allocation_id = 0;
  out_buffer->profile_session_id = 0;
  out_buffer->profile_pool_id = 0;
  out_buffer->profile_physical_device_ordinal = UINT32_MAX;
  out_buffer->profile_alignment = 0;
}

iree_status_t iree_hal_amdgpu_buffer_create(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    iree_hal_buffer_placement_t placement, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_device_size_t allocation_size,
    iree_device_size_t byte_length, void* host_ptr,
    iree_hal_buffer_release_callback_t release_callback,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(out_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_buffer = NULL;

  iree_hal_amdgpu_buffer_t* buffer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, sizeof(*buffer), (void**)&buffer));
  iree_hal_amdgpu_buffer_initialize(
      libhsa, placement, memory_type, allowed_access, allowed_usage,
      allocation_size, byte_length, host_ptr, release_callback, /*pool=*/NULL,
      host_allocator, buffer);

  *out_buffer = &buffer->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_buffer_create_pooled(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    iree_hal_buffer_placement_t placement, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_device_size_t allocation_size,
    iree_device_size_t byte_length, void* host_ptr,
    iree_hal_buffer_release_callback_t release_callback,
    iree_hal_amdgpu_buffer_pool_t* pool, iree_allocator_t host_allocator,
    iree_hal_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(pool);
  IREE_ASSERT_ARGUMENT(out_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_buffer = NULL;

  iree_hal_amdgpu_buffer_t* buffer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdgpu_buffer_pool_acquire(pool, &buffer));
  iree_hal_amdgpu_buffer_initialize(
      libhsa, placement, memory_type, allowed_access, allowed_usage,
      allocation_size, byte_length, host_ptr, release_callback, pool,
      host_allocator, buffer);

  *out_buffer = &buffer->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

void iree_hal_amdgpu_buffer_set_profile_allocation(
    iree_hal_buffer_t* base_buffer, uint64_t session_id, uint64_t allocation_id,
    uint64_t pool_id, uint32_t physical_device_ordinal,
    iree_device_size_t alignment) {
  iree_hal_amdgpu_buffer_t* buffer = iree_hal_amdgpu_buffer_cast(base_buffer);
  buffer->profile_allocation_id = allocation_id;
  buffer->profile_session_id = session_id;
  buffer->profile_pool_id = pool_id;
  buffer->profile_physical_device_ordinal = physical_device_ordinal;
  buffer->profile_alignment = alignment;
}

static void iree_hal_amdgpu_buffer_destroy(iree_hal_buffer_t* base_buffer) {
  iree_hal_amdgpu_buffer_t* buffer = iree_hal_amdgpu_buffer_cast(base_buffer);
  iree_allocator_t host_allocator = buffer->host_allocator;
  iree_hal_amdgpu_buffer_pool_t* pool = buffer->pool;
  IREE_TRACE_ZONE_BEGIN(z0);

  if (buffer->profile_allocation_id != 0 && base_buffer->placement.device) {
    iree_hal_profile_memory_event_t event =
        iree_hal_profile_memory_event_default();
    event.type = IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_BUFFER_FREE;
    event.allocation_id = buffer->profile_allocation_id;
    event.pool_id = buffer->profile_pool_id;
    event.backing_id = (uint64_t)(uintptr_t)buffer->host_ptr;
    event.physical_device_ordinal = buffer->profile_physical_device_ordinal;
    event.memory_type = base_buffer->memory_type;
    event.buffer_usage = base_buffer->allowed_usage;
    event.length = base_buffer->allocation_size;
    event.alignment = buffer->profile_alignment;
    iree_hal_amdgpu_logical_device_record_profile_memory_event_for_session(
        base_buffer->placement.device, buffer->profile_session_id, &event);
  }

  if (buffer->release_callback.fn) {
    buffer->release_callback.fn(buffer->release_callback.user_data,
                                base_buffer);
  } else if (buffer->host_ptr) {
    iree_hal_amdgpu_hsa_cleanup_assert_success(
        iree_hsa_amd_memory_pool_free_raw(buffer->libhsa, buffer->host_ptr));
  }

  buffer->libhsa = NULL;
  buffer->host_ptr = NULL;
  buffer->release_callback = iree_hal_buffer_release_callback_null();
  buffer->profile_allocation_id = 0;
  buffer->profile_session_id = 0;
  buffer->profile_pool_id = 0;
  buffer->profile_physical_device_ordinal = UINT32_MAX;
  buffer->profile_alignment = 0;
  if (pool) {
    iree_hal_amdgpu_buffer_pool_release(pool, buffer);
  } else {
    iree_allocator_free(host_allocator, buffer);
  }

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_amdgpu_buffer_map_range(
    iree_hal_buffer_t* base_buffer, iree_hal_mapping_mode_t mapping_mode,
    iree_hal_memory_access_t memory_access,
    iree_device_size_t local_byte_offset, iree_device_size_t local_byte_length,
    iree_hal_buffer_mapping_t* mapping) {
  iree_hal_amdgpu_buffer_t* buffer = iree_hal_amdgpu_buffer_cast(base_buffer);

  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_memory_type(
      iree_hal_buffer_memory_type(base_buffer),
      IREE_HAL_MEMORY_TYPE_HOST_VISIBLE));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(base_buffer),
      mapping_mode == IREE_HAL_MAPPING_MODE_PERSISTENT
          ? IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT
          : IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED));

  // Host-visible AMDGPU HSA allocations are directly host-accessible.
  mapping->contents = iree_make_byte_span(
      (uint8_t*)buffer->host_ptr + local_byte_offset, local_byte_length);

  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_buffer_unmap_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length, iree_hal_buffer_mapping_t* mapping) {
  // Nothing to do — all host-visible AMDGPU allocations are currently coherent.
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_buffer_invalidate_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  // Nothing to do — all host-visible AMDGPU allocations are currently coherent.
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_buffer_flush_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  // Nothing to do — all host-visible AMDGPU allocations are currently coherent.
  return iree_ok_status();
}

static const iree_hal_buffer_vtable_t iree_hal_amdgpu_buffer_vtable = {
    .recycle = iree_hal_buffer_recycle,
    .destroy = iree_hal_amdgpu_buffer_destroy,
    .map_range = iree_hal_amdgpu_buffer_map_range,
    .unmap_range = iree_hal_amdgpu_buffer_unmap_range,
    .invalidate_range = iree_hal_amdgpu_buffer_invalidate_range,
    .flush_range = iree_hal_amdgpu_buffer_flush_range,
};

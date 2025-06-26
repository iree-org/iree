// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/buffer.h"

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_external_buffer_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_amdgpu_external_buffer_t {
  iree_hal_buffer_t base;
  iree_allocator_t host_allocator;
  iree_hal_buffer_release_callback_t release_callback;
  uint64_t device_ptr;
} iree_hal_amdgpu_external_buffer_t;

static const iree_hal_buffer_vtable_t iree_hal_amdgpu_external_buffer_vtable;

static iree_hal_amdgpu_external_buffer_t* iree_hal_amdgpu_external_buffer_cast(
    iree_hal_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_amdgpu_external_buffer_vtable);
  return (iree_hal_amdgpu_external_buffer_t*)base_value;
}

static const iree_hal_amdgpu_external_buffer_t*
iree_hal_amdgpu_external_buffer_const_cast(
    const iree_hal_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_amdgpu_external_buffer_vtable);
  return (const iree_hal_amdgpu_external_buffer_t*)base_value;
}

iree_status_t iree_hal_amdgpu_external_buffer_wrap(
    iree_hal_buffer_placement_t placement, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_device_size_t allocation_size,
    iree_device_size_t byte_offset, iree_device_size_t byte_length,
    uint64_t device_ptr, iree_hal_buffer_release_callback_t release_callback,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(device_ptr);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdgpu_external_buffer_t* buffer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, sizeof(*buffer), (void**)&buffer));
  iree_hal_buffer_initialize(
      placement, &buffer->base, allocation_size, byte_offset, byte_length,
      memory_type, allowed_access, allowed_usage,
      &iree_hal_amdgpu_external_buffer_vtable, &buffer->base);
  buffer->host_allocator = host_allocator;
  buffer->release_callback = release_callback;
  buffer->device_ptr = device_ptr;

  *out_buffer = &buffer->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_amdgpu_external_buffer_destroy(
    iree_hal_buffer_t* base_buffer) {
  iree_hal_amdgpu_external_buffer_t* buffer =
      iree_hal_amdgpu_external_buffer_cast(base_buffer);
  iree_allocator_t host_allocator = buffer->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Optionally call a release callback when the buffer is destroyed. Not all
  // implementations may require this but it's cheap and provides additional
  // flexibility.
  if (buffer->release_callback.fn) {
    buffer->release_callback.fn(buffer->release_callback.user_data,
                                base_buffer);
  }

  iree_allocator_free(host_allocator, buffer);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_amdgpu_external_buffer_map_range(
    iree_hal_buffer_t* base_buffer, iree_hal_mapping_mode_t mapping_mode,
    iree_hal_memory_access_t memory_access,
    iree_device_size_t local_byte_offset, iree_device_size_t local_byte_length,
    iree_hal_buffer_mapping_t* mapping) {
  iree_hal_amdgpu_external_buffer_t* buffer =
      iree_hal_amdgpu_external_buffer_cast(base_buffer);

  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_memory_type(
      iree_hal_buffer_memory_type(base_buffer),
      IREE_HAL_MEMORY_TYPE_HOST_VISIBLE));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(base_buffer),
      mapping_mode == IREE_HAL_MAPPING_MODE_PERSISTENT
          ? IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT
          : IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED));

  mapping->contents = iree_make_byte_span(
      (IREE_AMDGPU_DEVICE_PTR void*)buffer->device_ptr, local_byte_length);

  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_external_buffer_unmap_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length, iree_hal_buffer_mapping_t* mapping) {
  // Nothing to do today (though maybe we may want to flush?).
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_external_buffer_invalidate_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  // TODO(benvanik): anything we need to do to invalidate?
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_external_buffer_flush_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  // TODO(benvanik): anything we need to do to flush?
  return iree_ok_status();
}

static const iree_hal_buffer_vtable_t iree_hal_amdgpu_external_buffer_vtable = {
    .recycle = iree_hal_buffer_recycle,
    .destroy = iree_hal_amdgpu_external_buffer_destroy,
    .map_range = iree_hal_amdgpu_external_buffer_map_range,
    .unmap_range = iree_hal_amdgpu_external_buffer_unmap_range,
    .invalidate_range = iree_hal_amdgpu_external_buffer_invalidate_range,
    .flush_range = iree_hal_amdgpu_external_buffer_flush_range,
};

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_transient_buffer_t
//===----------------------------------------------------------------------===//

static const iree_hal_buffer_vtable_t iree_hal_amdgpu_transient_buffer_vtable;

void iree_hal_amdgpu_transient_buffer_initialize(
    iree_hal_buffer_placement_t placement,
    iree_hal_amdgpu_device_allocation_handle_t* handle,
    iree_hal_buffer_release_callback_t release_callback,
    iree_hal_amdgpu_transient_buffer_t* out_buffer) {
  IREE_ASSERT_ARGUMENT(handle);
  IREE_ASSERT_ARGUMENT(out_buffer);

  memset(out_buffer, 0, sizeof(*out_buffer));

  // Transient buffers *are* asynchronous buffers.
  placement.flags |= IREE_HAL_BUFFER_PLACEMENT_FLAG_ASYNCHRONOUS;

  iree_hal_buffer_initialize(
      placement, &out_buffer->base,
      /*allocation_size=*/0,
      /*byte_offset=*/0, /*byte_length=*/0, /*memory_type=*/0,
      /*allowed_access=*/0, /*allowed_usage=*/0,
      &iree_hal_amdgpu_transient_buffer_vtable, &out_buffer->base);
  out_buffer->handle = handle;
  out_buffer->release_callback = release_callback;

  // NOTE: transient buffers start with 0 references as they are pooled.
  iree_atomic_ref_count_init_value(&out_buffer->base.resource.ref_count, 0);
}

void iree_hal_amdgpu_transient_buffer_deinitialize(
    iree_hal_amdgpu_transient_buffer_t* buffer) {
  // No-op.
}

static iree_hal_amdgpu_transient_buffer_t*
iree_hal_amdgpu_transient_buffer_cast(iree_hal_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_amdgpu_transient_buffer_vtable);
  return (iree_hal_amdgpu_transient_buffer_t*)base_value;
}

static const iree_hal_amdgpu_transient_buffer_t*
iree_hal_amdgpu_transient_buffer_const_cast(
    const iree_hal_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_amdgpu_transient_buffer_vtable);
  return (const iree_hal_amdgpu_transient_buffer_t*)base_value;
}

// Returns true if |buffer| is an iree_hal_amdgpu_transient_buffer_t.
static bool iree_hal_amdgpu_transient_buffer_isa(iree_hal_buffer_t* buffer) {
  return iree_hal_resource_is(buffer, &iree_hal_amdgpu_transient_buffer_vtable);
}

static void iree_hal_amdgpu_transient_buffer_recycle(
    iree_hal_buffer_t* base_buffer) {
  if (IREE_UNLIKELY(!base_buffer)) return;
  iree_hal_amdgpu_transient_buffer_t* buffer =
      iree_hal_amdgpu_transient_buffer_cast(base_buffer);
  IREE_ASSERT(buffer->release_callback.fn);
  if (buffer->release_callback.fn) {
    buffer->release_callback.fn(buffer->release_callback.user_data,
                                base_buffer);
  }
}

void iree_hal_amdgpu_transient_buffer_reset(
    iree_hal_amdgpu_transient_buffer_t* buffer, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_device_size_t byte_offset,
    iree_device_size_t byte_length) {
  buffer->base.memory_type = params.type;
  buffer->base.allowed_access = params.access;
  buffer->base.allowed_usage = params.usage;
  buffer->base.allocation_size = allocation_size;
  buffer->base.byte_offset = byte_offset;
  buffer->base.byte_length = byte_length;
  buffer->base.placement.queue_affinity = params.queue_affinity
                                              ? params.queue_affinity
                                              : IREE_HAL_QUEUE_AFFINITY_ANY;
}

// Returns the device pointer from the allocation handle if it is currently
// allocated. This may read from device memory and is only valid when the buffer
// is allocated and the completion signals have propagated (ordering the
// writes).
static iree_status_t iree_hal_amdgpu_transient_buffer_ptr(
    iree_hal_amdgpu_transient_buffer_t* buffer, void** out_ptr) {
  static_assert(sizeof(void*) == sizeof(iree_atomic_uint64_t),
                "only 64-bit pointers are supported");
  IREE_AMDGPU_DEVICE_PTR void* ptr = (void*)iree_atomic_load(
      (iree_atomic_uint64_t*)&buffer->handle->ptr, iree_memory_order_acquire);
  if (!ptr) {
    *out_ptr = NULL;
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "transient buffer has no backing allocation at this time; host usage "
        "is only valid once an alloca has signaled completion and prior to "
        "enqueuing any dealloca");
  }
  *out_ptr = ptr;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_transient_buffer_map_range(
    iree_hal_buffer_t* base_buffer, iree_hal_mapping_mode_t mapping_mode,
    iree_hal_memory_access_t memory_access,
    iree_device_size_t local_byte_offset, iree_device_size_t local_byte_length,
    iree_hal_buffer_mapping_t* mapping) {
  iree_hal_amdgpu_transient_buffer_t* buffer =
      iree_hal_amdgpu_transient_buffer_cast(base_buffer);

  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_memory_type(
      iree_hal_buffer_memory_type(base_buffer),
      IREE_HAL_MEMORY_TYPE_HOST_VISIBLE));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(base_buffer),
      mapping_mode == IREE_HAL_MAPPING_MODE_PERSISTENT
          ? IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT
          : IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED));

  // Resolve device pointer (if available).
  IREE_AMDGPU_DEVICE_PTR void* ptr = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_transient_buffer_ptr(buffer, &ptr));

  mapping->contents = iree_make_byte_span(ptr, local_byte_length);

  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_transient_buffer_unmap_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length, iree_hal_buffer_mapping_t* mapping) {
  // Nothing to do today (though maybe we may want to flush?).
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_transient_buffer_invalidate_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  // TODO(benvanik): anything we need to do to invalidate?
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_transient_buffer_flush_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  // TODO(benvanik): anything we need to do to flush?
  return iree_ok_status();
}

static const iree_hal_buffer_vtable_t iree_hal_amdgpu_transient_buffer_vtable =
    {
        .recycle = iree_hal_amdgpu_transient_buffer_recycle,
        .destroy = iree_hal_amdgpu_transient_buffer_recycle,
        .map_range = iree_hal_amdgpu_transient_buffer_map_range,
        .unmap_range = iree_hal_amdgpu_transient_buffer_unmap_range,
        .invalidate_range = iree_hal_amdgpu_transient_buffer_invalidate_range,
        .flush_range = iree_hal_amdgpu_transient_buffer_flush_range,
};

//===----------------------------------------------------------------------===//
// Buffer Resolution
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_amdgpu_resolve_buffer(
    iree_hal_buffer_t* base_buffer,
    iree_hal_amdgpu_device_buffer_type_t* out_type, uint64_t* out_bits) {
  if (iree_hal_resource_is(base_buffer,
                           &iree_hal_amdgpu_transient_buffer_vtable)) {
    *out_type = IREE_HAL_AMDGPU_DEVICE_BUFFER_TYPE_HANDLE;
    *out_bits =
        (uint64_t)((iree_hal_amdgpu_transient_buffer_t*)base_buffer)->handle;
    return iree_ok_status();
  } else if (iree_hal_resource_is(base_buffer,
                                  &iree_hal_amdgpu_external_buffer_vtable)) {
    *out_type = IREE_HAL_AMDGPU_DEVICE_BUFFER_TYPE_PTR;
    *out_bits =
        (uint64_t)((iree_hal_amdgpu_external_buffer_t*)base_buffer)->device_ptr;
    return iree_ok_status();
  } else {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "unsupported buffer type; expected something known to the AMDGPU HAL");
  }
}

iree_status_t iree_hal_amdgpu_resolve_transient_buffer(
    iree_hal_buffer_t* base_buffer,
    iree_hal_amdgpu_device_allocation_handle_t** out_handle) {
  if (!iree_hal_resource_is(base_buffer,
                            &iree_hal_amdgpu_transient_buffer_vtable)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "provided buffer is not a transient allocation; only buffers allocated "
        "with iree_hal_device_queue_alloca can be deallocated using "
        "iree_hal_device_queue_dealloca");
  }
  iree_hal_amdgpu_transient_buffer_t* buffer =
      (iree_hal_amdgpu_transient_buffer_t*)base_buffer;
  *out_handle = buffer->handle;
  return iree_ok_status();
}

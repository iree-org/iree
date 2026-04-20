// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/replay/recorder_buffer.h"

#include <string.h>

#include "iree/hal/replay/recorder_record.h"

#define IREE_HAL_REPLAY_VTABLE_DISPATCH(resource, type_prefix, method_name) \
  ((const type_prefix##_vtable_t*)((const iree_hal_resource_t*)(resource))  \
       ->vtable)                                                            \
      ->method_name

//===----------------------------------------------------------------------===//
// iree_hal_replay_recorder_buffer_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_replay_recorder_buffer_t {
  // HAL buffer header for the recording wrapper buffer.
  iree_hal_buffer_t base;
  // Host allocator used for wrapper lifetime.
  iree_allocator_t host_allocator;
  // Shared recorder receiving all captured operations.
  iree_hal_replay_recorder_t* recorder;
  // Underlying buffer receiving forwarded HAL calls.
  iree_hal_buffer_t* base_buffer;
  // Session-local device object id associated with this buffer.
  iree_hal_replay_object_id_t device_id;
  // Session-local object id assigned to this buffer.
  iree_hal_replay_object_id_t buffer_id;
} iree_hal_replay_recorder_buffer_t;

static const iree_hal_buffer_vtable_t iree_hal_replay_recorder_buffer_vtable;

static bool iree_hal_replay_recorder_buffer_isa(
    iree_hal_buffer_t* base_buffer) {
  return iree_hal_resource_is(base_buffer,
                              &iree_hal_replay_recorder_buffer_vtable);
}

static iree_hal_replay_recorder_buffer_t* iree_hal_replay_recorder_buffer_cast(
    iree_hal_buffer_t* base_buffer) {
  IREE_HAL_ASSERT_TYPE(base_buffer, &iree_hal_replay_recorder_buffer_vtable);
  return (iree_hal_replay_recorder_buffer_t*)base_buffer;
}

void iree_hal_replay_recorder_buffer_make_object_payload(
    iree_hal_buffer_t* base_buffer,
    iree_hal_replay_buffer_object_payload_t* out_payload) {
  memset(out_payload, 0, sizeof(*out_payload));
  const iree_hal_buffer_placement_t placement =
      iree_hal_buffer_allocation_placement(base_buffer);
  out_payload->allocation_size = iree_hal_buffer_allocation_size(base_buffer);
  out_payload->byte_offset = iree_hal_buffer_byte_offset(base_buffer);
  out_payload->byte_length = iree_hal_buffer_byte_length(base_buffer);
  out_payload->queue_affinity = placement.queue_affinity;
  out_payload->placement_flags = placement.flags;
  out_payload->memory_type = iree_hal_buffer_memory_type(base_buffer);
  out_payload->allowed_access = iree_hal_buffer_allowed_access(base_buffer);
  out_payload->allowed_usage = iree_hal_buffer_allowed_usage(base_buffer);
}

iree_status_t iree_hal_replay_recorder_buffer_create_proxy(
    iree_hal_replay_recorder_t* recorder, iree_hal_replay_object_id_t device_id,
    iree_hal_replay_object_id_t buffer_id, iree_hal_device_t* placement_device,
    iree_hal_buffer_t* base_buffer, iree_allocator_t host_allocator,
    iree_hal_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(recorder);
  IREE_ASSERT_ARGUMENT(base_buffer);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;

  if (iree_hal_replay_recorder_buffer_isa(base_buffer)) {
    iree_hal_buffer_retain(base_buffer);
    *out_buffer = base_buffer;
    return iree_ok_status();
  }

  iree_hal_replay_recorder_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*buffer), (void**)&buffer));
  memset(buffer, 0, sizeof(*buffer));

  iree_hal_buffer_placement_t placement =
      iree_hal_buffer_allocation_placement(base_buffer);
  if (placement.device) placement.device = placement_device;
  iree_hal_buffer_initialize(
      placement, &buffer->base, iree_hal_buffer_allocation_size(base_buffer),
      iree_hal_buffer_byte_offset(base_buffer),
      iree_hal_buffer_byte_length(base_buffer),
      iree_hal_buffer_memory_type(base_buffer),
      iree_hal_buffer_allowed_access(base_buffer),
      iree_hal_buffer_allowed_usage(base_buffer),
      &iree_hal_replay_recorder_buffer_vtable, &buffer->base);
  buffer->host_allocator = host_allocator;
  buffer->recorder = recorder;
  iree_hal_replay_recorder_retain(buffer->recorder);
  buffer->base_buffer = base_buffer;
  iree_hal_buffer_retain(buffer->base_buffer);
  buffer->device_id = device_id;
  buffer->buffer_id = buffer_id;

  *out_buffer = &buffer->base;
  return iree_ok_status();
}

iree_hal_buffer_t* iree_hal_replay_recorder_buffer_base_or_self(
    iree_hal_buffer_t* buffer) {
  return iree_hal_replay_recorder_buffer_isa(buffer)
             ? iree_hal_replay_recorder_buffer_cast(buffer)->base_buffer
             : buffer;
}

iree_status_t iree_hal_replay_recorder_buffer_unwrap_for_call(
    iree_hal_buffer_t* buffer, iree_allocator_t host_allocator,
    iree_hal_buffer_t** out_base_buffer,
    iree_hal_buffer_t** out_temporary_buffer) {
  IREE_ASSERT_ARGUMENT(buffer);
  IREE_ASSERT_ARGUMENT(out_base_buffer);
  IREE_ASSERT_ARGUMENT(out_temporary_buffer);
  *out_base_buffer = NULL;
  *out_temporary_buffer = NULL;

  if (iree_hal_replay_recorder_buffer_isa(buffer)) {
    *out_base_buffer =
        iree_hal_replay_recorder_buffer_cast(buffer)->base_buffer;
    return iree_ok_status();
  }

  iree_hal_buffer_t* allocated_buffer =
      iree_hal_buffer_allocated_buffer(buffer);
  if (!iree_hal_replay_recorder_buffer_isa(allocated_buffer)) {
    *out_base_buffer = buffer;
    return iree_ok_status();
  }

  iree_hal_replay_recorder_buffer_t* replay_allocated_buffer =
      iree_hal_replay_recorder_buffer_cast(allocated_buffer);
  iree_hal_buffer_t* base_allocated_buffer =
      iree_hal_buffer_allocated_buffer(replay_allocated_buffer->base_buffer);
  IREE_RETURN_IF_ERROR(iree_hal_buffer_subspan(
      base_allocated_buffer, iree_hal_buffer_byte_offset(buffer),
      iree_hal_buffer_byte_length(buffer), host_allocator,
      out_temporary_buffer));
  *out_base_buffer = *out_temporary_buffer;
  return iree_ok_status();
}

void iree_hal_replay_recorder_buffer_release_temporary(
    iree_hal_buffer_t* temporary_buffer) {
  iree_hal_buffer_release(temporary_buffer);
}

static iree_status_t iree_hal_replay_recorder_buffer_begin_operation(
    iree_hal_replay_recorder_buffer_t* buffer,
    iree_hal_replay_operation_code_t operation_code,
    iree_hal_replay_payload_type_t payload_type,
    iree_hal_replay_pending_record_t* out_pending_record) {
  return iree_hal_replay_recorder_begin_operation(
      buffer->recorder, buffer->device_id, buffer->buffer_id,
      IREE_HAL_REPLAY_OBJECT_ID_NONE, IREE_HAL_REPLAY_OBJECT_TYPE_BUFFER,
      operation_code, payload_type, out_pending_record);
}

static void iree_hal_replay_recorder_buffer_destroy(
    iree_hal_buffer_t* base_buffer) {
  iree_hal_replay_recorder_buffer_t* buffer =
      iree_hal_replay_recorder_buffer_cast(base_buffer);
  iree_allocator_t host_allocator = buffer->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_buffer_release(buffer->base_buffer);
  iree_hal_replay_recorder_release(buffer->recorder);
  iree_allocator_free(host_allocator, buffer);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_replay_recorder_buffer_map_range(
    iree_hal_buffer_t* base_buffer, iree_hal_mapping_mode_t mapping_mode,
    iree_hal_memory_access_t memory_access,
    iree_device_size_t local_byte_offset, iree_device_size_t local_byte_length,
    iree_hal_buffer_mapping_t* mapping) {
  iree_hal_replay_recorder_buffer_t* buffer =
      iree_hal_replay_recorder_buffer_cast(base_buffer);
  iree_hal_replay_buffer_range_payload_t payload = {
      .byte_offset = local_byte_offset,
      .byte_length = local_byte_length,
      .mapping_mode = mapping_mode,
      .memory_access = memory_access,
  };
  iree_const_byte_span_t iovec =
      iree_make_const_byte_span((const uint8_t*)&payload, sizeof(payload));
  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_buffer_begin_operation(
      buffer, IREE_HAL_REPLAY_OPERATION_CODE_BUFFER_MAP_RANGE,
      IREE_HAL_REPLAY_PAYLOAD_TYPE_BUFFER_RANGE, &pending_record));
  iree_status_t status = IREE_HAL_REPLAY_VTABLE_DISPATCH(
      buffer->base_buffer, iree_hal_buffer, map_range)(
      buffer->base_buffer, mapping_mode, memory_access, local_byte_offset,
      local_byte_length, mapping);
  return iree_hal_replay_recorder_end_operation_with_payload(&pending_record,
                                                             status, 1, &iovec);
}

static iree_status_t iree_hal_replay_recorder_buffer_unmap_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length, iree_hal_buffer_mapping_t* mapping) {
  iree_hal_replay_recorder_buffer_t* buffer =
      iree_hal_replay_recorder_buffer_cast(base_buffer);
  iree_hal_replay_buffer_range_payload_t payload = {
      .byte_offset = local_byte_offset,
      .byte_length = local_byte_length,
      .memory_access = mapping->impl.allowed_access,
  };
  iree_const_byte_span_t iovec =
      iree_make_const_byte_span((const uint8_t*)&payload, sizeof(payload));
  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_buffer_begin_operation(
      buffer, IREE_HAL_REPLAY_OPERATION_CODE_BUFFER_UNMAP_RANGE,
      IREE_HAL_REPLAY_PAYLOAD_TYPE_BUFFER_RANGE, &pending_record));
  iree_status_t status = IREE_HAL_REPLAY_VTABLE_DISPATCH(
      buffer->base_buffer, iree_hal_buffer, unmap_range)(
      buffer->base_buffer, local_byte_offset, local_byte_length, mapping);
  return iree_hal_replay_recorder_end_operation_with_payload(&pending_record,
                                                             status, 1, &iovec);
}

static iree_status_t iree_hal_replay_recorder_buffer_invalidate_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  iree_hal_replay_recorder_buffer_t* buffer =
      iree_hal_replay_recorder_buffer_cast(base_buffer);
  iree_hal_replay_buffer_range_payload_t payload = {
      .byte_offset = local_byte_offset,
      .byte_length = local_byte_length,
  };
  iree_const_byte_span_t iovec =
      iree_make_const_byte_span((const uint8_t*)&payload, sizeof(payload));
  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_buffer_begin_operation(
      buffer, IREE_HAL_REPLAY_OPERATION_CODE_BUFFER_INVALIDATE_RANGE,
      IREE_HAL_REPLAY_PAYLOAD_TYPE_BUFFER_RANGE, &pending_record));
  iree_status_t status = IREE_HAL_REPLAY_VTABLE_DISPATCH(
      buffer->base_buffer, iree_hal_buffer, invalidate_range)(
      buffer->base_buffer, local_byte_offset, local_byte_length);
  return iree_hal_replay_recorder_end_operation_with_payload(&pending_record,
                                                             status, 1, &iovec);
}

static iree_status_t iree_hal_replay_recorder_buffer_flush_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  iree_hal_replay_recorder_buffer_t* buffer =
      iree_hal_replay_recorder_buffer_cast(base_buffer);
  iree_hal_replay_buffer_range_payload_t payload = {
      .byte_offset = local_byte_offset,
      .byte_length = local_byte_length,
  };
  iree_const_byte_span_t iovec =
      iree_make_const_byte_span((const uint8_t*)&payload, sizeof(payload));
  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_buffer_begin_operation(
      buffer, IREE_HAL_REPLAY_OPERATION_CODE_BUFFER_FLUSH_RANGE,
      IREE_HAL_REPLAY_PAYLOAD_TYPE_BUFFER_RANGE, &pending_record));
  iree_status_t status = IREE_HAL_REPLAY_VTABLE_DISPATCH(
      buffer->base_buffer, iree_hal_buffer, flush_range)(
      buffer->base_buffer, local_byte_offset, local_byte_length);
  return iree_hal_replay_recorder_end_operation_with_payload(&pending_record,
                                                             status, 1, &iovec);
}

static const iree_hal_buffer_vtable_t iree_hal_replay_recorder_buffer_vtable = {
    .recycle = iree_hal_buffer_recycle,
    .destroy = iree_hal_replay_recorder_buffer_destroy,
    .map_range = iree_hal_replay_recorder_buffer_map_range,
    .unmap_range = iree_hal_replay_recorder_buffer_unmap_range,
    .invalidate_range = iree_hal_replay_recorder_buffer_invalidate_range,
    .flush_range = iree_hal_replay_recorder_buffer_flush_range,
};

// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/replay/recorder_buffer.h"

#include <string.h>

#include "iree/base/threading/mutex.h"
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
  // Mutex guarding active write mappings.
  iree_slim_mutex_t mutex;
  // Active write mappings whose flush/unmap bytes may need capture.
  struct iree_hal_replay_recorder_buffer_mapping_t* mapping_list;
} iree_hal_replay_recorder_buffer_t;

typedef struct iree_hal_replay_recorder_buffer_mapping_t {
  // Caller-owned mapping object used to identify the mapping on unmap.
  iree_hal_buffer_mapping_t* mapping;
  // Allowed access bits captured from the mapping.
  iree_hal_memory_access_t allowed_access;
  // Next active mapping for the same replay buffer.
  struct iree_hal_replay_recorder_buffer_mapping_t* next;
} iree_hal_replay_recorder_buffer_mapping_t;

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

iree_hal_replay_object_id_t iree_hal_replay_recorder_buffer_id_or_none(
    iree_hal_buffer_t* buffer) {
  if (!buffer) return IREE_HAL_REPLAY_OBJECT_ID_NONE;
  if (iree_hal_replay_recorder_buffer_isa(buffer)) {
    return iree_hal_replay_recorder_buffer_cast(buffer)->buffer_id;
  }
  iree_hal_buffer_t* allocated_buffer =
      iree_hal_buffer_allocated_buffer(buffer);
  if (iree_hal_replay_recorder_buffer_isa(allocated_buffer)) {
    return iree_hal_replay_recorder_buffer_cast(allocated_buffer)->buffer_id;
  }
  return IREE_HAL_REPLAY_OBJECT_ID_NONE;
}

void iree_hal_replay_recorder_buffer_ref_make_payload(
    iree_hal_buffer_ref_t buffer_ref,
    iree_hal_replay_buffer_ref_payload_t* out_payload) {
  memset(out_payload, 0, sizeof(*out_payload));
  out_payload->offset = buffer_ref.offset;
  out_payload->length = buffer_ref.length;
  if (buffer_ref.buffer) {
    out_payload->buffer_id =
        iree_hal_replay_recorder_buffer_id_or_none(buffer_ref.buffer);
    out_payload->offset += iree_hal_buffer_byte_offset(buffer_ref.buffer);
  } else {
    out_payload->buffer_slot = buffer_ref.buffer_slot;
  }
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
  iree_slim_mutex_initialize(&buffer->mutex);

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

static bool iree_hal_replay_recorder_buffer_mapping_has_write_access(
    const iree_hal_buffer_mapping_t* mapping) {
  return iree_all_bits_set(mapping->impl.allowed_access,
                           IREE_HAL_MEMORY_ACCESS_WRITE);
}

static iree_status_t iree_hal_replay_recorder_buffer_make_range_data_payload(
    const iree_hal_buffer_mapping_t* mapping,
    iree_device_size_t local_byte_offset, iree_device_size_t local_byte_length,
    iree_hal_replay_buffer_range_data_payload_t* out_payload,
    iree_const_byte_span_t* out_data) {
  IREE_ASSERT_ARGUMENT(mapping);
  IREE_ASSERT_ARGUMENT(out_payload);
  IREE_ASSERT_ARGUMENT(out_data);
  memset(out_payload, 0, sizeof(*out_payload));
  *out_data = iree_make_const_byte_span(NULL, 0);

  if (IREE_UNLIKELY(!mapping->contents.data)) {
    if (local_byte_length == 0) return iree_ok_status();
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "cannot capture replay mapped buffer data without mapped contents");
  }
  if (IREE_UNLIKELY(local_byte_offset < mapping->impl.byte_offset)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay mapped buffer range begins before mapping");
  }
  iree_device_size_t mapped_data_offset =
      local_byte_offset - mapping->impl.byte_offset;
  if (IREE_UNLIKELY(mapped_data_offset > mapping->contents.data_length ||
                    local_byte_length >
                        mapping->contents.data_length - mapped_data_offset)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay mapped buffer range extends past mapping");
  }
  if (IREE_UNLIKELY(local_byte_length > IREE_HOST_SIZE_MAX)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "replay mapped buffer range exceeds host size");
  }

  out_payload->byte_offset = local_byte_offset;
  out_payload->byte_length = local_byte_length;
  out_payload->data_length = local_byte_length;
  out_payload->memory_access = mapping->impl.allowed_access;
  *out_data = iree_make_const_byte_span(
      mapping->contents.data + (iree_host_size_t)mapped_data_offset,
      (iree_host_size_t)local_byte_length);
  return iree_ok_status();
}

static void iree_hal_replay_recorder_buffer_unregister_mapping(
    iree_hal_replay_recorder_buffer_t* buffer,
    iree_hal_buffer_mapping_t* mapping) {
  iree_hal_replay_recorder_buffer_mapping_t* removed_entry = NULL;
  iree_slim_mutex_lock(&buffer->mutex);
  iree_hal_replay_recorder_buffer_mapping_t** entry_ptr = &buffer->mapping_list;
  while (*entry_ptr) {
    iree_hal_replay_recorder_buffer_mapping_t* entry = *entry_ptr;
    if (entry->mapping == mapping) {
      removed_entry = entry;
      *entry_ptr = entry->next;
      break;
    }
    entry_ptr = &entry->next;
  }
  iree_slim_mutex_unlock(&buffer->mutex);
  iree_allocator_free(buffer->host_allocator, removed_entry);
}

static void iree_hal_replay_recorder_buffer_clear_mappings(
    iree_hal_replay_recorder_buffer_t* buffer) {
  iree_slim_mutex_lock(&buffer->mutex);
  iree_hal_replay_recorder_buffer_mapping_t* mapping_list =
      buffer->mapping_list;
  buffer->mapping_list = NULL;
  iree_slim_mutex_unlock(&buffer->mutex);
  while (mapping_list) {
    iree_hal_replay_recorder_buffer_mapping_t* entry = mapping_list;
    mapping_list = entry->next;
    iree_allocator_free(buffer->host_allocator, entry);
  }
}

static iree_status_t iree_hal_replay_recorder_buffer_capture_mapping_range(
    iree_hal_replay_recorder_buffer_t* buffer,
    iree_device_size_t local_byte_offset, iree_device_size_t local_byte_length,
    iree_hal_replay_buffer_range_data_payload_t* out_payload,
    iree_const_byte_span_t* out_data) {
  IREE_ASSERT_ARGUMENT(out_payload);
  IREE_ASSERT_ARGUMENT(out_data);
  memset(out_payload, 0, sizeof(*out_payload));
  *out_data = iree_make_const_byte_span(NULL, 0);
  if (local_byte_length == 0) return iree_ok_status();

  iree_slim_mutex_lock(&buffer->mutex);
  bool found_mapping = false;
  iree_status_t status = iree_ok_status();
  for (iree_hal_replay_recorder_buffer_mapping_t* entry = buffer->mapping_list;
       entry; entry = entry->next) {
    if (!iree_all_bits_set(entry->allowed_access,
                           IREE_HAL_MEMORY_ACCESS_WRITE)) {
      continue;
    }
    if (local_byte_offset < entry->mapping->impl.byte_offset) {
      continue;
    }
    const iree_device_size_t mapping_data_offset =
        local_byte_offset - entry->mapping->impl.byte_offset;
    if (mapping_data_offset > entry->mapping->contents.data_length ||
        local_byte_length >
            entry->mapping->contents.data_length - mapping_data_offset) {
      continue;
    }
    status = iree_hal_replay_recorder_buffer_make_range_data_payload(
        entry->mapping, local_byte_offset, local_byte_length, out_payload,
        out_data);
    found_mapping = true;
    break;
  }
  iree_slim_mutex_unlock(&buffer->mutex);
  if (!found_mapping) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "cannot capture replay buffer flush without an active write mapping");
  }
  return status;
}

static void iree_hal_replay_recorder_buffer_destroy(
    iree_hal_buffer_t* base_buffer) {
  iree_hal_replay_recorder_buffer_t* buffer =
      iree_hal_replay_recorder_buffer_cast(base_buffer);
  iree_allocator_t host_allocator = buffer->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_replay_recorder_buffer_clear_mappings(buffer);
  iree_slim_mutex_deinitialize(&buffer->mutex);
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
  const bool has_write_access =
      iree_all_bits_set(memory_access, IREE_HAL_MEMORY_ACCESS_WRITE);
  iree_hal_replay_recorder_buffer_mapping_t* mapping_entry = NULL;
  iree_status_t status = iree_ok_status();
  if (has_write_access &&
      iree_all_bits_set(mapping_mode, IREE_HAL_MAPPING_MODE_PERSISTENT)) {
    status = iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "HAL replay capture does not support persistent write mappings");
  }
  if (iree_status_is_ok(status) && has_write_access) {
    status = iree_allocator_malloc(
        buffer->host_allocator, sizeof(*mapping_entry), (void**)&mapping_entry);
  }
  if (iree_status_is_ok(status)) {
    status = IREE_HAL_REPLAY_VTABLE_DISPATCH(buffer->base_buffer,
                                             iree_hal_buffer, map_range)(
        buffer->base_buffer, mapping_mode, memory_access, local_byte_offset,
        local_byte_length, mapping);
  }
  if (iree_status_is_ok(status) && mapping_entry) {
    mapping_entry->mapping = mapping;
    mapping_entry->allowed_access = mapping->impl.allowed_access;
    iree_slim_mutex_lock(&buffer->mutex);
    mapping_entry->next = buffer->mapping_list;
    buffer->mapping_list = mapping_entry;
    mapping_entry = NULL;
    iree_slim_mutex_unlock(&buffer->mutex);
  }
  iree_allocator_free(buffer->host_allocator, mapping_entry);
  return iree_hal_replay_recorder_end_operation_with_payload(&pending_record,
                                                             status, 1, &iovec);
}

static iree_status_t iree_hal_replay_recorder_buffer_unmap_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length, iree_hal_buffer_mapping_t* mapping) {
  iree_hal_replay_recorder_buffer_t* buffer =
      iree_hal_replay_recorder_buffer_cast(base_buffer);
  const bool capture_data =
      iree_hal_replay_recorder_buffer_mapping_has_write_access(mapping);
  iree_hal_replay_buffer_range_payload_t payload = {
      .byte_offset = local_byte_offset,
      .byte_length = local_byte_length,
      .memory_access = mapping->impl.allowed_access,
  };
  iree_hal_replay_buffer_range_data_payload_t data_payload;
  iree_const_byte_span_t data_span = iree_make_const_byte_span(NULL, 0);
  uint8_t* data_snapshot = NULL;
  iree_status_t status = iree_ok_status();
  if (capture_data) {
    status = iree_hal_replay_recorder_buffer_make_range_data_payload(
        mapping, local_byte_offset, local_byte_length, &data_payload,
        &data_span);
    if (iree_status_is_ok(status) && data_span.data_length > 0) {
      status =
          iree_allocator_malloc(buffer->host_allocator, data_span.data_length,
                                (void**)&data_snapshot);
      if (iree_status_is_ok(status)) {
        memcpy(data_snapshot, data_span.data, data_span.data_length);
        data_span =
            iree_make_const_byte_span(data_snapshot, data_span.data_length);
      }
    }
  }
  iree_const_byte_span_t iovecs[2] = {
      iree_make_const_byte_span(
          capture_data ? (const uint8_t*)&data_payload
                       : (const uint8_t*)&payload,
          capture_data ? sizeof(data_payload) : sizeof(payload)),
      data_span,
  };
  iree_hal_replay_pending_record_t pending_record;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_buffer_begin_operation(
        buffer, IREE_HAL_REPLAY_OPERATION_CODE_BUFFER_UNMAP_RANGE,
        capture_data ? IREE_HAL_REPLAY_PAYLOAD_TYPE_BUFFER_RANGE_DATA
                     : IREE_HAL_REPLAY_PAYLOAD_TYPE_BUFFER_RANGE,
        &pending_record);
  }
  if (iree_status_is_ok(status)) {
    status = IREE_HAL_REPLAY_VTABLE_DISPATCH(buffer->base_buffer,
                                             iree_hal_buffer, unmap_range)(
        buffer->base_buffer, local_byte_offset, local_byte_length, mapping);
    status = iree_hal_replay_recorder_end_operation_with_payload(
        &pending_record, status, capture_data ? 2 : 1, iovecs);
  }
  if (capture_data) {
    iree_hal_replay_recorder_buffer_unregister_mapping(buffer, mapping);
  }
  iree_allocator_free(buffer->host_allocator, data_snapshot);
  return status;
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
  iree_hal_replay_buffer_range_data_payload_t data_payload;
  iree_const_byte_span_t data_span = iree_make_const_byte_span(NULL, 0);
  uint8_t* data_snapshot = NULL;
  iree_status_t status = iree_hal_replay_recorder_buffer_capture_mapping_range(
      buffer, local_byte_offset, local_byte_length, &data_payload, &data_span);
  if (iree_status_is_ok(status) && data_span.data_length > 0) {
    status = iree_allocator_malloc(
        buffer->host_allocator, data_span.data_length, (void**)&data_snapshot);
    if (iree_status_is_ok(status)) {
      memcpy(data_snapshot, data_span.data, data_span.data_length);
      data_span =
          iree_make_const_byte_span(data_snapshot, data_span.data_length);
    }
  }
  const bool capture_data = iree_status_is_ok(status);
  iree_const_byte_span_t iovecs[2] = {
      iree_make_const_byte_span(
          capture_data ? (const uint8_t*)&data_payload
                       : (const uint8_t*)&payload,
          capture_data ? sizeof(data_payload) : sizeof(payload)),
      data_span,
  };
  iree_hal_replay_pending_record_t pending_record;
  iree_status_t begin_status = iree_hal_replay_recorder_buffer_begin_operation(
      buffer, IREE_HAL_REPLAY_OPERATION_CODE_BUFFER_FLUSH_RANGE,
      capture_data ? IREE_HAL_REPLAY_PAYLOAD_TYPE_BUFFER_RANGE_DATA
                   : IREE_HAL_REPLAY_PAYLOAD_TYPE_BUFFER_RANGE,
      &pending_record);
  if (!iree_status_is_ok(begin_status)) {
    iree_allocator_free(buffer->host_allocator, data_snapshot);
    return begin_status;
  }
  if (iree_status_is_ok(status)) {
    status = IREE_HAL_REPLAY_VTABLE_DISPATCH(buffer->base_buffer,
                                             iree_hal_buffer, flush_range)(
        buffer->base_buffer, local_byte_offset, local_byte_length);
  }
  status = iree_hal_replay_recorder_end_operation_with_payload(
      &pending_record, status, capture_data ? 2 : 1, iovecs);
  iree_allocator_free(buffer->host_allocator, data_snapshot);
  return status;
}

static const iree_hal_buffer_vtable_t iree_hal_replay_recorder_buffer_vtable = {
    .recycle = iree_hal_buffer_recycle,
    .destroy = iree_hal_replay_recorder_buffer_destroy,
    .map_range = iree_hal_replay_recorder_buffer_map_range,
    .unmap_range = iree_hal_replay_recorder_buffer_unmap_range,
    .invalidate_range = iree_hal_replay_recorder_buffer_invalidate_range,
    .flush_range = iree_hal_replay_recorder_buffer_flush_range,
};

// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/replay/dump.h"

#include <inttypes.h>
#include <stddef.h>
#include <string.h>

#include "iree/hal/replay/file_reader.h"

typedef struct iree_hal_replay_dump_context_t {
  // Caller-provided streaming sink.
  iree_hal_replay_dump_write_fn_t write;
  // Caller-provided sink state.
  void* user_data;
  // Host allocator used for temporary line construction.
  iree_allocator_t host_allocator;
} iree_hal_replay_dump_context_t;

static iree_status_t iree_hal_replay_dump_emit(
    iree_hal_replay_dump_context_t* context, iree_string_builder_t* builder) {
  if (iree_string_builder_size(builder) == 0) return iree_ok_status();
  iree_status_t status =
      context->write(context->user_data, iree_string_builder_view(builder));
  iree_string_builder_reset(builder);
  return status;
}

static iree_status_t iree_hal_replay_dump_payload_length_check(
    const iree_hal_replay_file_record_t* record,
    iree_host_size_t expected_payload_length) {
  if (IREE_LIKELY(record->payload.data_length == expected_payload_length)) {
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_DATA_LOSS,
                          "replay payload type %u has %" PRIhsz
                          " bytes; expected %" PRIhsz,
                          record->header.payload_type,
                          record->payload.data_length, expected_payload_length);
}

static iree_hal_replay_file_range_t iree_hal_replay_dump_record_payload_range(
    const iree_hal_replay_file_record_t* record,
    iree_host_size_t record_offset) {
  iree_hal_replay_file_range_t range = iree_hal_replay_file_range_empty();
  range.offset = (uint64_t)record_offset + record->header.header_length;
  range.length = record->header.payload_length;
  range.uncompressed_length = record->header.payload_length;
  range.compression_type = IREE_HAL_REPLAY_COMPRESSION_TYPE_NONE;
  range.digest_type = IREE_HAL_REPLAY_DIGEST_TYPE_NONE;
  return range;
}

static iree_status_t iree_hal_replay_dump_append_json_string(
    iree_string_builder_t* builder, const char* value) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\""));
  for (const char* p = value; *p; ++p) {
    switch (*p) {
      case '\\':
      case '"': {
        IREE_RETURN_IF_ERROR(
            iree_string_builder_append_format(builder, "\\%c", *p));
        break;
      }
      case '\b': {
        IREE_RETURN_IF_ERROR(
            iree_string_builder_append_cstring(builder, "\\b"));
        break;
      }
      case '\f': {
        IREE_RETURN_IF_ERROR(
            iree_string_builder_append_cstring(builder, "\\f"));
        break;
      }
      case '\n': {
        IREE_RETURN_IF_ERROR(
            iree_string_builder_append_cstring(builder, "\\n"));
        break;
      }
      case '\r': {
        IREE_RETURN_IF_ERROR(
            iree_string_builder_append_cstring(builder, "\\r"));
        break;
      }
      case '\t': {
        IREE_RETURN_IF_ERROR(
            iree_string_builder_append_cstring(builder, "\\t"));
        break;
      }
      default: {
        if ((uint8_t)*p < 0x20) {
          IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
              builder, "\\u%04x", (uint32_t)(uint8_t)*p));
        } else {
          IREE_RETURN_IF_ERROR(
              iree_string_builder_append_format(builder, "%c", *p));
        }
        break;
      }
    }
  }
  return iree_string_builder_append_cstring(builder, "\"");
}

static iree_status_t iree_hal_replay_dump_append_json_file_range(
    iree_string_builder_t* builder, const char* field_name,
    const iree_hal_replay_file_range_t* range) {
  return iree_string_builder_append_format(
      builder,
      ",\"%s\":{\"offset\":%" PRIu64 ",\"length\":%" PRIu64
      ",\"uncompressed_length\":%" PRIu64
      ",\"compression_type\":%u"
      ",\"digest_type\":%u}",
      field_name, range->offset, range->length, range->uncompressed_length,
      (uint32_t)range->compression_type, (uint32_t)range->digest_type);
}

static iree_hal_replay_file_range_t iree_hal_replay_dump_payload_subrange(
    const iree_hal_replay_file_range_t* payload_range,
    iree_host_size_t payload_offset, iree_host_size_t payload_length) {
  iree_hal_replay_file_range_t range = iree_hal_replay_file_range_empty();
  range.offset = payload_range->offset + payload_offset;
  range.length = payload_length;
  range.uncompressed_length = payload_length;
  range.compression_type = IREE_HAL_REPLAY_COMPRESSION_TYPE_NONE;
  range.digest_type = IREE_HAL_REPLAY_DIGEST_TYPE_NONE;
  return range;
}

static iree_status_t iree_hal_replay_dump_append_text_buffer_ref(
    iree_string_builder_t* builder, const char* label,
    const iree_hal_replay_buffer_ref_payload_t* buffer_ref) {
  return iree_string_builder_append_format(
      builder,
      " %s={buffer_id=%" PRIu64 " offset=%" PRIu64 " length=%" PRIu64
      " slot=%" PRIu32 "}",
      label, buffer_ref->buffer_id, buffer_ref->offset, buffer_ref->length,
      buffer_ref->buffer_slot);
}

static iree_status_t iree_hal_replay_dump_append_json_buffer_ref(
    iree_string_builder_t* builder, const char* field_name,
    const iree_hal_replay_buffer_ref_payload_t* buffer_ref) {
  return iree_string_builder_append_format(
      builder,
      ",\"%s\":{\"buffer_id\":%" PRIu64 ",\"offset\":%" PRIu64
      ",\"length\":%" PRIu64 ",\"buffer_slot\":%" PRIu32 "}",
      field_name, buffer_ref->buffer_id, buffer_ref->offset, buffer_ref->length,
      buffer_ref->buffer_slot);
}

static iree_status_t iree_hal_replay_dump_dispatch_layout(
    const iree_hal_replay_file_record_t* record,
    const iree_hal_replay_dispatch_payload_t* payload,
    iree_host_size_t* out_wait_payloads_offset,
    iree_host_size_t* out_wait_payloads_size,
    iree_host_size_t* out_signal_payloads_offset,
    iree_host_size_t* out_signal_payloads_size,
    iree_host_size_t* out_constants_offset,
    iree_host_size_t* out_binding_payloads_offset,
    iree_host_size_t* out_binding_payloads_size) {
  iree_host_size_t wait_payloads_size = 0;
  iree_host_size_t signal_payloads_size = 0;
  iree_host_size_t binding_payloads_size = 0;
  if (payload->wait_semaphore_count > IREE_HOST_SIZE_MAX ||
      payload->signal_semaphore_count > IREE_HOST_SIZE_MAX ||
      payload->binding_count > IREE_HOST_SIZE_MAX ||
      payload->constants_length > IREE_HOST_SIZE_MAX ||
      !iree_host_size_checked_mul(
          (iree_host_size_t)payload->wait_semaphore_count,
          sizeof(iree_hal_replay_semaphore_timepoint_payload_t),
          &wait_payloads_size) ||
      !iree_host_size_checked_mul(
          (iree_host_size_t)payload->signal_semaphore_count,
          sizeof(iree_hal_replay_semaphore_timepoint_payload_t),
          &signal_payloads_size) ||
      !iree_host_size_checked_mul((iree_host_size_t)payload->binding_count,
                                  sizeof(iree_hal_replay_buffer_ref_payload_t),
                                  &binding_payloads_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "replay dispatch payload count overflow");
  }

  iree_host_size_t offset = sizeof(*payload);
  *out_wait_payloads_offset = offset;
  *out_wait_payloads_size = wait_payloads_size;
  if (!iree_host_size_checked_add(offset, wait_payloads_size, &offset)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "replay dispatch payload length overflow");
  }
  *out_signal_payloads_offset = offset;
  *out_signal_payloads_size = signal_payloads_size;
  if (!iree_host_size_checked_add(offset, signal_payloads_size, &offset)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "replay dispatch payload length overflow");
  }
  *out_constants_offset = offset;
  if (!iree_host_size_checked_add(
          offset, (iree_host_size_t)payload->constants_length, &offset)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "replay dispatch payload length overflow");
  }
  *out_binding_payloads_offset = offset;
  *out_binding_payloads_size = binding_payloads_size;
  if (!iree_host_size_checked_add(offset, binding_payloads_size, &offset) ||
      offset != record->payload.data_length) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay dispatch payload length mismatch");
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_replay_dump_queue_execute_layout(
    const iree_hal_replay_file_record_t* record,
    const iree_hal_replay_device_queue_execute_payload_t* payload,
    iree_host_size_t* out_wait_payloads_offset,
    iree_host_size_t* out_wait_payloads_size,
    iree_host_size_t* out_signal_payloads_offset,
    iree_host_size_t* out_signal_payloads_size,
    iree_host_size_t* out_binding_payloads_offset,
    iree_host_size_t* out_binding_payloads_size) {
  iree_host_size_t wait_payloads_size = 0;
  iree_host_size_t signal_payloads_size = 0;
  iree_host_size_t binding_payloads_size = 0;
  if (payload->wait_semaphore_count > IREE_HOST_SIZE_MAX ||
      payload->signal_semaphore_count > IREE_HOST_SIZE_MAX ||
      payload->binding_count > IREE_HOST_SIZE_MAX ||
      !iree_host_size_checked_mul(
          (iree_host_size_t)payload->wait_semaphore_count,
          sizeof(iree_hal_replay_semaphore_timepoint_payload_t),
          &wait_payloads_size) ||
      !iree_host_size_checked_mul(
          (iree_host_size_t)payload->signal_semaphore_count,
          sizeof(iree_hal_replay_semaphore_timepoint_payload_t),
          &signal_payloads_size) ||
      !iree_host_size_checked_mul((iree_host_size_t)payload->binding_count,
                                  sizeof(iree_hal_replay_buffer_ref_payload_t),
                                  &binding_payloads_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "replay queue execute payload count overflow");
  }

  iree_host_size_t offset = sizeof(*payload);
  *out_wait_payloads_offset = offset;
  *out_wait_payloads_size = wait_payloads_size;
  if (!iree_host_size_checked_add(offset, wait_payloads_size, &offset)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "replay queue execute payload length overflow");
  }
  *out_signal_payloads_offset = offset;
  *out_signal_payloads_size = signal_payloads_size;
  if (!iree_host_size_checked_add(offset, signal_payloads_size, &offset)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "replay queue execute payload length overflow");
  }
  *out_binding_payloads_offset = offset;
  *out_binding_payloads_size = binding_payloads_size;
  if (!iree_host_size_checked_add(offset, binding_payloads_size, &offset) ||
      offset != record->payload.data_length) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay queue execute payload length mismatch");
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_replay_dump_append_text_payload(
    iree_string_builder_t* builder, const iree_hal_replay_file_record_t* record,
    const iree_hal_replay_file_range_t* payload_range) {
  switch (record->header.payload_type) {
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_NONE:
      return iree_ok_status();
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_BUFFER_OBJECT: {
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_payload_length_check(
          record, sizeof(iree_hal_replay_buffer_object_payload_t)));
      iree_hal_replay_buffer_object_payload_t payload;
      memcpy(&payload, record->payload.data, sizeof(payload));
      return iree_string_builder_append_format(
          builder,
          " allocation_size=%" PRIu64 " byte_offset=%" PRIu64
          " byte_length=%" PRIu64 " queue_affinity=%" PRIu64
          " placement_flags=0x%08" PRIx32 " memory_type=0x%08" PRIx32
          " allowed_usage=0x%08" PRIx32 " allowed_access=0x%04" PRIx16,
          payload.allocation_size, payload.byte_offset, payload.byte_length,
          payload.queue_affinity, payload.placement_flags, payload.memory_type,
          payload.allowed_usage, payload.allowed_access);
    }
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_ALLOCATE_BUFFER: {
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_payload_length_check(
          record, sizeof(iree_hal_replay_allocator_allocate_buffer_payload_t)));
      iree_hal_replay_allocator_allocate_buffer_payload_t payload;
      memcpy(&payload, record->payload.data, sizeof(payload));
      return iree_string_builder_append_format(
          builder,
          " allocation_size=%" PRIu64 " queue_affinity=%" PRIu64
          " min_alignment=%" PRIu64 " usage=0x%08" PRIx32 " type=0x%08" PRIx32
          " access=0x%04" PRIx16,
          payload.allocation_size, payload.queue_affinity,
          payload.min_alignment, payload.usage, payload.type, payload.access);
    }
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_BUFFER_RANGE: {
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_payload_length_check(
          record, sizeof(iree_hal_replay_buffer_range_payload_t)));
      iree_hal_replay_buffer_range_payload_t payload;
      memcpy(&payload, record->payload.data, sizeof(payload));
      return iree_string_builder_append_format(
          builder,
          " byte_offset=%" PRIu64 " byte_length=%" PRIu64
          " mapping_mode=0x%08" PRIx32 " memory_access=0x%04" PRIx16,
          payload.byte_offset, payload.byte_length, payload.mapping_mode,
          payload.memory_access);
    }
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_BUFFER_RANGE_DATA: {
      if (record->payload.data_length <
          sizeof(iree_hal_replay_buffer_range_data_payload_t)) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "replay buffer range data payload is short");
      }
      iree_hal_replay_buffer_range_data_payload_t payload;
      memcpy(&payload, record->payload.data, sizeof(payload));
      if (payload.data_length >
          record->payload.data_length -
              sizeof(iree_hal_replay_buffer_range_data_payload_t)) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "replay buffer range data extends past record");
      }
      const uint64_t data_offset =
          payload_range->offset +
          (uint64_t)sizeof(iree_hal_replay_buffer_range_data_payload_t);
      return iree_string_builder_append_format(
          builder,
          " byte_offset=%" PRIu64 " byte_length=%" PRIu64
          " data_range=[%" PRIu64 ", +%" PRIu64
          "]"
          " memory_access=0x%04" PRIx16,
          payload.byte_offset, payload.byte_length, data_offset,
          payload.data_length, payload.memory_access);
    }
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_OBJECT: {
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_payload_length_check(
          record, sizeof(iree_hal_replay_command_buffer_object_payload_t)));
      iree_hal_replay_command_buffer_object_payload_t payload;
      memcpy(&payload, record->payload.data, sizeof(payload));
      return iree_string_builder_append_format(
          builder,
          " mode=0x%08" PRIx32 " categories=0x%08" PRIx32
          " queue_affinity=%" PRIu64 " binding_capacity=%" PRIu64,
          payload.mode, payload.command_categories, payload.queue_affinity,
          payload.binding_capacity);
    }
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_EXECUTABLE_CACHE_OBJECT: {
      if (record->payload.data_length <
          sizeof(iree_hal_replay_executable_cache_object_payload_t)) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "replay executable cache payload is short");
      }
      iree_hal_replay_executable_cache_object_payload_t payload;
      memcpy(&payload, record->payload.data, sizeof(payload));
      if (payload.identifier_length >
          record->payload.data_length -
              sizeof(iree_hal_replay_executable_cache_object_payload_t)) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "replay executable cache identifier extends "
                                "past record");
      }
      const uint64_t identifier_offset =
          payload_range->offset +
          (uint64_t)sizeof(iree_hal_replay_executable_cache_object_payload_t);
      return iree_string_builder_append_format(
          builder, " identifier_range=[%" PRIu64 ", +%" PRIu64 "]",
          identifier_offset, payload.identifier_length);
    }
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_EXECUTABLE_PREPARE: {
      if (record->payload.data_length <
          sizeof(iree_hal_replay_executable_prepare_payload_t)) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "replay executable prepare payload is short");
      }
      iree_hal_replay_executable_prepare_payload_t payload;
      memcpy(&payload, record->payload.data, sizeof(payload));
      iree_host_size_t constant_bytes = 0;
      if (payload.executable_data_length > IREE_HOST_SIZE_MAX ||
          payload.constant_count > IREE_HOST_SIZE_MAX ||
          !iree_host_size_checked_mul((iree_host_size_t)payload.constant_count,
                                      sizeof(uint32_t), &constant_bytes)) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "replay executable prepare payload overflow");
      }
      iree_host_size_t format_offset = sizeof(payload);
      iree_host_size_t data_offset = 0;
      iree_host_size_t constants_offset = 0;
      iree_host_size_t expected_length = 0;
      if (!iree_host_size_checked_add(
              format_offset, (iree_host_size_t)payload.executable_format_length,
              &data_offset) ||
          !iree_host_size_checked_add(
              data_offset, (iree_host_size_t)payload.executable_data_length,
              &constants_offset) ||
          !iree_host_size_checked_add(constants_offset, constant_bytes,
                                      &expected_length) ||
          expected_length != record->payload.data_length) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "replay executable prepare payload length "
                                "mismatch");
      }
      return iree_string_builder_append_format(
          builder,
          " queue_affinity=%" PRIu64 " caching_mode=0x%08" PRIx32
          " format_range=[%" PRIu64 ", +%" PRIu32 "] data_range=[%" PRIu64
          ", +%" PRIu64 "] constants_range=[%" PRIu64 ", +%" PRIhsz "]",
          payload.queue_affinity, payload.caching_mode,
          payload_range->offset + format_offset,
          payload.executable_format_length, payload_range->offset + data_offset,
          payload.executable_data_length,
          payload_range->offset + constants_offset, constant_bytes);
    }
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_SEMAPHORE_OBJECT: {
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_payload_length_check(
          record, sizeof(iree_hal_replay_semaphore_object_payload_t)));
      iree_hal_replay_semaphore_object_payload_t payload;
      memcpy(&payload, record->payload.data, sizeof(payload));
      return iree_string_builder_append_format(
          builder,
          " queue_affinity=%" PRIu64 " initial_value=%" PRIu64
          " flags=0x%016" PRIx64,
          payload.queue_affinity, payload.initial_value, payload.flags);
    }
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_DISPATCH: {
      if (record->payload.data_length <
          sizeof(iree_hal_replay_dispatch_payload_t)) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "replay dispatch payload is short");
      }
      iree_hal_replay_dispatch_payload_t payload;
      memcpy(&payload, record->payload.data, sizeof(payload));
      iree_host_size_t wait_offset = 0;
      iree_host_size_t wait_size = 0;
      iree_host_size_t signal_offset = 0;
      iree_host_size_t signal_size = 0;
      iree_host_size_t constants_offset = 0;
      iree_host_size_t bindings_offset = 0;
      iree_host_size_t bindings_size = 0;
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_dispatch_layout(
          record, &payload, &wait_offset, &wait_size, &signal_offset,
          &signal_size, &constants_offset, &bindings_offset, &bindings_size));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          " executable_id=%" PRIu64 " queue_affinity=%" PRIu64
          " export_ordinal=%" PRIu32 " flags=0x%08" PRIx32
          " workgroup_count=[%" PRIu32 ",%" PRIu32 ",%" PRIu32
          "] workgroup_size=[%" PRIu32 ",%" PRIu32 ",%" PRIu32
          "] wait_count=%" PRIu64 " signal_count=%" PRIu64
          " constants_range=[%" PRIu64 ", +%" PRIu64
          "] bindings_range=[%" PRIu64 ", +%" PRIhsz "]",
          payload.executable_id, payload.queue_affinity, payload.export_ordinal,
          payload.flags, payload.workgroup_count[0], payload.workgroup_count[1],
          payload.workgroup_count[2], payload.workgroup_size[0],
          payload.workgroup_size[1], payload.workgroup_size[2],
          payload.wait_semaphore_count, payload.signal_semaphore_count,
          payload_range->offset + constants_offset, payload.constants_length,
          payload_range->offset + bindings_offset, bindings_size));
      return iree_hal_replay_dump_append_text_buffer_ref(
          builder, "workgroup_count_ref", &payload.workgroup_count_ref);
    }
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_EXECUTE: {
      if (record->payload.data_length <
          sizeof(iree_hal_replay_device_queue_execute_payload_t)) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "replay queue execute payload is short");
      }
      iree_hal_replay_device_queue_execute_payload_t payload;
      memcpy(&payload, record->payload.data, sizeof(payload));
      iree_host_size_t wait_offset = 0;
      iree_host_size_t wait_size = 0;
      iree_host_size_t signal_offset = 0;
      iree_host_size_t signal_size = 0;
      iree_host_size_t bindings_offset = 0;
      iree_host_size_t bindings_size = 0;
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_queue_execute_layout(
          record, &payload, &wait_offset, &wait_size, &signal_offset,
          &signal_size, &bindings_offset, &bindings_size));
      return iree_string_builder_append_format(
          builder,
          " command_buffer_id=%" PRIu64 " queue_affinity=%" PRIu64
          " flags=0x%016" PRIx64 " wait_count=%" PRIu64 " signal_count=%" PRIu64
          " bindings_range=[%" PRIu64 ", +%" PRIhsz "]",
          payload.command_buffer_id, payload.queue_affinity, payload.flags,
          payload.wait_semaphore_count, payload.signal_semaphore_count,
          payload_range->offset + bindings_offset, bindings_size);
    }
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_COPY_BUFFER: {
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_payload_length_check(
          record,
          sizeof(iree_hal_replay_command_buffer_copy_buffer_payload_t)));
      iree_hal_replay_command_buffer_copy_buffer_payload_t payload;
      memcpy(&payload, record->payload.data, sizeof(payload));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder, " flags=0x%08" PRIx32, payload.flags));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_text_buffer_ref(
          builder, "source", &payload.source_ref));
      return iree_hal_replay_dump_append_text_buffer_ref(builder, "target",
                                                         &payload.target_ref);
    }
    default:
      return iree_ok_status();
  }
}

static iree_status_t iree_hal_replay_dump_append_json_payload(
    iree_string_builder_t* builder, const iree_hal_replay_file_record_t* record,
    const iree_hal_replay_file_range_t* payload_range) {
  switch (record->header.payload_type) {
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_NONE:
      return iree_ok_status();
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_BUFFER_OBJECT: {
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_payload_length_check(
          record, sizeof(iree_hal_replay_buffer_object_payload_t)));
      iree_hal_replay_buffer_object_payload_t payload;
      memcpy(&payload, record->payload.data, sizeof(payload));
      return iree_string_builder_append_format(
          builder,
          ",\"payload\":{\"allocation_size\":%" PRIu64
          ",\"byte_offset\":%" PRIu64 ",\"byte_length\":%" PRIu64
          ",\"queue_affinity\":%" PRIu64 ",\"placement_flags\":%" PRIu32
          ",\"memory_type\":%" PRIu32 ",\"allowed_usage\":%" PRIu32
          ",\"allowed_access\":%" PRIu16 "}",
          payload.allocation_size, payload.byte_offset, payload.byte_length,
          payload.queue_affinity, payload.placement_flags, payload.memory_type,
          payload.allowed_usage, payload.allowed_access);
    }
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_ALLOCATE_BUFFER: {
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_payload_length_check(
          record, sizeof(iree_hal_replay_allocator_allocate_buffer_payload_t)));
      iree_hal_replay_allocator_allocate_buffer_payload_t payload;
      memcpy(&payload, record->payload.data, sizeof(payload));
      return iree_string_builder_append_format(
          builder,
          ",\"payload\":{\"allocation_size\":%" PRIu64
          ",\"queue_affinity\":%" PRIu64 ",\"min_alignment\":%" PRIu64
          ",\"usage\":%" PRIu32 ",\"type\":%" PRIu32 ",\"access\":%" PRIu16 "}",
          payload.allocation_size, payload.queue_affinity,
          payload.min_alignment, payload.usage, payload.type, payload.access);
    }
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_BUFFER_RANGE: {
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_payload_length_check(
          record, sizeof(iree_hal_replay_buffer_range_payload_t)));
      iree_hal_replay_buffer_range_payload_t payload;
      memcpy(&payload, record->payload.data, sizeof(payload));
      return iree_string_builder_append_format(
          builder,
          ",\"payload\":{\"byte_offset\":%" PRIu64 ",\"byte_length\":%" PRIu64
          ",\"mapping_mode\":%" PRIu32 ",\"memory_access\":%" PRIu16 "}",
          payload.byte_offset, payload.byte_length, payload.mapping_mode,
          payload.memory_access);
    }
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_BUFFER_RANGE_DATA: {
      if (record->payload.data_length <
          sizeof(iree_hal_replay_buffer_range_data_payload_t)) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "replay buffer range data payload is short");
      }
      iree_hal_replay_buffer_range_data_payload_t payload;
      memcpy(&payload, record->payload.data, sizeof(payload));
      if (payload.data_length >
          record->payload.data_length -
              sizeof(iree_hal_replay_buffer_range_data_payload_t)) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "replay buffer range data extends past record");
      }
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          ",\"payload\":{\"byte_offset\":%" PRIu64 ",\"byte_length\":%" PRIu64
          ",\"mapping_mode\":%" PRIu32 ",\"memory_access\":%" PRIu16,
          payload.byte_offset, payload.byte_length, payload.mapping_mode,
          payload.memory_access));
      iree_hal_replay_file_range_t data_range =
          iree_hal_replay_file_range_empty();
      data_range.offset =
          payload_range->offset +
          (uint64_t)sizeof(iree_hal_replay_buffer_range_data_payload_t);
      data_range.length = payload.data_length;
      data_range.uncompressed_length = payload.data_length;
      data_range.compression_type = IREE_HAL_REPLAY_COMPRESSION_TYPE_NONE;
      data_range.digest_type = IREE_HAL_REPLAY_DIGEST_TYPE_NONE;
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_file_range(
          builder, "data_range", &data_range));
      return iree_string_builder_append_cstring(builder, "}");
    }
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_OBJECT: {
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_payload_length_check(
          record, sizeof(iree_hal_replay_command_buffer_object_payload_t)));
      iree_hal_replay_command_buffer_object_payload_t payload;
      memcpy(&payload, record->payload.data, sizeof(payload));
      return iree_string_builder_append_format(
          builder,
          ",\"payload\":{\"mode\":%" PRIu32 ",\"command_categories\":%" PRIu32
          ",\"queue_affinity\":%" PRIu64 ",\"binding_capacity\":%" PRIu64 "}",
          payload.mode, payload.command_categories, payload.queue_affinity,
          payload.binding_capacity);
    }
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_EXECUTABLE_CACHE_OBJECT: {
      if (record->payload.data_length <
          sizeof(iree_hal_replay_executable_cache_object_payload_t)) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "replay executable cache payload is short");
      }
      iree_hal_replay_executable_cache_object_payload_t payload;
      memcpy(&payload, record->payload.data, sizeof(payload));
      if (payload.identifier_length >
          record->payload.data_length -
              sizeof(iree_hal_replay_executable_cache_object_payload_t)) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "replay executable cache identifier extends "
                                "past record");
      }
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
          builder, ",\"payload\":{\"identifier_length\":"));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder, "%" PRIu64, payload.identifier_length));
      iree_hal_replay_file_range_t identifier_range =
          iree_hal_replay_dump_payload_subrange(
              payload_range,
              sizeof(iree_hal_replay_executable_cache_object_payload_t),
              (iree_host_size_t)payload.identifier_length);
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_file_range(
          builder, "identifier_range", &identifier_range));
      return iree_string_builder_append_cstring(builder, "}");
    }
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_EXECUTABLE_PREPARE: {
      if (record->payload.data_length <
          sizeof(iree_hal_replay_executable_prepare_payload_t)) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "replay executable prepare payload is short");
      }
      iree_hal_replay_executable_prepare_payload_t payload;
      memcpy(&payload, record->payload.data, sizeof(payload));
      iree_host_size_t constant_bytes = 0;
      if (payload.executable_data_length > IREE_HOST_SIZE_MAX ||
          payload.constant_count > IREE_HOST_SIZE_MAX ||
          !iree_host_size_checked_mul((iree_host_size_t)payload.constant_count,
                                      sizeof(uint32_t), &constant_bytes)) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "replay executable prepare payload overflow");
      }
      iree_host_size_t format_offset = sizeof(payload);
      iree_host_size_t data_offset = 0;
      iree_host_size_t constants_offset = 0;
      iree_host_size_t expected_length = 0;
      if (!iree_host_size_checked_add(
              format_offset, (iree_host_size_t)payload.executable_format_length,
              &data_offset) ||
          !iree_host_size_checked_add(
              data_offset, (iree_host_size_t)payload.executable_data_length,
              &constants_offset) ||
          !iree_host_size_checked_add(constants_offset, constant_bytes,
                                      &expected_length) ||
          expected_length != record->payload.data_length) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "replay executable prepare payload length "
                                "mismatch");
      }
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          ",\"payload\":{\"queue_affinity\":%" PRIu64
          ",\"caching_mode\":%" PRIu32 ",\"executable_format_length\":%" PRIu32
          ",\"executable_data_length\":%" PRIu64 ",\"constant_count\":%" PRIu64,
          payload.queue_affinity, payload.caching_mode,
          payload.executable_format_length, payload.executable_data_length,
          payload.constant_count));
      iree_hal_replay_file_range_t format_range =
          iree_hal_replay_dump_payload_subrange(
              payload_range, format_offset,
              (iree_host_size_t)payload.executable_format_length);
      iree_hal_replay_file_range_t data_range =
          iree_hal_replay_dump_payload_subrange(
              payload_range, data_offset,
              (iree_host_size_t)payload.executable_data_length);
      iree_hal_replay_file_range_t constants_range =
          iree_hal_replay_dump_payload_subrange(payload_range, constants_offset,
                                                constant_bytes);
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_file_range(
          builder, "format_range", &format_range));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_file_range(
          builder, "data_range", &data_range));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_file_range(
          builder, "constants_range", &constants_range));
      return iree_string_builder_append_cstring(builder, "}");
    }
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_SEMAPHORE_OBJECT: {
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_payload_length_check(
          record, sizeof(iree_hal_replay_semaphore_object_payload_t)));
      iree_hal_replay_semaphore_object_payload_t payload;
      memcpy(&payload, record->payload.data, sizeof(payload));
      return iree_string_builder_append_format(
          builder,
          ",\"payload\":{\"queue_affinity\":%" PRIu64
          ",\"initial_value\":%" PRIu64 ",\"flags\":%" PRIu64 "}",
          payload.queue_affinity, payload.initial_value, payload.flags);
    }
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_DISPATCH: {
      if (record->payload.data_length <
          sizeof(iree_hal_replay_dispatch_payload_t)) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "replay dispatch payload is short");
      }
      iree_hal_replay_dispatch_payload_t payload;
      memcpy(&payload, record->payload.data, sizeof(payload));
      iree_host_size_t wait_offset = 0;
      iree_host_size_t wait_size = 0;
      iree_host_size_t signal_offset = 0;
      iree_host_size_t signal_size = 0;
      iree_host_size_t constants_offset = 0;
      iree_host_size_t bindings_offset = 0;
      iree_host_size_t bindings_size = 0;
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_dispatch_layout(
          record, &payload, &wait_offset, &wait_size, &signal_offset,
          &signal_size, &constants_offset, &bindings_offset, &bindings_size));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          ",\"payload\":{\"executable_id\":%" PRIu64
          ",\"queue_affinity\":%" PRIu64 ",\"export_ordinal\":%" PRIu32
          ",\"flags\":%" PRIu32 ",\"workgroup_count\":[%" PRIu32 ",%" PRIu32
          ",%" PRIu32 "],\"workgroup_size\":[%" PRIu32 ",%" PRIu32 ",%" PRIu32
          "],\"dynamic_workgroup_local_memory\":%" PRIu32
          ",\"wait_semaphore_count\":%" PRIu64
          ",\"signal_semaphore_count\":%" PRIu64
          ",\"constants_length\":%" PRIu64 ",\"binding_count\":%" PRIu64,
          payload.executable_id, payload.queue_affinity, payload.export_ordinal,
          payload.flags, payload.workgroup_count[0], payload.workgroup_count[1],
          payload.workgroup_count[2], payload.workgroup_size[0],
          payload.workgroup_size[1], payload.workgroup_size[2],
          payload.dynamic_workgroup_local_memory, payload.wait_semaphore_count,
          payload.signal_semaphore_count, payload.constants_length,
          payload.binding_count));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_buffer_ref(
          builder, "workgroup_count_ref", &payload.workgroup_count_ref));
      iree_hal_replay_file_range_t wait_range =
          iree_hal_replay_dump_payload_subrange(payload_range, wait_offset,
                                                wait_size);
      iree_hal_replay_file_range_t signal_range =
          iree_hal_replay_dump_payload_subrange(payload_range, signal_offset,
                                                signal_size);
      iree_hal_replay_file_range_t constants_range =
          iree_hal_replay_dump_payload_subrange(
              payload_range, constants_offset,
              (iree_host_size_t)payload.constants_length);
      iree_hal_replay_file_range_t bindings_range =
          iree_hal_replay_dump_payload_subrange(payload_range, bindings_offset,
                                                bindings_size);
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_file_range(
          builder, "wait_semaphores_range", &wait_range));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_file_range(
          builder, "signal_semaphores_range", &signal_range));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_file_range(
          builder, "constants_range", &constants_range));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_file_range(
          builder, "bindings_range", &bindings_range));
      return iree_string_builder_append_cstring(builder, "}");
    }
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_EXECUTE: {
      if (record->payload.data_length <
          sizeof(iree_hal_replay_device_queue_execute_payload_t)) {
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "replay queue execute payload is short");
      }
      iree_hal_replay_device_queue_execute_payload_t payload;
      memcpy(&payload, record->payload.data, sizeof(payload));
      iree_host_size_t wait_offset = 0;
      iree_host_size_t wait_size = 0;
      iree_host_size_t signal_offset = 0;
      iree_host_size_t signal_size = 0;
      iree_host_size_t bindings_offset = 0;
      iree_host_size_t bindings_size = 0;
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_queue_execute_layout(
          record, &payload, &wait_offset, &wait_size, &signal_offset,
          &signal_size, &bindings_offset, &bindings_size));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder,
          ",\"payload\":{\"command_buffer_id\":%" PRIu64
          ",\"queue_affinity\":%" PRIu64 ",\"flags\":%" PRIu64
          ",\"wait_semaphore_count\":%" PRIu64
          ",\"signal_semaphore_count\":%" PRIu64 ",\"binding_count\":%" PRIu64,
          payload.command_buffer_id, payload.queue_affinity, payload.flags,
          payload.wait_semaphore_count, payload.signal_semaphore_count,
          payload.binding_count));
      iree_hal_replay_file_range_t wait_range =
          iree_hal_replay_dump_payload_subrange(payload_range, wait_offset,
                                                wait_size);
      iree_hal_replay_file_range_t signal_range =
          iree_hal_replay_dump_payload_subrange(payload_range, signal_offset,
                                                signal_size);
      iree_hal_replay_file_range_t bindings_range =
          iree_hal_replay_dump_payload_subrange(payload_range, bindings_offset,
                                                bindings_size);
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_file_range(
          builder, "wait_semaphores_range", &wait_range));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_file_range(
          builder, "signal_semaphores_range", &signal_range));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_file_range(
          builder, "bindings_range", &bindings_range));
      return iree_string_builder_append_cstring(builder, "}");
    }
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_COPY_BUFFER: {
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_payload_length_check(
          record,
          sizeof(iree_hal_replay_command_buffer_copy_buffer_payload_t)));
      iree_hal_replay_command_buffer_copy_buffer_payload_t payload;
      memcpy(&payload, record->payload.data, sizeof(payload));
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder, ",\"payload\":{\"flags\":%" PRIu32, payload.flags));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_buffer_ref(
          builder, "source", &payload.source_ref));
      IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_buffer_ref(
          builder, "target", &payload.target_ref));
      return iree_string_builder_append_cstring(builder, "}");
    }
    default:
      return iree_string_builder_append_cstring(builder, ",\"payload\":null");
  }
}

static iree_status_t iree_hal_replay_dump_emit_text_record(
    iree_hal_replay_dump_context_t* context, iree_string_builder_t* builder,
    const iree_hal_replay_file_record_t* record,
    const iree_hal_replay_file_range_t* payload_range,
    iree_host_size_t record_offset) {
  const iree_hal_replay_file_record_header_t* header = &record->header;
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder,
      "  @%" PRIhsz " #%" PRIu64 " %-11s dev=%" PRIu64 " obj=%" PRIu64
      " rel=%" PRIu64 " thread=%" PRIu64 " status=%s",
      record_offset, header->sequence_ordinal,
      iree_hal_replay_file_record_type_string(header->record_type),
      header->device_id, header->object_id, header->related_object_id,
      header->thread_id,
      iree_status_code_string((iree_status_code_t)header->status_code)));
  if (header->object_type != IREE_HAL_REPLAY_OBJECT_TYPE_NONE) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, " object=%s(%u)",
        iree_hal_replay_object_type_string(header->object_type),
        header->object_type));
  }
  if (header->operation_code != IREE_HAL_REPLAY_OPERATION_CODE_NONE) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, " op=%s(%u)",
        iree_hal_replay_operation_code_string(header->operation_code),
        header->operation_code));
  }
  if (header->payload_type != IREE_HAL_REPLAY_PAYLOAD_TYPE_NONE ||
      header->payload_length != 0) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, " payload=%s(%u) range=[%" PRIu64 ", +%" PRIu64 "]",
        iree_hal_replay_payload_type_string(header->payload_type),
        header->payload_type, payload_range->offset, payload_range->length));
    IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_text_payload(
        builder, record, payload_range));
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
  return iree_hal_replay_dump_emit(context, builder);
}

static iree_status_t iree_hal_replay_dump_emit_json_record(
    iree_hal_replay_dump_context_t* context, iree_string_builder_t* builder,
    const iree_hal_replay_file_record_t* record,
    const iree_hal_replay_file_range_t* payload_range,
    iree_host_size_t record_offset) {
  const iree_hal_replay_file_record_header_t* header = &record->header;
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, "{\"kind\":"));
  IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_string(
      builder, iree_hal_replay_file_record_type_string(header->record_type)));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder,
      ",\"file_offset\":%" PRIhsz ",\"record_length\":%" PRIu64
      ",\"payload_length\":%" PRIu64 ",\"sequence_ordinal\":%" PRIu64
      ",\"thread_id\":%" PRIu64 ",\"device_id\":%" PRIu64
      ",\"object_id\":%" PRIu64 ",\"related_object_id\":%" PRIu64
      ",\"record_type_code\":%u,\"record_flags\":%u",
      record_offset, header->record_length, header->payload_length,
      header->sequence_ordinal, header->thread_id, header->device_id,
      header->object_id, header->related_object_id, header->record_type,
      header->record_flags));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, ",\"object_type\":"));
  IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_string(
      builder, iree_hal_replay_object_type_string(header->object_type)));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, ",\"object_type_code\":%u", header->object_type));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, ",\"operation\":"));
  IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_string(
      builder, iree_hal_replay_operation_code_string(header->operation_code)));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, ",\"operation_code\":%u,\"status_code\":%u,\"status\":",
      header->operation_code, header->status_code));
  IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_string(
      builder,
      iree_status_code_string((iree_status_code_t)header->status_code)));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, ",\"payload_type\":"));
  IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_string(
      builder, iree_hal_replay_payload_type_string(header->payload_type)));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, ",\"payload_type_code\":%u", header->payload_type));
  IREE_RETURN_IF_ERROR(iree_hal_replay_dump_append_json_file_range(
      builder, "payload_range", payload_range));
  IREE_RETURN_IF_ERROR(
      iree_hal_replay_dump_append_json_payload(builder, record, payload_range));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "}\n"));
  return iree_hal_replay_dump_emit(context, builder);
}

static iree_status_t iree_hal_replay_dump_emit_text_file(
    iree_hal_replay_dump_context_t* context, iree_string_builder_t* builder,
    const iree_hal_replay_file_header_t* header) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder,
      "IREE HAL replay v%u.%u\nfile_length: %" PRIu64
      "\nheader_length: %u\nrecords:\n",
      header->version_major, header->version_minor, header->file_length,
      header->header_length));
  return iree_hal_replay_dump_emit(context, builder);
}

static iree_status_t iree_hal_replay_dump_emit_json_file(
    iree_hal_replay_dump_context_t* context, iree_string_builder_t* builder,
    const iree_hal_replay_file_header_t* header) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder,
      "{\"kind\":\"file\",\"version_major\":%u,\"version_minor\":%u"
      ",\"header_length\":%u,\"flags\":%u,\"file_length\":%" PRIu64 "}\n",
      header->version_major, header->version_minor, header->header_length,
      header->flags, header->file_length));
  return iree_hal_replay_dump_emit(context, builder);
}

IREE_API_EXPORT iree_status_t
iree_hal_replay_dump_file(iree_const_byte_span_t file_contents,
                          const iree_hal_replay_dump_options_t* options,
                          iree_hal_replay_dump_write_fn_t write,
                          void* user_data, iree_allocator_t host_allocator) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(write);
  if (options->format == IREE_HAL_REPLAY_DUMP_FORMAT_C) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "C replay dump emission is reserved for the replay range reader");
  }
  if (IREE_UNLIKELY(options->format != IREE_HAL_REPLAY_DUMP_FORMAT_TEXT &&
                    options->format != IREE_HAL_REPLAY_DUMP_FORMAT_JSONL)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported replay dump format");
  }

  iree_hal_replay_file_header_t file_header;
  iree_host_size_t offset = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_replay_file_parse_header(file_contents, &file_header, &offset));

  iree_const_byte_span_t valid_contents = file_contents;
  if (file_header.file_length != 0) {
    valid_contents.data_length = (iree_host_size_t)file_header.file_length;
  } else {
    file_header.file_length = file_contents.data_length;
  }

  iree_hal_replay_dump_context_t context = {
      .write = write,
      .user_data = user_data,
      .host_allocator = host_allocator,
  };
  iree_string_builder_t builder;
  iree_string_builder_initialize(host_allocator, &builder);

  iree_status_t status = iree_ok_status();
  if (options->format == IREE_HAL_REPLAY_DUMP_FORMAT_TEXT) {
    status =
        iree_hal_replay_dump_emit_text_file(&context, &builder, &file_header);
  } else {
    status =
        iree_hal_replay_dump_emit_json_file(&context, &builder, &file_header);
  }

  uint64_t expected_sequence_ordinal = 0;
  while (iree_status_is_ok(status) && offset < valid_contents.data_length) {
    const iree_host_size_t record_offset = offset;
    iree_hal_replay_file_record_t record;
    status = iree_hal_replay_file_parse_record(valid_contents, record_offset,
                                               &record, &offset);
    if (!iree_status_is_ok(status)) break;

    if (record.header.sequence_ordinal != expected_sequence_ordinal) {
      status = iree_make_status(IREE_STATUS_DATA_LOSS,
                                "replay record sequence ordinal mismatch");
      break;
    }
    ++expected_sequence_ordinal;

    iree_hal_replay_file_range_t payload_range =
        iree_hal_replay_dump_record_payload_range(&record, record_offset);
    if (options->format == IREE_HAL_REPLAY_DUMP_FORMAT_TEXT) {
      status = iree_hal_replay_dump_emit_text_record(
          &context, &builder, &record, &payload_range, record_offset);
    } else {
      status = iree_hal_replay_dump_emit_json_record(
          &context, &builder, &record, &payload_range, record_offset);
    }
  }

  iree_string_builder_deinitialize(&builder);
  return status;
}

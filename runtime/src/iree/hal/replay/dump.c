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

static iree_status_t iree_hal_replay_dump_append_text_payload(
    iree_string_builder_t* builder,
    const iree_hal_replay_file_record_t* record) {
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
    default:
      return iree_ok_status();
  }
}

static iree_status_t iree_hal_replay_dump_append_json_payload(
    iree_string_builder_t* builder,
    const iree_hal_replay_file_record_t* record) {
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
    IREE_RETURN_IF_ERROR(
        iree_hal_replay_dump_append_text_payload(builder, record));
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
      iree_hal_replay_dump_append_json_payload(builder, record));
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

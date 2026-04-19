// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/profile/cat.h"

#include "iree/tooling/profile/reader.h"

static void iree_profile_dump_header_text(
    const iree_hal_profile_file_header_t* header, FILE* file) {
  fprintf(file, "IREE HAL profile bundle\n");
  fprintf(file, "version: %u.%u\n", header->version_major,
          header->version_minor);
  fprintf(file, "header_length: %u\n", header->header_length);
  fprintf(file, "flags: 0x%08x\n", header->flags);
  fprintf(file, "records:\n");
}

static void iree_profile_dump_record_text(
    iree_host_size_t record_index, const iree_hal_profile_file_record_t* record,
    FILE* file) {
  const iree_hal_profile_file_record_header_t* header = &record->header;
  fprintf(file,
          "[%" PRIhsz "] %s record_length=%" PRIu64 " payload_length=%" PRIu64
          "\n",
          record_index, iree_profile_record_type_name(header->record_type),
          header->record_length, header->payload_length);
  fprintf(file, "  content_type: %.*s\n", (int)record->content_type.size,
          record->content_type.data);
  fprintf(file, "  name: %.*s\n", (int)record->name.size, record->name.data);
  fprintf(file,
          "  session_id=%" PRIu64 " stream_id=%" PRIu64 " event_id=%" PRIu64
          "\n",
          header->session_id, header->stream_id, header->event_id);
  fprintf(file, "  executable_id=%" PRIu64 " command_buffer_id=%" PRIu64 "\n",
          header->executable_id, header->command_buffer_id);
  fprintf(file, "  physical_device_ordinal=%u queue_ordinal=%u\n",
          header->physical_device_ordinal, header->queue_ordinal);
  fprintf(file,
          "  chunk_flags=0x%016" PRIx64
          " session_status_code=%u session_status=%s\n",
          header->chunk_flags, header->session_status_code,
          iree_profile_status_code_name(header->session_status_code));
}

static void iree_profile_dump_header_jsonl(
    const iree_hal_profile_file_header_t* header, FILE* file) {
  fprintf(file,
          "{\"type\":\"file\",\"magic\":\"IRPF\","
          "\"version_major\":%u,\"version_minor\":%u,"
          "\"header_length\":%u,\"flags\":%u}\n",
          header->version_major, header->version_minor, header->header_length,
          header->flags);
}

static void iree_profile_dump_record_jsonl(
    iree_host_size_t record_index, const iree_hal_profile_file_record_t* record,
    FILE* file) {
  const iree_hal_profile_file_record_header_t* header = &record->header;
  fprintf(file, "{\"type\":\"record\",\"index\":%" PRIhsz, record_index);
  fprintf(file, ",\"record_type\":");
  iree_profile_fprint_json_string(
      file, iree_make_cstring_view(
                iree_profile_record_type_name(header->record_type)));
  fprintf(file, ",\"record_type_value\":%u", header->record_type);
  fprintf(file, ",\"record_length\":%" PRIu64, header->record_length);
  fprintf(file, ",\"payload_length\":%" PRIu64, header->payload_length);
  fprintf(file, ",\"content_type\":");
  iree_profile_fprint_json_string(file, record->content_type);
  fprintf(file, ",\"name\":");
  iree_profile_fprint_json_string(file, record->name);
  fprintf(file, ",\"session_id\":%" PRIu64, header->session_id);
  fprintf(file, ",\"stream_id\":%" PRIu64, header->stream_id);
  fprintf(file, ",\"event_id\":%" PRIu64, header->event_id);
  fprintf(file, ",\"executable_id\":%" PRIu64, header->executable_id);
  fprintf(file, ",\"command_buffer_id\":%" PRIu64, header->command_buffer_id);
  fprintf(file, ",\"physical_device_ordinal\":%u",
          header->physical_device_ordinal);
  fprintf(file, ",\"queue_ordinal\":%u", header->queue_ordinal);
  fprintf(file, ",\"chunk_flags\":%" PRIu64, header->chunk_flags);
  fprintf(file, ",\"session_status_code\":%u", header->session_status_code);
  fprintf(file, ",\"session_status\":");
  iree_profile_fprint_json_string(
      file, iree_make_cstring_view(
                iree_profile_status_code_name(header->session_status_code)));
  fputs("}\n", file);
}

typedef struct iree_profile_cat_context_t {
  // True when emitting human-readable text instead of JSONL records.
  bool is_text;
  // Output stream receiving the decoded record stream.
  FILE* file;
} iree_profile_cat_context_t;

static iree_status_t iree_profile_cat_record(
    void* user_data, const iree_hal_profile_file_record_t* record,
    iree_host_size_t record_index) {
  iree_profile_cat_context_t* context = (iree_profile_cat_context_t*)user_data;
  if (context->is_text) {
    iree_profile_dump_record_text(record_index, record, context->file);
  } else {
    iree_profile_dump_record_jsonl(record_index, record, context->file);
  }
  return iree_ok_status();
}

iree_status_t iree_profile_cat_file(iree_string_view_t path,
                                    iree_string_view_t format, FILE* file,
                                    iree_allocator_t host_allocator) {
  bool is_text = iree_string_view_equal(format, IREE_SV("text"));
  bool is_jsonl = iree_string_view_equal(format, IREE_SV("jsonl"));
  if (!is_text && !is_jsonl) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported profile output format '%.*s'",
                            (int)format.size, format.data);
  }

  iree_profile_file_t profile_file;
  IREE_RETURN_IF_ERROR(
      iree_profile_file_open(path, host_allocator, &profile_file));

  if (is_text) {
    iree_profile_dump_header_text(&profile_file.header, file);
  } else {
    iree_profile_dump_header_jsonl(&profile_file.header, file);
  }

  iree_profile_cat_context_t context = {
      .is_text = is_text,
      .file = file,
  };
  iree_profile_file_record_callback_t record_callback = {
      .fn = iree_profile_cat_record,
      .user_data = &context,
  };
  iree_status_t status =
      iree_profile_file_for_each_record(&profile_file, record_callback);
  iree_profile_file_close(&profile_file);
  return status;
}

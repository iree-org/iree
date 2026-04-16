// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/utils/profile_file.h"
#include "iree/io/file_contents.h"

IREE_FLAG(string, format, "text",
          "Output format for `cat`: one of `text` or `jsonl`.");

static const char* iree_profile_record_type_name(
    iree_hal_profile_file_record_type_t record_type) {
  switch (record_type) {
    case IREE_HAL_PROFILE_FILE_RECORD_TYPE_SESSION_BEGIN:
      return "session_begin";
    case IREE_HAL_PROFILE_FILE_RECORD_TYPE_CHUNK:
      return "chunk";
    case IREE_HAL_PROFILE_FILE_RECORD_TYPE_SESSION_END:
      return "session_end";
    default:
      return "unknown";
  }
}

static void iree_profile_fprint_json_string(FILE* file,
                                            iree_string_view_t value) {
  fputc('"', file);
  for (iree_host_size_t i = 0; i < value.size; ++i) {
    uint8_t c = (uint8_t)value.data[i];
    switch (c) {
      case '"':
        fputs("\\\"", file);
        break;
      case '\\':
        fputs("\\\\", file);
        break;
      case '\b':
        fputs("\\b", file);
        break;
      case '\f':
        fputs("\\f", file);
        break;
      case '\n':
        fputs("\\n", file);
        break;
      case '\r':
        fputs("\\r", file);
        break;
      case '\t':
        fputs("\\t", file);
        break;
      default:
        if (c < 0x20) {
          fprintf(file, "\\u%04x", c);
        } else {
          fputc(c, file);
        }
        break;
    }
  }
  fputc('"', file);
}

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
  fprintf(file, "  chunk_flags=0x%016" PRIx64 " session_status_code=%u\n",
          header->chunk_flags, header->session_status_code);
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
  fputs("}\n", file);
}

static iree_status_t iree_profile_cat_file(iree_string_view_t path,
                                           iree_string_view_t format,
                                           FILE* file,
                                           iree_allocator_t host_allocator) {
  bool is_text = iree_string_view_equal(format, IREE_SV("text"));
  bool is_jsonl = iree_string_view_equal(format, IREE_SV("jsonl"));
  if (!is_text && !is_jsonl) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported profile output format '%.*s'",
                            (int)format.size, format.data);
  }

  iree_io_file_contents_t* file_contents = NULL;
  iree_status_t status = iree_io_file_contents_map(
      path, IREE_IO_FILE_ACCESS_READ, host_allocator, &file_contents);

  iree_hal_profile_file_header_t header;
  iree_host_size_t record_offset = 0;
  if (iree_status_is_ok(status)) {
    status = iree_hal_profile_file_parse_header(file_contents->const_buffer,
                                                &header, &record_offset);
  }
  if (iree_status_is_ok(status)) {
    if (is_text) {
      iree_profile_dump_header_text(&header, file);
    } else {
      iree_profile_dump_header_jsonl(&header, file);
    }
  }

  iree_host_size_t record_index = 0;
  while (iree_status_is_ok(status) &&
         record_offset < file_contents->const_buffer.data_length) {
    iree_hal_profile_file_record_t record;
    iree_host_size_t next_record_offset = 0;
    status = iree_hal_profile_file_parse_record(file_contents->const_buffer,
                                                record_offset, &record,
                                                &next_record_offset);
    if (iree_status_is_ok(status)) {
      if (is_text) {
        iree_profile_dump_record_text(record_index, &record, file);
      } else {
        iree_profile_dump_record_jsonl(record_index, &record, file);
      }
      record_offset = next_record_offset;
      ++record_index;
    }
  }

  iree_io_file_contents_free(file_contents);
  return status;
}

int main(int argc, char** argv) {
  IREE_TRACE_APP_ENTER();
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_t host_allocator = iree_allocator_system();
  int exit_code = EXIT_SUCCESS;

  iree_flags_set_usage(
      "iree-profile",
      "Inspects IREE HAL profile bundles.\n"
      "\n"
      "Usage:\n"
      "  iree-profile cat [--format=text|jsonl] <file.ireeprof>\n"
      "  iree-profile [--format=text|jsonl] <file.ireeprof>\n");
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);

  iree_string_view_t command = IREE_SV("cat");
  iree_string_view_t path = iree_string_view_empty();
  if (argc == 2) {
    path = iree_make_cstring_view(argv[1]);
  } else if (argc == 3) {
    command = iree_make_cstring_view(argv[1]);
    path = iree_make_cstring_view(argv[2]);
  }

  iree_status_t status = iree_ok_status();
  if (argc != 2 && argc != 3) {
    fprintf(stderr,
            "Error: expected profile bundle path.\n"
            "Usage: iree-profile cat [--format=text|jsonl] <file.ireeprof>\n");
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "expected profile bundle path");
  } else if (iree_string_view_is_empty(path)) {
    fprintf(stderr,
            "Error: missing profile bundle path.\n"
            "Usage: iree-profile cat [--format=text|jsonl] <file.ireeprof>\n");
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "missing profile bundle path");
  } else if (!iree_string_view_equal(command, IREE_SV("cat"))) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported iree-profile command '%.*s'",
                              (int)command.size, command.data);
  }

  if (iree_status_is_ok(status)) {
    status = iree_profile_cat_file(path, iree_make_cstring_view(FLAG_format),
                                   stdout, host_allocator);
  }

  fflush(stdout);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    exit_code = EXIT_FAILURE;
  }
  fflush(stderr);

  IREE_TRACE_ZONE_END(z0);
  IREE_TRACE_APP_EXIT(exit_code);
  return exit_code;
}

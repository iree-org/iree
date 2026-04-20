// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>

#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/replay/dump.h"
#include "iree/io/file_contents.h"

IREE_FLAG(string, format, "text", "Output format: text, jsonl, or c.");

static iree_status_t iree_dump_replay_parse_format(
    const char* value, iree_hal_replay_dump_format_t* out_format) {
  iree_string_view_t format = iree_make_cstring_view(value);
  if (iree_string_view_equal(format, IREE_SV("text"))) {
    *out_format = IREE_HAL_REPLAY_DUMP_FORMAT_TEXT;
    return iree_ok_status();
  } else if (iree_string_view_equal(format, IREE_SV("jsonl"))) {
    *out_format = IREE_HAL_REPLAY_DUMP_FORMAT_JSONL;
    return iree_ok_status();
  } else if (iree_string_view_equal(format, IREE_SV("c"))) {
    *out_format = IREE_HAL_REPLAY_DUMP_FORMAT_C;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unsupported --format=%s; expected text, jsonl, or c",
                          value);
}

static iree_status_t iree_dump_replay_write_file(void* user_data,
                                                 iree_string_view_t text) {
  FILE* file = (FILE*)user_data;
  if (text.size == 0) return iree_ok_status();
  if (IREE_UNLIKELY(fwrite(text.data, 1, text.size, file) != text.size)) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "failed to write replay dump output");
  }
  return iree_ok_status();
}

int main(int argc, char** argv) {
  IREE_TRACE_APP_ENTER();
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_t host_allocator = iree_allocator_system();
  int exit_code = EXIT_SUCCESS;

  iree_flags_set_usage(
      "iree-dump-replay",
      "Dumps information from an IREE HAL replay file.\n"
      "\n"
      "The text format is intended for humans. The jsonl format emits one\n"
      "object per line and reports blob data as byte ranges in the original\n"
      ".ireereplay file.\n");
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);

  iree_status_t status = iree_ok_status();
  if (argc != 2) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "expected one replay file path argument");
  }

  iree_hal_replay_dump_options_t options =
      iree_hal_replay_dump_options_default();
  if (iree_status_is_ok(status)) {
    status = iree_dump_replay_parse_format(FLAG_format, &options.format);
  }

  iree_io_file_contents_t* file_contents = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_io_file_contents_map(iree_make_cstring_view(argv[1]),
                                       IREE_IO_FILE_ACCESS_READ, host_allocator,
                                       &file_contents);
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_dump_file(file_contents->const_buffer, &options,
                                       iree_dump_replay_write_file, stdout,
                                       host_allocator);
  }

  iree_io_file_contents_free(file_contents);

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

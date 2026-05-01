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

IREE_FLAG(string, format, "text", "Output format: text or jsonl.");
IREE_FLAG(bool, agents_md, false,
          "Prints AGENTS.md guidance for iree-dump-replay and exits.");

static const char kIreeDumpReplayUsage[] =
    "Dumps information from an IREE HAL replay file.\n"
    "\n"
    "The dumper validates the replay container and emits projections without\n"
    "materializing large payload bytes. Blob data and embedded payloads are\n"
    "reported as byte ranges in the original .ireereplay file so JSONL and\n"
    "text output can reference the capture directly.\n"
    "\n"
    "Usage:\n"
    "  iree-dump-replay [--format=text|jsonl] <capture.ireereplay>\n"
    "\n"
    "Formats:\n"
    "  text\n"
    "      Human-readable record stream for quick inspection.\n"
    "  jsonl\n"
    "      One JSON object per line for jq and automation. Payload bytes are\n"
    "      represented as replay-file ranges.\n"
    "\n"
    "Important flags:\n"
    "  --agents_md\n"
    "      Prints AGENTS.md guidance specific to iree-dump-replay. Use\n"
    "      `iree-run-replay --agents_md` for the full replay tool playbook.\n"
    "\n"
    "Examples:\n"
    "  iree-dump-replay --format=text /tmp/model.ireereplay\n"
    "  iree-dump-replay --format=jsonl /tmp/model.ireereplay | \\\n"
    "      jq 'select(.kind==\"operation\" and "
    ".operation==\"device.queue_execute\")'\n";

static void iree_dump_replay_print_agent_markdown(FILE* file) {
  fputs(
      "# iree-dump-replay\n"
      "\n"
      "Use `iree-dump-replay --format=text` for quick inspection and\n"
      "`--format=jsonl` for agent workflows. JSONL emits one object per line "
      "and\n"
      "keeps payload bytes as replay-file ranges instead of dumping raw blob "
      "data.\n"
      "\n"
      "Useful queries:\n"
      "\n"
      "```bash\n"
      "iree-dump-replay --format=jsonl /tmp/model.ireereplay | \\\n"
      "  jq 'select(.kind==\"operation\" and "
      ".operation==\"device.queue_execute\")'\n"
      "iree-dump-replay --format=jsonl /tmp/model.ireereplay | \\\n"
      "  jq 'select(.payload_type==\"replay_scope\") | .payload.name'\n"
      "```\n"
      "\n"
      "Rows carry object ids, operation ids, queue wait/signal lists, buffer\n"
      "binding tables, file references, and replay-file byte ranges.\n"
      "\n"
      "For replay execution, executable substitution, file remapping, and the\n"
      "shared replay failure contract, pipe `iree-run-replay --agents_md` "
      "into\n"
      "your AGENTS.md.\n",
      file);
}

static iree_status_t iree_dump_replay_parse_format(
    const char* value, iree_hal_replay_dump_format_t* out_format) {
  iree_string_view_t format = iree_make_cstring_view(value);
  if (iree_string_view_equal(format, IREE_SV("text"))) {
    *out_format = IREE_HAL_REPLAY_DUMP_FORMAT_TEXT;
    return iree_ok_status();
  } else if (iree_string_view_equal(format, IREE_SV("jsonl"))) {
    *out_format = IREE_HAL_REPLAY_DUMP_FORMAT_JSONL;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unsupported --format=%s; expected text or jsonl",
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

  iree_flags_set_usage("iree-dump-replay", kIreeDumpReplayUsage);
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);
  if (FLAG_agents_md) {
    iree_dump_replay_print_agent_markdown(stdout);
    fflush(stdout);
    IREE_TRACE_ZONE_END(z0);
    IREE_TRACE_APP_EXIT(exit_code);
    return exit_code;
  }

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
    iree_hal_replay_dump_write_callback_t write_callback = {
        .fn = iree_dump_replay_write_file,
        .user_data = stdout,
    };
    status = iree_hal_replay_dump_file(file_contents->const_buffer, &options,
                                       write_callback, host_allocator);
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

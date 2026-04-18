// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>
#include <stdlib.h>

#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"
#include "iree/tooling/profile/commands.h"
#include "iree/tooling/profile/help.h"

#ifndef IREE_PROFILE_HAVE_AMDGPU_ATT
#define IREE_PROFILE_HAVE_AMDGPU_ATT 0
#endif  // IREE_PROFILE_HAVE_AMDGPU_ATT

#if IREE_PROFILE_HAVE_AMDGPU_ATT
#include "iree/tooling/profile/att/att.h"
#endif  // IREE_PROFILE_HAVE_AMDGPU_ATT

IREE_FLAG(string, format, "text",
          "Output format for the selected command. Report commands support "
          "`text` and `jsonl`; export supports `ireeperf-jsonl`.");
IREE_FLAG(string, filter, "*",
          "Name/key wildcard filter for commands that consume named rows.");
IREE_FLAG(string, output, "-",
          "Output file path for export commands, or `-` for stdout.");
IREE_FLAG(int64_t, id, -1,
          "Optional id filter interpreted by commands that accept ids.");
IREE_FLAG(bool, dispatch_events, false,
          "Emits individual dispatch event rows for projection commands with "
          "`--format=jsonl`.");
IREE_FLAG(bool, counter_samples, false,
          "Emits individual counter sample rows for the counter command with "
          "`--format=jsonl`.");
IREE_FLAG(bool, agents_md, false,
          "Prints an agent-oriented Markdown guide for iree-profile JSONL "
          "workflows and exits.");
IREE_FLAG(string, rocm_library_path, "",
          "ROCm library directory or exact dynamic library path used by ATT "
          "decode. Overrides IREE_HAL_AMDGPU_LIBAQLPROFILE_PATH and "
          "IREE_HAL_AMDGPU_LIBHSA_PATH.");

int main(int argc, char** argv) {
  IREE_TRACE_APP_ENTER();
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_t host_allocator = iree_allocator_system();
  int exit_code = EXIT_SUCCESS;
  const iree_profile_command_t* const commands[] = {
      iree_profile_summary_command(),    iree_profile_explain_command(),
      iree_profile_executable_command(), iree_profile_dispatch_command(),
      iree_profile_command_command(),    iree_profile_counter_command(),
      iree_profile_memory_command(),     iree_profile_queue_command(),
      iree_profile_export_command(),     iree_profile_cat_command(),
#if IREE_PROFILE_HAVE_AMDGPU_ATT
      iree_profile_att_command(),
#endif  // IREE_PROFILE_HAVE_AMDGPU_ATT
  };

  iree_flags_set_usage("iree-profile", iree_profile_usage_text());
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);
  if (FLAG_agents_md) {
    iree_profile_print_agent_markdown(stdout);
    fflush(stdout);
    IREE_TRACE_ZONE_END(z0);
    IREE_TRACE_APP_EXIT(exit_code);
    return exit_code;
  }

  iree_string_view_t command_name = IREE_SV("cat");
  iree_string_view_t input_path = iree_string_view_empty();
  if (argc == 2) {
    input_path = iree_make_cstring_view(argv[1]);
  } else if (argc == 3) {
    command_name = iree_make_cstring_view(argv[1]);
    input_path = iree_make_cstring_view(argv[2]);
  }

  iree_status_t status = iree_ok_status();
  const iree_profile_command_t* command = NULL;
  if (argc != 2 && argc != 3) {
    fprintf(stderr, "Error: expected profile bundle path.\n");
    iree_profile_fprint_usage(stderr);
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "expected profile bundle path");
  } else if (iree_string_view_is_empty(input_path)) {
    fprintf(stderr, "Error: missing profile bundle path.\n");
    iree_profile_fprint_usage(stderr);
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "missing profile bundle path");
  } else {
    command = iree_profile_find_command(commands, IREE_ARRAYSIZE(commands),
                                        command_name);
    if (!command) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "unsupported iree-profile command '%.*s'",
                                (int)command_name.size, command_name.data);
    }
  }

  if (iree_status_is_ok(status)) {
    const iree_profile_command_options_t options = {
        .format = iree_make_cstring_view(FLAG_format),
        .filter = iree_make_cstring_view(FLAG_filter),
        .output_path = iree_make_cstring_view(FLAG_output),
        .rocm_library_path = iree_make_cstring_view(FLAG_rocm_library_path),
        .id = FLAG_id,
        .emit_dispatch_events = FLAG_dispatch_events,
        .emit_counter_samples = FLAG_counter_samples,
        .host_allocator = host_allocator,
    };
    status = iree_profile_command_validate_options(command, &options);
    if (iree_status_is_ok(status)) {
      const iree_profile_command_invocation_t invocation = {
          .input_path = input_path,
          .output_file = stdout,
          .options = &options,
      };
      status = command->run(&invocation);
    }
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

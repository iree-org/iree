// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOOLING_PROFILE_COMMANDS_H_
#define IREE_TOOLING_PROFILE_COMMANDS_H_

#include <stdio.h>

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_profile_command_options_t {
  // Output format requested by the caller, such as "text" or "jsonl".
  iree_string_view_t format;
  // Wildcard filter used by commands that project named rows.
  iree_string_view_t filter;
  // Export output path, or "-" for stdout.
  iree_string_view_t output_path;
  // ROCm dynamic library directory/path used by optional ATT decode.
  iree_string_view_t rocm_library_path;
  // Optional command-specific numeric id filter, or -1 when absent.
  int64_t id;
  // Emits individual dispatch_event rows for projection commands.
  bool emit_dispatch_events;
  // Emits individual counter_sample rows for the counter command.
  bool emit_counter_samples;
  // Host allocator used for command-owned temporary allocations.
  iree_allocator_t host_allocator;
} iree_profile_command_options_t;

// Bitmask of output encodings a command can produce.
enum iree_profile_command_format_bits_e {
  IREE_PROFILE_COMMAND_FORMAT_NONE = 0u,
  IREE_PROFILE_COMMAND_FORMAT_TEXT = 1u << 0,
  IREE_PROFILE_COMMAND_FORMAT_JSONL = 1u << 1,
  IREE_PROFILE_COMMAND_FORMAT_IREEPERF_JSONL = 1u << 2,
};
typedef uint32_t iree_profile_command_format_bits_t;

// Bitmask of shared command options a command consumes.
enum iree_profile_command_option_bits_e {
  IREE_PROFILE_COMMAND_OPTION_NONE = 0u,
  IREE_PROFILE_COMMAND_OPTION_FILTER = 1u << 0,
  IREE_PROFILE_COMMAND_OPTION_ID = 1u << 1,
  IREE_PROFILE_COMMAND_OPTION_OUTPUT = 1u << 2,
  IREE_PROFILE_COMMAND_OPTION_DISPATCH_EVENTS = 1u << 3,
  IREE_PROFILE_COMMAND_OPTION_COUNTER_SAMPLES = 1u << 4,
  IREE_PROFILE_COMMAND_OPTION_ROCM_LIBRARY_PATH = 1u << 5,
};
typedef uint32_t iree_profile_command_option_bits_t;

typedef struct iree_profile_command_invocation_t {
  // Input .ireeprof bundle path.
  iree_string_view_t input_path;
  // Primary command output stream.
  FILE* output_file;
  // Borrowed command options.
  const iree_profile_command_options_t* options;
} iree_profile_command_invocation_t;

typedef iree_status_t (*iree_profile_command_run_fn_t)(
    const iree_profile_command_invocation_t* invocation);

typedef struct iree_profile_command_t {
  // CLI command name.
  const char* name;
  // One-line human-readable command summary.
  const char* summary;
  // Output formats accepted by this command.
  iree_profile_command_format_bits_t supported_formats;
  // Shared command options consumed by this command.
  iree_profile_command_option_bits_t accepted_options;
  // Runs the command for one input profile bundle.
  iree_profile_command_run_fn_t run;
} iree_profile_command_t;

const iree_profile_command_t* iree_profile_cat_command(void);
const iree_profile_command_t* iree_profile_command_command(void);
const iree_profile_command_t* iree_profile_counter_command(void);
const iree_profile_command_t* iree_profile_dispatch_command(void);
const iree_profile_command_t* iree_profile_executable_command(void);
const iree_profile_command_t* iree_profile_explain_command(void);
const iree_profile_command_t* iree_profile_export_command(void);
const iree_profile_command_t* iree_profile_memory_command(void);
const iree_profile_command_t* iree_profile_queue_command(void);
const iree_profile_command_t* iree_profile_statistics_command(void);
const iree_profile_command_t* iree_profile_summary_command(void);

// Validates |options| against |command| without opening the input bundle.
iree_status_t iree_profile_command_validate_options(
    const iree_profile_command_t* command,
    const iree_profile_command_options_t* options);

const iree_profile_command_t* iree_profile_find_command(
    const iree_profile_command_t* const* commands,
    iree_host_size_t command_count, iree_string_view_t name);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TOOLING_PROFILE_COMMANDS_H_

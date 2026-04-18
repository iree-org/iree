// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/profile/commands.h"

#include "iree/tooling/profile/cat.h"
#include "iree/tooling/profile/counter.h"
#include "iree/tooling/profile/explain.h"
#include "iree/tooling/profile/export.h"
#include "iree/tooling/profile/memory.h"
#include "iree/tooling/profile/projection.h"
#include "iree/tooling/profile/summary.h"

#define IREE_PROFILE_COMMAND_REPORT_FORMATS \
  (IREE_PROFILE_COMMAND_FORMAT_TEXT | IREE_PROFILE_COMMAND_FORMAT_JSONL)

#define IREE_PROFILE_COMMAND_PROJECTION_OPTIONS                          \
  (IREE_PROFILE_COMMAND_OPTION_FILTER | IREE_PROFILE_COMMAND_OPTION_ID | \
   IREE_PROFILE_COMMAND_OPTION_DISPATCH_EVENTS)

static iree_profile_command_format_bits_t iree_profile_command_format_bit(
    iree_string_view_t format) {
  if (iree_string_view_equal(format, IREE_SV("text"))) {
    return IREE_PROFILE_COMMAND_FORMAT_TEXT;
  } else if (iree_string_view_equal(format, IREE_SV("jsonl"))) {
    return IREE_PROFILE_COMMAND_FORMAT_JSONL;
  } else if (iree_string_view_equal(format, IREE_SV("ireeperf-jsonl"))) {
    return IREE_PROFILE_COMMAND_FORMAT_IREEPERF_JSONL;
  }
  return IREE_PROFILE_COMMAND_FORMAT_NONE;
}

static bool iree_profile_command_accepts_option(
    const iree_profile_command_t* command,
    iree_profile_command_option_bits_t option) {
  return iree_all_bits_set(command->accepted_options, option);
}

static bool iree_profile_command_filter_is_default(iree_string_view_t filter) {
  return iree_string_view_is_empty(filter) ||
         iree_string_view_equal(filter, IREE_SV("*"));
}

static bool iree_profile_command_output_path_is_default(
    iree_string_view_t output_path) {
  return iree_string_view_is_empty(output_path) ||
         iree_string_view_equal(output_path, IREE_SV("-"));
}

static iree_status_t iree_profile_command_require_option(
    const iree_profile_command_t* command,
    iree_profile_command_option_bits_t option, const char* flag_name) {
  if (iree_profile_command_accepts_option(command, option)) {
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "iree-profile command '%s' does not support %s",
                          command->name, flag_name);
}

static iree_status_t iree_profile_command_require_jsonl_format(
    iree_profile_command_format_bits_t format, const char* flag_name) {
  if (format == IREE_PROFILE_COMMAND_FORMAT_JSONL) return iree_ok_status();
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "%s requires --format=jsonl", flag_name);
}

static iree_status_t iree_profile_run_cat(
    const iree_profile_command_invocation_t* invocation) {
  return iree_profile_cat_file(
      invocation->input_path, invocation->options->format,
      invocation->output_file, invocation->options->host_allocator);
}

static iree_status_t iree_profile_run_summary(
    const iree_profile_command_invocation_t* invocation) {
  return iree_profile_summary_file(
      invocation->input_path, invocation->options->format,
      invocation->output_file, invocation->options->host_allocator);
}

static iree_status_t iree_profile_run_explain(
    const iree_profile_command_invocation_t* invocation) {
  return iree_profile_explain_file(
      invocation->input_path, invocation->options->format,
      invocation->options->filter, invocation->options->id,
      invocation->output_file, invocation->options->host_allocator);
}

static iree_status_t iree_profile_run_export(
    const iree_profile_command_invocation_t* invocation) {
  return iree_profile_export_file(
      invocation->input_path, invocation->options->format,
      invocation->options->output_path, invocation->options->host_allocator);
}

static iree_status_t iree_profile_run_command(
    const iree_profile_command_invocation_t* invocation) {
  return iree_profile_projection_file(
      invocation->input_path, invocation->options->format,
      invocation->options->filter, IREE_PROFILE_PROJECTION_MODE_COMMAND,
      invocation->options->id, invocation->options->emit_dispatch_events,
      invocation->output_file, invocation->options->host_allocator);
}

static iree_status_t iree_profile_run_counter(
    const iree_profile_command_invocation_t* invocation) {
  return iree_profile_counter_file(
      invocation->input_path, invocation->options->format,
      invocation->options->filter, invocation->options->id,
      invocation->options->emit_counter_samples, invocation->output_file,
      invocation->options->host_allocator);
}

static iree_status_t iree_profile_run_dispatch(
    const iree_profile_command_invocation_t* invocation) {
  return iree_profile_projection_file(
      invocation->input_path, invocation->options->format,
      invocation->options->filter, IREE_PROFILE_PROJECTION_MODE_DISPATCH,
      invocation->options->id, invocation->options->emit_dispatch_events,
      invocation->output_file, invocation->options->host_allocator);
}

static iree_status_t iree_profile_run_executable(
    const iree_profile_command_invocation_t* invocation) {
  return iree_profile_projection_file(
      invocation->input_path, invocation->options->format,
      invocation->options->filter, IREE_PROFILE_PROJECTION_MODE_EXECUTABLE,
      invocation->options->id, invocation->options->emit_dispatch_events,
      invocation->output_file, invocation->options->host_allocator);
}

static iree_status_t iree_profile_run_memory(
    const iree_profile_command_invocation_t* invocation) {
  return iree_profile_memory_report_file(
      invocation->input_path, invocation->options->format,
      invocation->options->filter, invocation->options->id,
      invocation->output_file, invocation->options->host_allocator);
}

static iree_status_t iree_profile_run_queue(
    const iree_profile_command_invocation_t* invocation) {
  return iree_profile_projection_file(
      invocation->input_path, invocation->options->format,
      invocation->options->filter, IREE_PROFILE_PROJECTION_MODE_QUEUE,
      invocation->options->id, invocation->options->emit_dispatch_events,
      invocation->output_file, invocation->options->host_allocator);
}

static const iree_profile_command_t kIreeProfileCatCommand = {
    .name = "cat",
    .summary = "Raw bundle record dump for format archaeology/debugging.",
    .supported_formats = IREE_PROFILE_COMMAND_REPORT_FORMATS,
    .accepted_options = IREE_PROFILE_COMMAND_OPTION_NONE,
    .run = iree_profile_run_cat,
};
static const iree_profile_command_t kIreeProfileCommandCommand = {
    .name = "command",
    .summary = "Recorded command-buffer operations and execution spans.",
    .supported_formats = IREE_PROFILE_COMMAND_REPORT_FORMATS,
    .accepted_options = IREE_PROFILE_COMMAND_PROJECTION_OPTIONS,
    .run = iree_profile_run_command,
};
static const iree_profile_command_t kIreeProfileCounterCommand = {
    .name = "counter",
    .summary = "Hardware counter metadata and sample aggregates.",
    .supported_formats = IREE_PROFILE_COMMAND_REPORT_FORMATS,
    .accepted_options = IREE_PROFILE_COMMAND_OPTION_FILTER |
                        IREE_PROFILE_COMMAND_OPTION_ID |
                        IREE_PROFILE_COMMAND_OPTION_COUNTER_SAMPLES,
    .run = iree_profile_run_counter,
};
static const iree_profile_command_t kIreeProfileDispatchCommand = {
    .name = "dispatch",
    .summary = "Per-export dispatch timing aggregates or event rows.",
    .supported_formats = IREE_PROFILE_COMMAND_REPORT_FORMATS,
    .accepted_options = IREE_PROFILE_COMMAND_PROJECTION_OPTIONS,
    .run = iree_profile_run_dispatch,
};
static const iree_profile_command_t kIreeProfileExecutableCommand = {
    .name = "executable",
    .summary = "Executable/export catalog joined with dispatch timing.",
    .supported_formats = IREE_PROFILE_COMMAND_REPORT_FORMATS,
    .accepted_options = IREE_PROFILE_COMMAND_PROJECTION_OPTIONS,
    .run = iree_profile_run_executable,
};
static const iree_profile_command_t kIreeProfileExplainCommand = {
    .name = "explain",
    .summary = "Opinionated bottleneck summary and evidence-backed hints.",
    .supported_formats = IREE_PROFILE_COMMAND_REPORT_FORMATS,
    .accepted_options =
        IREE_PROFILE_COMMAND_OPTION_FILTER | IREE_PROFILE_COMMAND_OPTION_ID,
    .run = iree_profile_run_explain,
};
static const iree_profile_command_t kIreeProfileExportCommand = {
    .name = "export",
    .summary = "Decoded tooling interchange export.",
    .supported_formats = IREE_PROFILE_COMMAND_FORMAT_IREEPERF_JSONL,
    .accepted_options = IREE_PROFILE_COMMAND_OPTION_OUTPUT,
    .run = iree_profile_run_export,
};
static const iree_profile_command_t kIreeProfileMemoryCommand = {
    .name = "memory",
    .summary = "Memory lifecycle events and high-water summaries.",
    .supported_formats = IREE_PROFILE_COMMAND_REPORT_FORMATS,
    .accepted_options =
        IREE_PROFILE_COMMAND_OPTION_FILTER | IREE_PROFILE_COMMAND_OPTION_ID,
    .run = iree_profile_run_memory,
};
static const iree_profile_command_t kIreeProfileQueueCommand = {
    .name = "queue",
    .summary = "Queue operation events and dispatch-derived submission spans.",
    .supported_formats = IREE_PROFILE_COMMAND_REPORT_FORMATS,
    .accepted_options = IREE_PROFILE_COMMAND_PROJECTION_OPTIONS,
    .run = iree_profile_run_queue,
};
static const iree_profile_command_t kIreeProfileSummaryCommand = {
    .name = "summary",
    .summary = "Bundle health, metadata counts, clock fit, and timing totals.",
    .supported_formats = IREE_PROFILE_COMMAND_REPORT_FORMATS,
    .accepted_options = IREE_PROFILE_COMMAND_OPTION_NONE,
    .run = iree_profile_run_summary,
};

const iree_profile_command_t* iree_profile_cat_command(void) {
  return &kIreeProfileCatCommand;
}
const iree_profile_command_t* iree_profile_command_command(void) {
  return &kIreeProfileCommandCommand;
}
const iree_profile_command_t* iree_profile_counter_command(void) {
  return &kIreeProfileCounterCommand;
}
const iree_profile_command_t* iree_profile_dispatch_command(void) {
  return &kIreeProfileDispatchCommand;
}
const iree_profile_command_t* iree_profile_executable_command(void) {
  return &kIreeProfileExecutableCommand;
}
const iree_profile_command_t* iree_profile_explain_command(void) {
  return &kIreeProfileExplainCommand;
}
const iree_profile_command_t* iree_profile_export_command(void) {
  return &kIreeProfileExportCommand;
}
const iree_profile_command_t* iree_profile_memory_command(void) {
  return &kIreeProfileMemoryCommand;
}
const iree_profile_command_t* iree_profile_queue_command(void) {
  return &kIreeProfileQueueCommand;
}
const iree_profile_command_t* iree_profile_summary_command(void) {
  return &kIreeProfileSummaryCommand;
}

iree_status_t iree_profile_command_validate_options(
    const iree_profile_command_t* command,
    const iree_profile_command_options_t* options) {
  const iree_profile_command_format_bits_t format =
      iree_profile_command_format_bit(options->format);
  if (!iree_any_bit_set(command->supported_formats, format)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "iree-profile command '%s' does not support "
                            "--format=%.*s",
                            command->name, (int)options->format.size,
                            options->format.data);
  }
  if (!iree_profile_command_filter_is_default(options->filter)) {
    IREE_RETURN_IF_ERROR(iree_profile_command_require_option(
        command, IREE_PROFILE_COMMAND_OPTION_FILTER, "--filter"));
  }
  if (options->id >= 0) {
    IREE_RETURN_IF_ERROR(iree_profile_command_require_option(
        command, IREE_PROFILE_COMMAND_OPTION_ID, "--id"));
  }
  if (!iree_profile_command_output_path_is_default(options->output_path)) {
    IREE_RETURN_IF_ERROR(iree_profile_command_require_option(
        command, IREE_PROFILE_COMMAND_OPTION_OUTPUT, "--output"));
  }
  if (options->emit_dispatch_events) {
    IREE_RETURN_IF_ERROR(iree_profile_command_require_option(
        command, IREE_PROFILE_COMMAND_OPTION_DISPATCH_EVENTS,
        "--dispatch_events"));
    IREE_RETURN_IF_ERROR(
        iree_profile_command_require_jsonl_format(format, "--dispatch_events"));
  }
  if (options->emit_counter_samples) {
    IREE_RETURN_IF_ERROR(iree_profile_command_require_option(
        command, IREE_PROFILE_COMMAND_OPTION_COUNTER_SAMPLES,
        "--counter_samples"));
    IREE_RETURN_IF_ERROR(
        iree_profile_command_require_jsonl_format(format, "--counter_samples"));
  }
  if (!iree_string_view_is_empty(options->rocm_library_path)) {
    IREE_RETURN_IF_ERROR(iree_profile_command_require_option(
        command, IREE_PROFILE_COMMAND_OPTION_ROCM_LIBRARY_PATH,
        "--rocm_library_path"));
  }
  return iree_ok_status();
}

const iree_profile_command_t* iree_profile_find_command(
    const iree_profile_command_t* const* commands,
    iree_host_size_t command_count, iree_string_view_t name) {
  for (iree_host_size_t i = 0; i < command_count; ++i) {
    if (iree_string_view_equal(name,
                               iree_make_cstring_view(commands[i]->name))) {
      return commands[i];
    }
  }
  return NULL;
}

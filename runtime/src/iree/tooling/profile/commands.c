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
    "cat", "Raw bundle record dump for format archaeology/debugging.",
    iree_profile_run_cat};
static const iree_profile_command_t kIreeProfileCommandCommand = {
    "command", "Recorded command-buffer operations and execution spans.",
    iree_profile_run_command};
static const iree_profile_command_t kIreeProfileCounterCommand = {
    "counter", "Hardware counter metadata and sample aggregates.",
    iree_profile_run_counter};
static const iree_profile_command_t kIreeProfileDispatchCommand = {
    "dispatch", "Per-export dispatch timing aggregates or event rows.",
    iree_profile_run_dispatch};
static const iree_profile_command_t kIreeProfileExecutableCommand = {
    "executable", "Executable/export catalog joined with dispatch timing.",
    iree_profile_run_executable};
static const iree_profile_command_t kIreeProfileExplainCommand = {
    "explain", "Opinionated bottleneck summary and evidence-backed hints.",
    iree_profile_run_explain};
static const iree_profile_command_t kIreeProfileExportCommand = {
    "export", "Decoded tooling interchange export.", iree_profile_run_export};
static const iree_profile_command_t kIreeProfileMemoryCommand = {
    "memory", "Memory lifecycle events and high-water summaries.",
    iree_profile_run_memory};
static const iree_profile_command_t kIreeProfileQueueCommand = {
    "queue", "Queue operation events and dispatch-derived submission spans.",
    iree_profile_run_queue};
static const iree_profile_command_t kIreeProfileSummaryCommand = {
    "summary", "Bundle health, metadata counts, clock fit, and timing totals.",
    iree_profile_run_summary};

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

// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// DISCLAIMER: this is leaky under error conditions as it's a benchmark tool and
// not a correctness test.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/internal/file_io.h"
#include "iree/base/internal/flags.h"
#include "iree/base/internal/path.h"
#include "iree/hal/api.h"
#include "iree/testing/benchmark.h"
#include "iree/tooling/device_util.h"
#include "iree/tooling/trace_replay.h"
#include "iree/tooling/vm_util.h"
#include "iree/tooling/yaml_util.h"
#include "iree/vm/api.h"

IREE_FLAG(bool, capture_stdin, false,
          "Captures stdin up to EOF on startup to use during trace execution.");

IREE_FLAG(bool, print_statistics, false,
          "Prints runtime statistics to stderr on exit.");

IREE_FLAG(bool, reuse_devices, true,
          "Only creates a HAL device once and reuses it for all iterations.");
IREE_FLAG(bool, reuse_modules, true,
          "Only loads modules once and reuses them for all iterations.");

IREE_FLAG_LIST(
    string, input,
    "An input (a) value or (b) buffer of the format:\n"
    "  (a) scalar value\n"
    "     value\n"
    "     e.g.: --input=\"3.14\"\n"
    "  (b) buffer:\n"
    "     [shape]xtype=[value]\n"
    "     e.g.: --input=\"2x2xi32=1 2 3 4\"\n"
    "Optionally, brackets may be used to separate the element values:\n"
    "  2x2xi32=[[1 2][3 4]]\n"
    "Raw binary files can be read to provide buffer contents:\n"
    "  2x2xi32=@some/file.bin\n"
    "\n"
    "Numpy npy files from numpy.save can be read to provide 1+ values:\n"
    "  @some.npy\n"
    "\n"
    "Each occurrence of the flag indicates an input in the order they were\n"
    "specified on the command line.");

// Global state shared across all benchmarks and iterations.
// Immutable and thread-safe (so much as anything contained within is).
typedef struct iree_replay_benchmark_globals_t {
  // Shared VM instance. Unowned; callers must retain for the valid lifetime of
  // the registration.
  iree_vm_instance_t* instance;
  // Entire contents of stdin as read on startup.
  // This allows repeated benchmark calls to consume stdin.
  // Only captured if --capture_stdin is passed.
  iree_const_byte_span_t stdin_contents;
} iree_replay_benchmark_globals_t;

// A benchmark registration for each file to run.
typedef struct iree_replay_benchmark_registration_t {
  iree_benchmark_def_t benchmark_def;  // Must be first.
  // Used for relative file path lookup when referencing files in the trace.
  iree_string_view_t root_path;
  // Trace file path.
  iree_string_view_t file_path;
  // Global state shared across all benchmarks and iterations.
  const iree_replay_benchmark_globals_t* globals;
} iree_replay_benchmark_registration_t;

IREE_TRACE(static const char* IREE_REPLAY_ACTIVE_PLOT_ID = "Timing Active");

static iree_status_t iree_replay_benchmark_call_before(
    void* user_data, iree_trace_replay_t* replay, yaml_document_t* document,
    yaml_node_t* event_node, iree_vm_function_t function,
    iree_vm_list_t* input_list) {
  iree_benchmark_state_t* benchmark_state = (iree_benchmark_state_t*)user_data;
  IREE_TRACE_PLOT_VALUE_I64(IREE_REPLAY_ACTIVE_PLOT_ID, 1);
  iree_benchmark_resume_timing(benchmark_state);
  return iree_ok_status();
}

static iree_status_t iree_replay_benchmark_call_after(
    void* user_data, iree_trace_replay_t* replay, yaml_document_t* document,
    yaml_node_t* event_node, iree_vm_function_t function,
    iree_vm_list_t* output_list) {
  iree_benchmark_state_t* benchmark_state = (iree_benchmark_state_t*)user_data;
  iree_benchmark_pause_timing(benchmark_state);
  IREE_TRACE_PLOT_VALUE_I64(IREE_REPLAY_ACTIVE_PLOT_ID, 0);
  return iree_ok_status();
}

static iree_status_t iree_replay_benchmark_run_documents(
    iree_trace_replay_t* replay, FILE* file,
    iree_benchmark_state_t* benchmark_state) {
  // One-pass parsing through the file.
  yaml_parser_t parser;
  if (!yaml_parser_initialize(&parser)) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "yaml_parser_initialize failed");
  }
  yaml_parser_set_input_file(&parser, file);

  bool have_parsed_inputs = false;
  iree_status_t status = iree_ok_status();
  for (bool document_eof = false; !document_eof;) {
    // Parse the subdocument event.
    IREE_TRACE_ZONE_BEGIN_NAMED(z_load, "yaml_parser_load");
    yaml_document_t document;
    bool did_load = yaml_parser_load(&parser, &document);
    IREE_TRACE_ZONE_END(z_load);
    if (!did_load) {
      status = iree_status_from_yaml_parser_error(&parser);
      break;
    }

    // Execute the event or handle EOF (empty document).
    yaml_node_t* event_node = yaml_document_get_root_node(&document);
    if (event_node) {
      status = iree_trace_replay_event(replay, &document, event_node);
    } else {
      document_eof = true;
    }

    // Reclaim subdocument resources before moving on to the next.
    IREE_TRACE_ZONE_BEGIN_NAMED(z_delete, "yaml_document_delete");
    yaml_document_delete(&document);
    IREE_TRACE_ZONE_END(z_delete);

    // If the event created a device and we haven't yet performed our input
    // loading we can do that now before processing subsequent events.
    if (iree_status_is_ok(status) && !have_parsed_inputs && replay->device) {
      status = iree_tooling_parse_into_variant_list(
          iree_hal_device_allocator(replay->device), FLAG_input_list().values,
          FLAG_input_list().count, replay->host_allocator, replay->inputs);
      have_parsed_inputs = true;
    }

    if (!iree_status_is_ok(status)) break;
  }

  yaml_parser_delete(&parser);

  return status;
}

// Benchmark function that runs a trace file.
static iree_status_t iree_replay_benchmark_run_file(
    const iree_benchmark_def_t* benchmark_def,
    iree_benchmark_state_t* benchmark_state) {
  const iree_replay_benchmark_registration_t* registration =
      (const iree_replay_benchmark_registration_t*)benchmark_def->user_data;
  const iree_replay_benchmark_globals_t* globals = registration->globals;

  iree_trace_replay_flags_t replay_flags = IREE_TRACE_REPLAY_FLAG_NONE;
  if (FLAG_print_statistics) {
    replay_flags |= IREE_TRACE_REPLAY_FLAG_PRINT_STATISTICS;
  }
  if (FLAG_reuse_devices) {
    replay_flags |= IREE_TRACE_REPLAY_FLAG_REUSE_DEVICES;
  }
  if (FLAG_reuse_modules) {
    replay_flags |= IREE_TRACE_REPLAY_FLAG_REUSE_MODULES;
  }

  // Setup replay state used for this benchmark.
  iree_trace_replay_t replay;
  IREE_RETURN_IF_ERROR(iree_trace_replay_initialize(
      registration->root_path, globals->instance, replay_flags,
      IREE_VM_CONTEXT_FLAG_NONE, iree_hal_available_driver_registry(),
      iree_allocator_system(), &replay));
  replay.stdin_contents = globals->stdin_contents;

  // Hook into all calls processed during the trace so we can time them.
  replay.call_hooks.user_data = benchmark_state;
  replay.call_hooks.before = iree_replay_benchmark_call_before;
  replay.call_hooks.after = iree_replay_benchmark_call_after;

  // Query device overrides, if any. When omitted the devices from the trace
  // file will be used.
  // TODO(#5724): remove this and instead provide a device set on initialize.
  iree_host_size_t device_uri_count = 0;
  const iree_string_view_t* device_uris = NULL;
  iree_hal_get_devices_flag_list(&device_uri_count, &device_uris);
  iree_trace_replay_set_hal_devices_override(&replay, device_uri_count,
                                             device_uris);

  // Open trace YAML file from the given file_path.
  FILE* file = fopen(registration->file_path.data, "rb");
  if (!file) {
    return iree_make_status(
        iree_status_code_from_errno(errno), "failed to open trace file '%.*s'",
        (int)registration->file_path.size, registration->file_path.data);
  }

  // Call the functions within the trace in order.
  while (iree_benchmark_keep_running(benchmark_state,
                                     /*batch_count=*/1)) {
    // Pause timing that was started automatically. We'll resume/pause around
    // each call.
    // TODO(benvanik): see if we can tell benchmark to start paused?
    iree_benchmark_pause_timing(benchmark_state);

    // Clear replay state.
    iree_trace_replay_reset(&replay);

    // Run all events in the document from start to end.
    IREE_RETURN_IF_ERROR(
        iree_replay_benchmark_run_documents(&replay, file, benchmark_state));

    // Reset file back to the start.
    fseek(file, 0, SEEK_SET);

    // Resume before looping because keep_running requires it.
    iree_benchmark_resume_timing(benchmark_state);
  }

  iree_trace_replay_deinitialize(&replay);
  return iree_ok_status();
}

// Registers benchmarks for each trace file.
static void iree_replay_benchmark_register_trace_files(
    int file_count, char** file_paths,
    const iree_replay_benchmark_globals_t* globals) {
  static iree_replay_benchmark_registration_t* registrations = NULL;
  if (!registrations) free(registrations);
  registrations = (iree_replay_benchmark_registration_t*)malloc(
      file_count * sizeof(iree_replay_benchmark_registration_t));

  for (int i = 0; i < file_count; ++i) {
    iree_string_view_t file_path = iree_make_cstring_view(file_paths[i]);
    registrations[i].root_path = iree_file_path_dirname(file_path);
    registrations[i].file_path = file_path;
    registrations[i].globals = globals;
    registrations[i].benchmark_def = (iree_benchmark_def_t){
        .flags = IREE_BENCHMARK_FLAG_MEASURE_PROCESS_CPU_TIME |
                 IREE_BENCHMARK_FLAG_USE_REAL_TIME,
        .time_unit = IREE_BENCHMARK_UNIT_MILLISECOND,
        .minimum_duration_ns = 0,
        .iteration_count = 0,
        .run = iree_replay_benchmark_run_file,
        .user_data = &registrations[i],
    };
    iree_benchmark_register(iree_file_path_stem(file_path),
                            &registrations[i].benchmark_def);
  }
}

int main(int argc, char** argv) {
  iree_allocator_t host_allocator = iree_allocator_system();

  // Pass through flags to benchmark (allowing --help to fall through).
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_UNDEFINED_OK |
                               IREE_FLAGS_PARSE_MODE_CONTINUE_AFTER_HELP,
                           &argc, &argv);
  iree_benchmark_initialize(&argc, argv);
  if (argc <= 1) {
    fprintf(stderr,
            "no trace files provided; pass one or more yaml file paths");
    return 1;
  }

  // Used when tracing to display benchmark state.
  IREE_TRACE_SET_PLOT_TYPE(IREE_REPLAY_ACTIVE_PLOT_ID,
                           IREE_TRACING_PLOT_TYPE_NUMBER,
                           /*step=*/true, /*fill=*/true, /*color=*/0);

  // Setup shared instance used for each benchmark.
  iree_vm_instance_t* instance = NULL;
  IREE_CHECK_OK(iree_vm_instance_create(IREE_VM_TYPE_CAPACITY_DEFAULT,
                                        host_allocator, &instance));

  // Parse all of stdin right away. The traces we run may source things from it
  // and because we're running in a benchmark loop it'll quickly drain. To
  // ensure every iteration of every benchmark gets the same input we retain it.
  iree_file_contents_t* stdin_contents = NULL;
  if (FLAG_capture_stdin) {
    fprintf(stderr, "Capturing stdin contents at startup...\n");
    IREE_CHECK_OK(iree_stdin_read_contents(host_allocator, &stdin_contents));
    fprintf(stderr, "Captured %" PRIhsz " bytes of stdin content\n",
            stdin_contents->const_buffer.data_length);
  }

  // Register a benchmark per file provided and run them.
  iree_replay_benchmark_globals_t globals = {
      .instance = instance,
      .stdin_contents = stdin_contents ? stdin_contents->const_buffer
                                       : iree_const_byte_span_empty(),
  };
  iree_replay_benchmark_register_trace_files(argc - 1, argv + 1, &globals);
  iree_benchmark_run_specified();

  iree_file_contents_free(stdin_contents);
  iree_vm_instance_release(instance);
  return 0;
}

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
#include "iree/base/internal/flags.h"
#include "iree/base/internal/path.h"
#include "iree/hal/api.h"
#include "iree/testing/benchmark.h"
#include "iree/tooling/device_util.h"
#include "iree/tooling/trace_replay.h"
#include "iree/tooling/yaml_util.h"
#include "iree/vm/api.h"

IREE_FLAG(bool, print_statistics, false,
          "Prints runtime statistics to stderr on exit.");

IREE_FLAG(int32_t, call_iterations, 1,
          "Number of times to invoke each call in the trace. May break usage "
          "with stateful models.");

// A benchmark registration for each file to run.
typedef struct iree_replay_benchmark_registration_t {
  iree_benchmark_def_t benchmark_def;  // Must be first.
  // Used for relative file path lookup when referencing files in the trace.
  iree_string_view_t root_path;
  // Trace file path.
  iree_string_view_t file_path;
  // Shared VM instance. Unowned; callers must retain for the valid lifetime of
  // the registration.
  iree_vm_instance_t* instance;
} iree_replay_benchmark_registration_t;

// A parsed call event from the trace file.
typedef struct iree_replay_benchmark_call_t {
  iree_vm_function_t function;
  iree_vm_list_t* input_list;
  iree_vm_list_t* output_list;
} iree_replay_benchmark_call_t;

// A growable list of calls.
typedef struct iree_replay_benchmark_call_list_t {
  size_t count;
  size_t capacity;
  iree_replay_benchmark_call_t* items;
} iree_replay_benchmark_call_list_t;

// Initializes |out_list| with an initial allocation.
static void iree_replay_benchmark_call_list_initialize(
    iree_replay_benchmark_call_list_t* out_list) {
  out_list->count = 0;
  out_list->capacity = 32;
  out_list->items = (iree_replay_benchmark_call_t*)malloc(
      out_list->capacity * sizeof(*out_list->items));
}

// Deinitializes |list| and frees all call resources.
static void iree_replay_benchmark_call_list_deinitialize(
    iree_replay_benchmark_call_list_t* list) {
  if (!list->items) return;
  for (size_t i = 0; i < list->count; ++i) {
    iree_vm_list_release(list->items[i].input_list);
    iree_vm_list_release(list->items[i].output_list);
  }
  free(list->items);
  memset(list, 0, sizeof(*list));
}

// Acquires a call from the back of the call list, growing as needed.
// The returned pointer is only valid until the next acquire.
static iree_replay_benchmark_call_t*
iree_replay_benchmark_call_list_acquire_back(
    iree_replay_benchmark_call_list_t* list) {
  if (list->count >= list->capacity) {
    list->capacity *= 2;
    list->items = (iree_replay_benchmark_call_t*)realloc(
        list->items, list->capacity * sizeof(*list->items));
  }
  return &list->items[list->count++];
}

// Processes a call trace event by preparing the inputs and appending it to the
// provided |call_list|.
static iree_status_t iree_replay_benchmark_prepare_call(
    iree_trace_replay_t* replay, yaml_document_t* document,
    yaml_node_t* event_node, iree_replay_benchmark_call_list_t* call_list) {
  // Acquire a call in the list.
  iree_replay_benchmark_call_t* call =
      iree_replay_benchmark_call_list_acquire_back(call_list);
  memset(call, 0, sizeof(*call));

  // Prepare the call event state.
  IREE_RETURN_IF_ERROR(iree_trace_replay_event_call_prepare(
      replay, document, event_node, &call->function, &call->input_list));

  // To avoid allocations in the inner benchmark loop we preallocate the outputs
  // here. To avoid a memory leak we'll need to trim it as soon as the call
  // returns as otherwise the list is retained for the lifetime of the
  // registration.
  IREE_RETURN_IF_ERROR(
      iree_vm_list_create(/*element_type=*/NULL, /*initial_capacity=*/8,
                          replay->host_allocator, &call->output_list));

  return iree_ok_status();
}

// Processes a trace event by either setting up the |replay| or appending a call
// to the |call_list|.
static iree_status_t iree_replay_benchmark_process_event(
    iree_trace_replay_t* replay, yaml_document_t* document,
    yaml_node_t* event_node, iree_replay_benchmark_call_list_t* call_list) {
  if (event_node->type != YAML_MAPPING_NODE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "(%zu): expected mapping node",
                            event_node->start_mark.line);
  }
  yaml_node_t* type_node = NULL;
  IREE_RETURN_IF_ERROR(iree_yaml_mapping_find(
      document, event_node, iree_make_cstring_view("type"), &type_node));
  if (iree_yaml_string_equal(type_node,
                             iree_make_cstring_view("context_load"))) {
    return iree_trace_replay_event_context_load(replay, document, event_node);
  } else if (iree_yaml_string_equal(type_node,
                                    iree_make_cstring_view("module_load"))) {
    return iree_trace_replay_event_module_load(replay, document, event_node);
  } else if (iree_yaml_string_equal(type_node,
                                    iree_make_cstring_view("call"))) {
    return iree_replay_benchmark_prepare_call(replay, document, event_node,
                                              call_list);
  }
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED, "(%zu): unhandled type '%.*s'",
      event_node->start_mark.line, (int)type_node->data.scalar.length,
      type_node->data.scalar.value);
}

// Loads the trace at |file_path| and configures |replay| for making calls in
// the |call_list|.
static iree_status_t iree_replay_benchmark_load_trace(
    iree_string_view_t file_path, iree_trace_replay_t* replay,
    iree_replay_benchmark_call_list_t* call_list) {
  // Open trace YAML file from the given file_path.
  FILE* file = fopen(file_path.data, "rb");
  if (!file) {
    return iree_make_status(iree_status_code_from_errno(errno),
                            "failed to open trace file '%.*s'",
                            (int)file_path.size, file_path.data);
  }

  // One-pass parsing through the file.
  yaml_parser_t parser;
  if (!yaml_parser_initialize(&parser)) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "yaml_parser_initialize failed");
  }
  yaml_parser_set_input_file(&parser, file);

  // Read each event document in the file until EOF.
  iree_status_t status = iree_ok_status();
  for (bool document_eof = false; !document_eof;) {
    yaml_document_t document;
    if (!yaml_parser_load(&parser, &document)) {
      status = iree_status_from_yaml_parser_error(&parser);
      break;
    }
    yaml_node_t* event_node = yaml_document_get_root_node(&document);
    if (event_node) {
      status = iree_replay_benchmark_process_event(replay, &document,
                                                   event_node, call_list);
    } else {
      document_eof = true;
    }
    yaml_document_delete(&document);
    if (!iree_status_is_ok(status)) break;
  }

  yaml_parser_delete(&parser);
  fclose(file);
  return status;
}

// Benchmark function that runs a trace file.
static iree_status_t iree_replay_benchmark_run_file(
    const iree_benchmark_def_t* benchmark_def,
    iree_benchmark_state_t* benchmark_state) {
  const iree_replay_benchmark_registration_t* registration =
      (const iree_replay_benchmark_registration_t*)benchmark_def->user_data;

  // Setup replay state used for this benchmark.
  iree_trace_replay_t replay;
  IREE_RETURN_IF_ERROR(iree_trace_replay_initialize(
      registration->root_path, registration->instance,
      IREE_VM_CONTEXT_FLAG_NONE, iree_hal_available_driver_registry(),
      iree_allocator_system(), &replay));

  // Query device overrides, if any. When omitted the devices from the trace
  // file will be used.
  // TODO(#5724): remove this and instead provide a device set on initialize.
  iree_host_size_t device_uri_count = 0;
  const iree_string_view_t* device_uris = NULL;
  iree_hal_get_devices_flag_list(&device_uri_count, &device_uris);
  iree_trace_replay_set_hal_devices_override(&replay, device_uri_count,
                                             device_uris);

  // Load YAML file and setup replay state with all modules loaded and ready.
  iree_replay_benchmark_call_list_t call_list;
  iree_replay_benchmark_call_list_initialize(&call_list);
  IREE_RETURN_IF_ERROR(iree_replay_benchmark_load_trace(registration->file_path,
                                                        &replay, &call_list));

  // Call the functions within the trace in order.
  while (iree_benchmark_keep_running(benchmark_state,
                                     /*batch_count=*/FLAG_call_iterations)) {
    for (size_t i = 0; i < call_list.count; ++i) {
      iree_replay_benchmark_call_t* call = &call_list.items[i];
      for (int32_t j = 0; j < FLAG_call_iterations; ++j) {
        IREE_RETURN_IF_ERROR(iree_vm_invoke(
            replay.context, call->function, IREE_VM_INVOCATION_FLAG_NONE,
            /*policy=*/NULL, call->input_list, call->output_list,
            replay.host_allocator));
        IREE_RETURN_IF_ERROR(iree_vm_list_resize(call->output_list, 0));
      }
    }
  }

  iree_replay_benchmark_call_list_deinitialize(&call_list);
  iree_trace_replay_deinitialize(
      &replay, FLAG_print_statistics
                   ? IREE_TRACE_REPLAY_SHUTDOWN_PRINT_STATISTICS
                   : IREE_TRACE_REPLAY_SHUTDOWN_QUIET);
  return iree_ok_status();
}

// Registers benchmarks for each trace file.
static void iree_replay_benchmark_register_trace_files(
    int file_count, char** file_paths, iree_vm_instance_t* instance) {
  static iree_replay_benchmark_registration_t* registrations = NULL;
  if (!registrations) free(registrations);
  registrations = (iree_replay_benchmark_registration_t*)malloc(
      file_count * sizeof(iree_replay_benchmark_registration_t));

  for (int i = 0; i < file_count; ++i) {
    iree_string_view_t file_path = iree_make_cstring_view(file_paths[i]);
    registrations[i].root_path = iree_file_path_dirname(file_path);
    registrations[i].file_path = file_path;
    registrations[i].instance = instance;
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

  // Setup shared instance used for each benchmark.
  iree_vm_instance_t* instance = NULL;
  IREE_CHECK_OK(iree_vm_instance_create(iree_allocator_system(), &instance));

  // Register a benchmark per file provided and run them.
  iree_replay_benchmark_register_trace_files(argc - 1, argv + 1, instance);
  iree_benchmark_run_specified();

  iree_vm_instance_release(instance);
  return 0;
}

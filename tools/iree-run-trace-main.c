// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/internal/flags.h"
#include "iree/base/internal/path.h"
#include "iree/hal/api.h"
#include "iree/tooling/device_util.h"
#include "iree/tooling/trace_replay.h"
#include "iree/tooling/yaml_util.h"
#include "iree/vm/api.h"

IREE_FLAG(bool, trace_execution, false, "Traces VM execution to stderr.");

IREE_FLAG(bool, print_statistics, false,
          "Prints runtime statistics to stderr on exit.");

// Runs the trace in |file| using |root_path| as the base for any path lookups
// required for external files referenced in |file|.
static iree_status_t iree_run_trace_file(iree_string_view_t root_path,
                                         FILE* file,
                                         iree_vm_instance_t* instance) {
  iree_trace_replay_t replay;
  IREE_RETURN_IF_ERROR(iree_trace_replay_initialize(
      root_path, instance,
      FLAG_trace_execution ? IREE_VM_CONTEXT_FLAG_TRACE_EXECUTION
                           : IREE_VM_CONTEXT_FLAG_NONE,
      iree_hal_available_driver_registry(), iree_allocator_system(), &replay));

  // Query device overrides, if any. When omitted the devices from the trace
  // file will be used.
  // TODO(#5724): remove this and instead provide a device set on initialize.
  iree_host_size_t device_uri_count = 0;
  const iree_string_view_t* device_uris = NULL;
  iree_hal_get_devices_flag_list(&device_uri_count, &device_uris);
  iree_trace_replay_set_hal_devices_override(&replay, device_uri_count,
                                             device_uris);

  yaml_parser_t parser;
  if (!yaml_parser_initialize(&parser)) {
    iree_trace_replay_deinitialize(&replay, IREE_TRACE_REPLAY_SHUTDOWN_QUIET);
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "yaml_parser_initialize failed");
  }
  yaml_parser_set_input_file(&parser, file);

  iree_status_t status = iree_ok_status();
  for (bool document_eof = false; !document_eof;) {
    yaml_document_t document;
    if (!yaml_parser_load(&parser, &document)) {
      status = iree_status_from_yaml_parser_error(&parser);
      break;
    }
    yaml_node_t* event_node = yaml_document_get_root_node(&document);
    if (event_node) {
      status = iree_trace_replay_event(&replay, &document, event_node);
    } else {
      document_eof = true;
    }
    yaml_document_delete(&document);
    if (!iree_status_is_ok(status)) break;
  }

  yaml_parser_delete(&parser);
  iree_trace_replay_deinitialize(
      &replay, FLAG_print_statistics
                   ? IREE_TRACE_REPLAY_SHUTDOWN_PRINT_STATISTICS
                   : IREE_TRACE_REPLAY_SHUTDOWN_QUIET);
  return status;
}

// Runs each of the given traces files sequentially in isolated contexts.
static iree_status_t iree_run_trace_files(int file_count, char** file_paths,
                                          iree_vm_instance_t* instance) {
  for (int i = 0; i < file_count; ++i) {
    iree_string_view_t file_path = iree_make_cstring_view(file_paths[i]);
    iree_string_view_t root_path = iree_file_path_dirname(file_path);
    FILE* file = fopen(file_paths[i], "rb");
    if (!file) {
      return iree_make_status(iree_status_code_from_errno(errno),
                              "failed to open trace file '%.*s'",
                              (int)file_path.size, file_path.data);
    }
    iree_status_t status = iree_run_trace_file(root_path, file, instance);
    fclose(file);
    IREE_RETURN_IF_ERROR(status, "replaying trace file '%.*s'",
                         (int)file_path.size, file_path.data);
  }
  return iree_ok_status();
}

int main(int argc, char** argv) {
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);
  if (argc <= 1) {
    fprintf(stderr,
            "no trace files provided; pass one or more yaml file paths");
    return 1;
  }

  iree_vm_instance_t* instance = NULL;
  iree_status_t status =
      iree_vm_instance_create(iree_allocator_system(), &instance);
  if (iree_status_is_ok(status)) {
    status = iree_run_trace_files(argc - 1, argv + 1, instance);
  }
  iree_vm_instance_release(instance);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    return 1;
  }
  return 0;
}

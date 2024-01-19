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
#include "iree/tooling/vm_util.h"
#include "iree/tooling/yaml_util.h"
#include "iree/vm/api.h"

IREE_FLAG(bool, trace_execution, false, "Traces VM execution to stderr.");

IREE_FLAG(bool, print_statistics, false,
          "Prints runtime statistics to stderr on exit.");

IREE_FLAG(bool, print_calls, false, "Prints all I/O for each call to stdout.");
IREE_FLAG(bool, print_call_inputs, false,
          "Prints all inputs for each call before they are made to stdout.");
IREE_FLAG(bool, print_call_outputs, false,
          "Prints all outputs for each call after they are made to stdout.");

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

IREE_FLAG_LIST(
    string, output,
    "Specifies how to handle an output from the invocation:\n"
    "  `` (empty): ignore output\n"
    "     e.g.: --output=\n"
    "  `-`: print textual form to stdout\n"
    "     e.g.: --output=-\n"
    "  `@file.npy`: create/overwrite a numpy npy file and write buffer view\n"
    "     e.g.: --output=@file.npy\n"
    "  `+file.npy`: create/append a numpy npy file and write buffer view\n"
    "     e.g.: --output=+file.npy\n"
    "\n"
    "Numpy npy files can be read in Python using numpy.load, for example an\n"
    "invocation producing two outputs can be concatenated as:\n"
    "    --output=@file.npy --output=+file.npy\n"
    "And then loaded in Python by reading from the same file:\n"
    "  with open('file.npy', 'rb') as f:\n"
    "    print(numpy.load(f))\n"
    "    print(numpy.load(f))\n"
    "\n"
    "Each occurrence of the flag indicates an output in the order they were\n"
    "specified on the command line.");

IREE_FLAG_LIST(string, expected_output,
               "An expected function output following the same format as "
               "--input. When present the results of the "
               "invocation will be compared against these values and the "
               "tool will return non-zero if any differ. If the value of a "
               "particular output is not of interest provide `(ignored)`.");

IREE_FLAG(int32_t, output_max_element_count, 1024,
          "Prints up to the maximum number of elements of output tensors, "
          "eliding the remainder.");

static iree_status_t iree_trace_replay_call_before(void* user_data,
                                                   iree_trace_replay_t* replay,
                                                   yaml_document_t* document,
                                                   yaml_node_t* event_node,
                                                   iree_vm_function_t function,
                                                   iree_vm_list_t* input_list) {
  if (FLAG_print_calls || FLAG_print_call_inputs) {
    iree_string_view_t function_name = iree_vm_function_name(&function);
    fprintf(stdout, "--- CALL[%.*s] ---\n", (int)function_name.size,
            function_name.data);
    IREE_RETURN_IF_ERROR(iree_tooling_variant_list_fprint(
        IREE_SV("arg"), input_list,
        (iree_host_size_t)FLAG_output_max_element_count, stdout));
  }
  return iree_ok_status();
}

static iree_status_t iree_trace_replay_call_after(void* user_data,
                                                  iree_trace_replay_t* replay,
                                                  yaml_document_t* document,
                                                  yaml_node_t* event_node,
                                                  iree_vm_function_t function,
                                                  iree_vm_list_t* output_list) {
  if (FLAG_print_calls || FLAG_print_call_outputs) {
    if (!FLAG_print_calls && !FLAG_print_call_inputs) {
      iree_string_view_t function_name = iree_vm_function_name(&function);
      fprintf(stdout, "--- CALL[%.*s] ---\n", (int)function_name.size,
              function_name.data);
    }
    IREE_RETURN_IF_ERROR(iree_tooling_variant_list_fprint(
        IREE_SV("result"), output_list,
        (iree_host_size_t)FLAG_output_max_element_count, stdout));
  }
  return iree_ok_status();
}

// Runs the trace in |file| using |root_path| as the base for any path lookups
// required for external files referenced in |file|.
static iree_status_t iree_run_trace_file(iree_string_view_t root_path,
                                         FILE* file,
                                         iree_vm_instance_t* instance) {
  iree_trace_replay_flags_t replay_flags = IREE_TRACE_REPLAY_FLAG_NONE;
  if (FLAG_print_statistics) {
    replay_flags |= IREE_TRACE_REPLAY_FLAG_PRINT_STATISTICS;
  }

  iree_vm_context_flags_t context_flags = IREE_VM_CONTEXT_FLAG_NONE;
  if (FLAG_trace_execution) {
    context_flags |= IREE_VM_CONTEXT_FLAG_TRACE_EXECUTION;
  }

  iree_trace_replay_t replay;
  IREE_RETURN_IF_ERROR(iree_trace_replay_initialize(
      root_path, instance, replay_flags, context_flags,
      iree_hal_available_driver_registry(), iree_allocator_system(), &replay));

  // Hook into all calls processed during the trace.
  replay.call_hooks.user_data = NULL;
  replay.call_hooks.before = iree_trace_replay_call_before;
  replay.call_hooks.after = iree_trace_replay_call_after;

  // Query device overrides, if any. When omitted the devices from the trace
  // file will be used.
  iree_trace_replay_set_hal_devices_override(&replay,
                                             iree_hal_device_flag_list());

  yaml_parser_t parser;
  if (!yaml_parser_initialize(&parser)) {
    iree_trace_replay_deinitialize(&replay);
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "yaml_parser_initialize failed");
  }
  yaml_parser_set_input_file(&parser, file);

  bool have_parsed_inputs = false;
  iree_status_t status = iree_ok_status();
  for (bool document_eof = false; !document_eof;) {
    // Parse the subdocument event.
    yaml_document_t document;
    if (!yaml_parser_load(&parser, &document)) {
      status = iree_status_from_yaml_parser_error(&parser);
      break;
    }

    // Execute the event or handle EOF (empty document).
    yaml_node_t* event_node = yaml_document_get_root_node(&document);
    if (event_node) {
      status = iree_trace_replay_event(&replay, &document, event_node);
    } else {
      document_eof = true;
    }

    // Reclaim subdocument resources before moving on to the next.
    yaml_document_delete(&document);
    if (!iree_status_is_ok(status)) break;

    // If the event created a device and we haven't yet performed our input
    // loading we can do that now before processing subsequent events.
    if (!have_parsed_inputs && replay.device) {
      status = iree_tooling_parse_into_variant_list(
          replay.device, iree_hal_device_allocator(replay.device),
          FLAG_input_list().values, FLAG_input_list().count,
          replay.host_allocator, replay.inputs);
      have_parsed_inputs = true;
    }
    if (!iree_status_is_ok(status)) break;
  }

  yaml_parser_delete(&parser);

  // Transfer outputs to the host so they can be processed.
  if (iree_status_is_ok(status) && replay.device != NULL) {
    iree_hal_buffer_params_t target_params = {
        .usage = IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING,
        .access = IREE_HAL_MEMORY_ACCESS_ALL,
        .type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
                IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
        .queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY,
        .min_alignment = 0,
    };
    status = iree_tooling_transfer_variant_list(
        replay.device, replay.outputs, iree_hal_device_allocator(replay.device),
        target_params, /*wait_fence=*/NULL, /*signal_fence=*/NULL);
  }

  // Optionally process outputs from the replay session.
  if (iree_status_is_ok(status)) {
    if (FLAG_output_list().count == 0) {
      IREE_RETURN_IF_ERROR(
          iree_tooling_variant_list_fprint(
              IREE_SV("output"), replay.outputs,
              (iree_host_size_t)FLAG_output_max_element_count, stdout),
          "printing results");
    } else {
      IREE_RETURN_IF_ERROR(
          iree_tooling_output_variant_list(
              replay.outputs, FLAG_output_list().values,
              FLAG_output_list().count,
              (iree_host_size_t)FLAG_output_max_element_count, stdout),
          "outputting results");
    }
  }

  iree_trace_replay_deinitialize(&replay);
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
  IREE_TRACE_APP_ENTER();

  iree_flags_set_usage(
      "iree-run-trace",
      "Executes a YAML trace file containing a sequence of context operations\n"
      "and calls represented as subdocuments.\n"
      "\n"
      "Example loading a bytecode module and calling a function:\n"
      "\n"
      "```yaml\n"
      "type: context_load\n"
      "---\n"
      "type: module_load\n"
      "module:\n"
      "  type: buildin\n"
      "  name: hal\n"
      "---\n"
      "type: module_load\n"
      "module:\n"
      "  type: bytecode\n"
      "  path: ../build/some_module.vmfb\n"
      "  mmap: true\n"
      "---\n"
      "type: call\n"
      "function: module.mul\n"
      "args:\n"
      "- !input.take 0\n"
      "- !input.take 1\n"
      "results:\n"
      "- !output.push\n"
      "- !output.push\n"
      "```\n"
      "\n"
      "This can be invoked like iree-run-module specifying inputs/outputs:\n"
      "  iree-run-trace trace.yml    \\\n"
      "      --device=local-sync     \\\n"
      "      --input=4xf32=0,1,2,3,4 \\\n"
      "      --input=@input1.npy     \\\n"
      "      --output=@outputs.npy   \\\n"
      "      --output=+outputs.npy\n"
      "\n"
      "In addition to `--input=`/`--output=` flag access a user-defined\n"
      "blackboard exists for preserving temporary values used within the\n"
      "trace. Blackboard slots are defined by ordinal and they can be used\n"
      "in any context and input/output can be, `!blackboard.get` instead of\n"
      "`!input.get` and `!blackboard.set` instead of `!output.set`.\n"
      "\n"
      "--- Events ---\n"
      "\n"
      "`type: context_load`\n"
      "Loads an empty VM context with no modules registered.\n"
      "\n"
      "`type: module_load`\n"
      "Loads a module into the current context. Modules may either be\n"
      "`builtin` (compiled into the binary) or dynamically-loaded `bytecode`.\n"
      "\n"
      "`type: blackboard_clear`\n"
      "Clears the contents of the blackboard and resets it to 0 elements.\n"
      "\n"
      "`type: assign`\n"
      "Assigns sources from a `from` sequence to targets in a `to` sequence.\n"
      "Equivalent to an identity function call and can be used to move\n"
      "between inputs, outputs, and the blackboard.\n"
      "\n"
      "`type: numpy_load`\n"
      "Loads one or more ndarrays from a .npy value. Each array has a target\n"
      "where the array will be retained such as `!blackboard.set 2`.\n"
      "\n"
      "`type: numpy_save\n"
      "Saves one or more ndarrays to a .npy value. Each array has a source\n"
      "where the array will be taken from such as `!blackboard.get 2`.\n"
      "\n"
      "`type: call`\n"
      "Invokes a function in the context by fully-qualified `function` name.\n"
      "Uses arguments from an `args` sequence and produces results into a\n"
      "`results` sequence.\n"
      "\n"
      "--- Sources ---\n"
      "\n"
      "`type: null`\n"
      "A null ref value.\n"
      "\n"
      "`!hal.buffer_view 4xf32=0,1,2,3`\n"
      "A constant iree_hal_buffer_view_t/!hal.buffer_view value using the\n"
      "same formatting as iree-run-module's `--input=` flag.\n"
      "\n"
      "`!hal.buffer 4xf32=0,1,2,3`\n"
      "An initialized iree_hal_buffer_t/!hal.buffer without the wrapping view\n"
      "metadata.\n"
      "\n"
      "`!input.get ORDINAL` / `!input.take ORDINAL`\n"
      "Returns a reference to `--input=` flag at ORDINAL. Note that a single\n"
      "npy file may expand to multiple inputs. The `take` variant transfers\n"
      "ownership and clears the slot in the list and is recommended to avoid\n"
      "keeping unneeded inputs around for the duration of the trace.\n"
      "\n"
      "`!output.get ORDINAL` / `!output.take ORDINAL`\n"
      "Returns a reference to the `--output=` flag at ORDINAL. These are\n"
      "initially empty until assigned by the trace.\n"
      "\n"
      "`!blackboard.get ORDINAL` / `!blackboard.take ORDINAL`\n"
      "Returns a reference to the blackboard slot ORDINAL. The blackboard is\n"
      "initially empty and slots must be assigned in order to define them.\n"
      "The `take` variant transfers ownership and clears the slot in the\n"
      "blackboard and is recommended to avoid keeping large resources live\n"
      "in the blackboard longer than they need to be.\n"
      "\n"
      "--- Targets ---\n"
      "\n"
      "`!output.set ORDINAL` / `!output.push`\n"
      "Sets the `--output=` flag result value at ORDINAL or pushes it to the\n"
      "back of the output list. Outputs can either be dumped to files or by\n"
      "default printed to stdout.\n"
      "\n"
      "`!blackboard.set ORDINAL` / `blackboard.push`\n"
      "Sets the value of the blackboard slot ORDINAL or pushes it to the back\n"
      "of the blackboard list. Blackboard values will be retained until they\n"
      "are consumed via `!blackboard.take` or the blackboard is cleared.\n"
      "\n");
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);
  if (argc <= 1) {
    fprintf(stderr,
            "no trace files provided; pass one or more yaml file paths");
    IREE_TRACE_APP_EXIT(EXIT_FAILURE);
    return EXIT_FAILURE;
  }

  iree_vm_instance_t* instance = NULL;
  iree_status_t status = iree_vm_instance_create(
      IREE_VM_TYPE_CAPACITY_DEFAULT, iree_allocator_system(), &instance);
  if (iree_status_is_ok(status)) {
    status = iree_run_trace_files(argc - 1, argv + 1, instance);
  }
  iree_vm_instance_release(instance);
  int exit_code = EXIT_SUCCESS;
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    exit_code = EXIT_FAILURE;
  }
  IREE_TRACE_APP_EXIT(exit_code);
  return exit_code;
}

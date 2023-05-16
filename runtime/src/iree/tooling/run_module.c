// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/run_module.h"

#include "iree/base/api.h"
#include "iree/base/internal/flags.h"
#include "iree/base/tracing.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/types.h"
#include "iree/tooling/comparison.h"
#include "iree/tooling/context_util.h"
#include "iree/tooling/device_util.h"
#include "iree/tooling/instrument_util.h"
#include "iree/tooling/vm_util.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode/module.h"

IREE_FLAG(string, function, "",
          "Name of a function contained in the module specified by --module= "
          "to run.");

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

IREE_FLAG_LIST(
    string, expected_output,
    "An expected function output following the same format as `--input=`.\n"
    "When present the results of the invocation will be compared against\n"
    "these values and the tool will return non-zero if any differ. If the\n"
    "value of a particular output is not of interest provide `(ignored)`.");

IREE_FLAG(
    int32_t, output_max_element_count, 1024,
    "Prints up to the maximum number of elements of output tensors and elides\n"
    "the remainder.");

IREE_FLAG(bool, print_statistics, false,
          "Prints runtime statistics to stderr on exit.");

static iree_status_t iree_tooling_process_outputs(
    iree_vm_list_t* outputs, iree_allocator_t host_allocator,
    int* out_exit_code);

static iree_status_t iree_tooling_create_run_context(
    iree_vm_instance_t* instance, iree_string_view_t default_device_uri,
    iree_const_byte_span_t module_contents, iree_allocator_t host_allocator,
    iree_vm_context_t** out_context, iree_vm_function_t* out_function,
    iree_hal_device_t** out_device,
    iree_hal_allocator_t** out_device_allocator) {
  // Load all modules specified by --module= flags.
  iree_tooling_module_list_t module_list;
  iree_tooling_module_list_initialize(&module_list);
  IREE_RETURN_IF_ERROR(iree_tooling_load_modules_from_flags(
                           instance, host_allocator, &module_list),
                       "loading modules and dependencies");

  // Load the optional bytecode module from the provided flatbuffer data.
  // Note that we do this after all other --module= flags are processed so that
  // we ensure any dependent types are registered with the instance.
  iree_status_t status = iree_ok_status();
  if (!iree_const_byte_span_is_empty(module_contents)) {
    iree_vm_module_t* module = NULL;
    status = iree_status_annotate_f(
        iree_vm_bytecode_module_create(instance, module_contents,
                                       iree_allocator_null(), host_allocator,
                                       &module),
        "loading custom bytecode module from memory");
    if (iree_status_is_ok(status)) {
      status = iree_tooling_module_list_push_back(&module_list, module);
    }
  }
  if (!iree_status_is_ok(status)) {
    iree_tooling_module_list_reset(&module_list);
    return status;
  }

  // There's concept of a "main module" in the VM but here we use it to allow
  // for a shorthand --function= flag that doesn't need the module name.
  iree_vm_module_t* main_module = iree_tooling_module_list_back(&module_list);
  if (!main_module) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "no user module specified; use --module=file.vmfb to load from a "
        "file or --module=- to load from stdin");
  }

  // Create the VM context with all of the modules. Dependent modules will be
  // loaded (like the HAL) and special things like the HAL device and allocator
  // are returned for convenience. Note that not all programs need the HAL.
  iree_vm_context_t* context = NULL;
  iree_hal_device_t* device = NULL;
  iree_hal_allocator_t* device_allocator = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_status_annotate_f(
        iree_tooling_create_context_from_flags(
            instance, module_list.count, module_list.values, default_device_uri,
            host_allocator, &context, &device, &device_allocator),
        "creating VM context");
  }

  iree_tooling_module_list_reset(&module_list);
  if (!iree_status_is_ok(status)) {
    return status;
  }

  // Choose which function to run - either the one specified in the flag or the
  // only exported non-internal function.
  iree_vm_function_t function = {0};
  if (strlen(FLAG_function) == 0) {
    status = iree_tooling_find_single_exported_function(main_module, &function);
  } else {
    status = iree_status_annotate_f(
        iree_vm_module_lookup_function_by_name(
            main_module, IREE_VM_FUNCTION_LINKAGE_EXPORT,
            iree_make_cstring_view(FLAG_function), &function),
        "looking up function '%s'", FLAG_function);
  }

  if (iree_status_is_ok(status)) {
    *out_context = context;
    *out_function = function;
    *out_device = device;
    *out_device_allocator = device_allocator;
  } else {
    iree_vm_context_release(context);
    iree_hal_allocator_release(device_allocator);
    iree_hal_device_release(device);
  }
  return status;
}

static iree_status_t iree_tooling_run_function(
    iree_vm_context_t* context, iree_vm_function_t function,
    iree_hal_device_t* device, iree_hal_allocator_t* device_allocator,
    iree_allocator_t host_allocator, int* out_exit_code) {
  iree_string_view_t function_name = iree_vm_function_name(&function);
  (void)function_name;

  // Parse --input= values into device buffers.
  iree_vm_list_t* inputs = NULL;
  iree_status_t status = iree_status_annotate_f(
      iree_tooling_parse_to_variant_list(
          device_allocator, FLAG_input_list().values, FLAG_input_list().count,
          host_allocator, &inputs),
      "parsing function inputs");

  // If the function is async add fences so we can invoke it synchronously.
  iree_hal_fence_t* finish_fence = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_status_annotate_f(
        iree_tooling_append_async_fence_inputs(
            inputs, &function, device, /*wait_fence=*/NULL, &finish_fence),
        "setting up async-external fence inputs");
  }

  // Empty output list to be populated by the invocation.
  iree_vm_list_t* outputs = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_vm_list_create(iree_vm_make_undefined_type_def(), 16,
                                 host_allocator, &outputs);
  }

  // TODO(benvanik): move behind a --verbose flag and add more logging.
  if (iree_status_is_ok(status)) {
    fprintf(stdout, "EXEC @%.*s\n", (int)function_name.size,
            function_name.data);
  }

  // Begin profiling immediate prior to invocation.
  if (iree_status_is_ok(status)) {
    status = iree_status_annotate_f(iree_hal_begin_profiling_from_flags(device),
                                    "beginning device profiling");
  }

  // Invoke the function with the provided inputs.
  if (iree_status_is_ok(status)) {
    status = iree_status_annotate_f(
        iree_vm_invoke(context, function, IREE_VM_INVOCATION_FLAG_NONE,
                       /*policy=*/NULL, inputs, outputs, host_allocator),
        "invoking function '%.*s'", (int)function_name.size,
        function_name.data);
  }
  iree_vm_list_release(inputs);

  // If the function is async we need to wait for it to complete.
  if (iree_status_is_ok(status) && finish_fence) {
    IREE_RETURN_IF_ERROR(
        iree_hal_fence_wait(finish_fence, iree_infinite_timeout()),
        "waiting on finish fence");
  }

  // End profiling after waiting for the invocation to finish.
  if (iree_status_is_ok(status)) {
    status = iree_status_annotate_f(iree_hal_end_profiling_from_flags(device),
                                    "ending device profiling");
  }

  // Grab any instrumentation data present in the context and write it to disk.
  if (iree_status_is_ok(status)) {
    status = iree_status_annotate_f(
        iree_tooling_process_instrument_data(context, host_allocator),
        "processing instrument data");
  }

  // Handle either printing/writing the outputs or checking them against
  // expected values (basic pass/fail testing).
  if (iree_status_is_ok(status)) {
    status = iree_status_annotate_f(
        iree_tooling_process_outputs(outputs, host_allocator, out_exit_code),
        "processing function outputs");
  }
  iree_vm_list_release(outputs);

  return status;
}

static iree_status_t iree_tooling_process_outputs(
    iree_vm_list_t* outputs, iree_allocator_t host_allocator,
    int* out_exit_code) {
  *out_exit_code = EXIT_SUCCESS;

  // Basic output handling to route to the console or files.
  if (FLAG_expected_output_list().count == 0) {
    if (FLAG_output_list().count == 0) {
      // Print all outputs.
      return iree_status_annotate_f(
          iree_tooling_variant_list_fprint(
              IREE_SV("result"), outputs,
              (iree_host_size_t)FLAG_output_max_element_count, stdout),
          "printing results");
    } else {
      // Write (or ignore) all outputs.
      return iree_status_annotate_f(
          iree_tooling_output_variant_list(
              outputs, FLAG_output_list().values, FLAG_output_list().count,
              (iree_host_size_t)FLAG_output_max_element_count, stdout),
          "outputting results");
    }
  }

  // Compare against contents in host-local memory. This avoids polluting
  // device memory statistics.
  iree_hal_allocator_t* heap_allocator = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_allocator_create_heap(
      IREE_SV("heap"), host_allocator, host_allocator, &heap_allocator));

  // Parse expected list into host-local memory that we can easily access.
  iree_vm_list_t* expected_list = NULL;
  iree_status_t status = iree_status_annotate_f(
      iree_tooling_parse_to_variant_list(
          heap_allocator, FLAG_expected_output_list().values,
          FLAG_expected_output_list().count, host_allocator, &expected_list),
      "parsing expected function outputs");

  // Compare expected vs actual lists and output diffs.
  if (iree_status_is_ok(status)) {
    bool did_match = iree_tooling_compare_variant_lists(expected_list, outputs,
                                                        host_allocator, stdout);
    if (did_match) {
      fprintf(
          stdout,
          "[SUCCESS] all function outputs matched their expected values.\n");
    }

    // Exit code 0 if all results matched the expected values.
    *out_exit_code = did_match ? EXIT_SUCCESS : EXIT_FAILURE;
  }

  iree_vm_list_release(expected_list);
  iree_hal_allocator_release(heap_allocator);
  return status;
}

iree_status_t iree_tooling_run_module_from_flags(
    iree_vm_instance_t* instance, iree_allocator_t host_allocator,
    int* out_exit_code) {
  return iree_tooling_run_module_with_data(instance, iree_string_view_empty(),
                                           iree_const_byte_span_empty(),
                                           host_allocator, out_exit_code);
}

iree_status_t iree_tooling_run_module_with_data(
    iree_vm_instance_t* instance, iree_string_view_t default_device_uri,
    iree_const_byte_span_t module_contents, iree_allocator_t host_allocator,
    int* out_exit_code) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Setup the VM context with all required modules and get the function to run.
  // This also returns the HAL device and allocator (if any) for I/O handling.
  iree_vm_context_t* context = NULL;
  iree_vm_function_t function = {0};
  iree_hal_device_t* device = NULL;
  iree_hal_allocator_t* device_allocator = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_tooling_create_run_context(instance, default_device_uri,
                                      module_contents, host_allocator, &context,
                                      &function, &device, &device_allocator),
      "creating run context");

  // Parse inputs, run the function, and process outputs.
  iree_status_t status =
      iree_tooling_run_function(context, function, device, device_allocator,
                                host_allocator, out_exit_code);

  // Release the context and all retained resources (variables, constants, etc).
  iree_vm_context_release(context);

  // Print statistics after we've released the inputs/outputs and the context
  // which may be holding on to resources like constants/variables.
  if (device_allocator && FLAG_print_statistics) {
    IREE_IGNORE_ERROR(
        iree_hal_allocator_statistics_fprint(stderr, device_allocator));
  }

  iree_hal_allocator_release(device_allocator);
  iree_hal_device_release(device);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

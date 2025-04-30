// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/run_module.h"

#include "iree/base/api.h"
#include "iree/base/internal/flags.h"
#include "iree/hal/api.h"
#include "iree/io/stdio_stream.h"
#include "iree/modules/hal/types.h"
#include "iree/tooling/comparison.h"
#include "iree/tooling/context_util.h"
#include "iree/tooling/device_util.h"
#include "iree/tooling/function_io.h"
#include "iree/tooling/function_util.h"
#include "iree/tooling/instrument_util.h"
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
    "  e.g.: --input=\"2x2xi32=[[1 2][3 4]]\"\n"
    "Raw binary files can be read to provide buffer contents:\n"
    "  e.g.: --input=2x2xi32=@some/file.bin\n"
    "\n"
    "Numpy npy files from numpy.save can be read to provide 1+ values:\n"
    "  e.g.: --input=@some.npy\n"
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
    "  `@file.npy`: create/overwrite a numpy npy file and write an ndarray\n"
    "     e.g.: --output=@file.npy\n"
    "  `+file.npy`: create/append a numpy npy file and write an ndarray\n"
    "     e.g.: --output=+file.npy\n"
    "  `@file.bin`: create/overwrite a binary file and write value contents\n"
    "     e.g.: --output=@file.bin\n"
    "  `+file.bin`: create/append a binary file and write value contents\n"
    "     e.g.: --output=+file.bin\n"
    "\n"
    "Numpy npy files can be read in Python using numpy.load, for example an\n"
    "invocation producing two outputs can be concatenated as:\n"
    "    --output=@file.npy --output=+file.npy\n"
    "And then loaded in Python by reading from the same file:\n"
    "  with open('file.npy', 'rb') as f:\n"
    "    print(numpy.load(f))\n"
    "    print(numpy.load(f))\n"
    "Primitive values are written as shape=() ndarrays and buffers are\n"
    "written as i8 arrays with the length of the buffer.\n"
    "\n"
    "Binary files contain only the contents of the values/buffers provided\n"
    "without metadata; users must know the shape/type of the output.\n"
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

static iree_status_t iree_tooling_process_results(
    iree_hal_device_t* device, iree_string_view_t results_cconv,
    iree_vm_list_t* results, iree_io_stream_t* stream,
    iree_allocator_t host_allocator, int* out_exit_code);

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
    iree_vm_module_release(module);  // Now tracked in module_list.
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

static iree_status_t iree_tooling_annotate_status_with_function_decl(
    iree_status_t base_status, iree_vm_function_t function) {
  iree_string_view_t decl = iree_vm_function_lookup_attr_by_name(
      &function, IREE_SV("iree.abi.declaration"));
  if (!iree_string_view_is_empty(decl)) {
    return iree_status_annotate_f(base_status, "`%.*s`", (int)decl.size,
                                  decl.data);
  }
  return base_status;
}

static iree_status_t iree_tooling_run_function(
    iree_vm_context_t* context, iree_vm_function_t function,
    iree_hal_device_t* device, iree_hal_allocator_t* device_allocator,
    iree_allocator_t host_allocator, int* out_exit_code) {
  iree_string_view_t function_name = iree_vm_function_name(&function);
  (void)function_name;

  iree_vm_function_signature_t signature =
      iree_vm_function_signature(&function);
  iree_string_view_t arguments_cconv, results_cconv;
  iree_status_t status = iree_vm_function_call_get_cconv_fragments(
      &signature, &arguments_cconv, &results_cconv);

  // Parse --input= values into device buffers.
  iree_vm_list_t* inputs = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_status_annotate_f(
        iree_tooling_parse_variants(arguments_cconv, FLAG_input_list(), device,
                                    device_allocator, host_allocator, &inputs),
        "parsing function inputs");
  }

  // If the function is async add fences so we can invoke it synchronously.
  iree_hal_fence_t* finish_fence = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_status_annotate_f(
        iree_tooling_append_async_fences(inputs, function, device,
                                         /*wait_fence=*/NULL, &finish_fence),
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
    fflush(stdout);
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
  iree_hal_fence_release(finish_fence);

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

  // Transfer outputs to the host so they can be processed. Only required when
  // using full HAL device-based execution.
  if (iree_status_is_ok(status) && device != NULL) {
    iree_hal_buffer_params_t target_params = {
        .usage = IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING,
        .access = IREE_HAL_MEMORY_ACCESS_ALL,
        .type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
                IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
        .queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY,
        .min_alignment = 0,
    };
    status = iree_tooling_transfer_variants(
        outputs, device, device_allocator, target_params,
        /*wait_fence=*/NULL, /*signal_fence=*/NULL);
  }

  // Wrap stdout for printing results.
  iree_io_stream_t* stdout_stream = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_status_annotate_f(
        iree_io_stdio_stream_wrap(IREE_IO_STREAM_MODE_WRITABLE, stdout,
                                  /*owns_handle=*/false, host_allocator,
                                  &stdout_stream),
        "opening stdout stream");
  }

  // Handle either printing/writing the outputs or checking them against
  // expected values (basic pass/fail testing).
  if (iree_status_is_ok(status)) {
    status = iree_status_annotate_f(
        iree_tooling_process_results(device, results_cconv, outputs,
                                     stdout_stream, host_allocator,
                                     out_exit_code),
        "processing function outputs");
  }
  iree_vm_list_release(outputs);

  iree_io_stream_release(stdout_stream);
  fflush(stdout);

  return status;
}

static iree_status_t iree_tooling_process_results(
    iree_hal_device_t* device, iree_string_view_t results_cconv,
    iree_vm_list_t* results, iree_io_stream_t* stream,
    iree_allocator_t host_allocator, int* out_exit_code) {
  *out_exit_code = EXIT_SUCCESS;

  // Basic output handling to route to the console or files.
  if (FLAG_expected_output_list().count == 0) {
    if (FLAG_output_list().count == 0) {
      // Print all outputs.
      return iree_status_annotate_f(
          iree_tooling_print_variants(
              IREE_SV("result"), results,
              (iree_host_size_t)FLAG_output_max_element_count, stream,
              host_allocator),
          "printing results");
    } else {
      // Write (or ignore) all outputs.
      return iree_status_annotate_f(
          iree_tooling_write_variants(
              results, FLAG_output_list(),
              (iree_host_size_t)FLAG_output_max_element_count, stream,
              host_allocator),
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
      iree_tooling_parse_variants(results_cconv, FLAG_expected_output_list(),
                                  device, heap_allocator, host_allocator,
                                  &expected_list),
      "parsing expected function outputs");

  // Compare expected vs actual lists and output diffs.
  if (iree_status_is_ok(status)) {
    bool did_match = iree_tooling_compare_variant_lists(expected_list, results,
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

  // Annotate errors with the function description.
  if (!iree_status_is_ok(status)) {
    status = iree_tooling_annotate_status_with_function_decl(status, function);
  }

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

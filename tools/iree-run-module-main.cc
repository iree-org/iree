// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

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

IREE_FLAG(string, function, "",
          "Name of a function contained in the module specified by --module= "
          "to run.");

IREE_FLAG(bool, print_statistics, false,
          "Prints runtime statistics to stderr on exit.");

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

namespace iree {
namespace {

iree_status_t Run(int* out_exit_code) {
  IREE_TRACE_SCOPE0("iree-run-module");

  iree_allocator_t host_allocator = iree_allocator_system();
  vm::ref<iree_vm_instance_t> instance;
  IREE_RETURN_IF_ERROR(iree_tooling_create_instance(host_allocator, &instance),
                       "creating instance");

  iree_tooling_module_list_t module_list;
  iree_tooling_module_list_initialize(&module_list);
  IREE_RETURN_IF_ERROR(iree_tooling_load_modules_from_flags(
      instance.get(), host_allocator, &module_list));

  vm::ref<iree_vm_context_t> context;
  vm::ref<iree_hal_device_t> device;
  vm::ref<iree_hal_allocator_t> device_allocator;
  IREE_RETURN_IF_ERROR(iree_tooling_create_context_from_flags(
      instance.get(), module_list.count, module_list.values,
      /*default_device_uri=*/iree_string_view_empty(), host_allocator, &context,
      &device, &device_allocator));

  std::string function_name = std::string(FLAG_function);
  iree_vm_function_t function;
  if (function_name.empty()) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "no --function= specified");
  } else {
    IREE_RETURN_IF_ERROR(
        iree_vm_module_lookup_function_by_name(
            iree_tooling_module_list_back(&module_list),
            IREE_VM_FUNCTION_LINKAGE_EXPORT,
            iree_string_view_t{function_name.data(), function_name.size()},
            &function),
        "looking up function '%s'", function_name.c_str());
  }

  IREE_RETURN_IF_ERROR(iree_hal_begin_profiling_from_flags(device.get()));

  vm::ref<iree_vm_list_t> inputs;
  IREE_RETURN_IF_ERROR(iree_tooling_parse_to_variant_list(
      device_allocator.get(), FLAG_input_list().values, FLAG_input_list().count,
      host_allocator, &inputs));

  // If the function is async add fences so we can invoke it synchronously.
  vm::ref<iree_hal_fence_t> finish_fence;
  IREE_RETURN_IF_ERROR(iree_tooling_append_async_fence_inputs(
      inputs.get(), &function, device.get(), /*wait_fence=*/NULL,
      &finish_fence));

  vm::ref<iree_vm_list_t> outputs;
  IREE_RETURN_IF_ERROR(iree_vm_list_create(iree_vm_make_undefined_type_def(),
                                           16, host_allocator, &outputs));

  printf("EXEC @%s\n", function_name.c_str());
  IREE_RETURN_IF_ERROR(
      iree_vm_invoke(context.get(), function, IREE_VM_INVOCATION_FLAG_NONE,
                     /*policy=*/nullptr, inputs.get(), outputs.get(),
                     host_allocator),
      "invoking function '%s'", function_name.c_str());

  // If the function is async we need to wait for it to complete.
  if (finish_fence) {
    IREE_RETURN_IF_ERROR(
        iree_hal_fence_wait(finish_fence.get(), iree_infinite_timeout()));
  }

  IREE_RETURN_IF_ERROR(iree_hal_end_profiling_from_flags(device.get()));

  if (FLAG_expected_output_list().count == 0) {
    if (FLAG_output_list().count == 0) {
      IREE_RETURN_IF_ERROR(
          iree_tooling_variant_list_fprint(
              IREE_SV("result"), outputs.get(),
              (iree_host_size_t)FLAG_output_max_element_count, stdout),
          "printing results");
    } else {
      IREE_RETURN_IF_ERROR(
          iree_tooling_output_variant_list(
              outputs.get(), FLAG_output_list().values,
              FLAG_output_list().count,
              (iree_host_size_t)FLAG_output_max_element_count, stdout),
          "outputting results");
    }
  } else {
    // Parse expected list into host-local memory that we can easily access.
    // Note that we return a status here as this can fail on user inputs.
    vm::ref<iree_hal_allocator_t> heap_allocator;
    IREE_RETURN_IF_ERROR(iree_hal_allocator_create_heap(
        IREE_SV("heap"), host_allocator, host_allocator, &heap_allocator));
    vm::ref<iree_vm_list_t> expected_list;
    IREE_RETURN_IF_ERROR(iree_tooling_parse_to_variant_list(
        heap_allocator.get(), FLAG_expected_output_list().values,
        FLAG_expected_output_list().count, host_allocator, &expected_list));

    // Compare expected vs actual lists and output diffs.
    bool did_match = iree_tooling_compare_variant_lists(
        expected_list.get(), outputs.get(), host_allocator, stdout);
    if (did_match) {
      printf("[SUCCESS] all function outputs matched their expected values.\n");
    }
    *out_exit_code = did_match ? EXIT_SUCCESS : EXIT_FAILURE;
  }

  // Grab any instrumentation data present in the module and write it to disk.
  IREE_RETURN_IF_ERROR(
      iree_tooling_process_instrument_data(context.get(), host_allocator));

  // Release resources before gathering statistics.
  inputs.reset();
  outputs.reset();
  iree_tooling_module_list_reset(&module_list);
  context.reset();

  if (device_allocator && FLAG_print_statistics) {
    IREE_IGNORE_ERROR(
        iree_hal_allocator_statistics_fprint(stderr, device_allocator.get()));
  }

  device_allocator.reset();
  device.reset();
  instance.reset();
  return iree_ok_status();
}

}  // namespace

extern "C" int main(int argc, char** argv) {
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);
  if (argc > 1) {
    // Avoid iree-run-module spinning endlessly on stdin if the user uses single
    // dashes for flags.
    printf(
        "[ERROR] unexpected positional argument (expected none)."
        " Did you use pass a flag with a single dash ('-')?"
        " Use '--' instead.\n");
    return 1;
  }

  int exit_code = EXIT_SUCCESS;
  iree_status_t status = Run(&exit_code);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    return EXIT_FAILURE;
  }

  return exit_code;
}

}  // namespace iree

// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/process_results.h"

#include <stdio.h>
#include <stdlib.h>

#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/api.h"
#include "iree/io/stdio_stream.h"
#include "iree/tooling/comparison.h"
#include "iree/tooling/function_io.h"
#include "iree/vm/api.h"

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

iree_status_t iree_tooling_process_results(iree_hal_device_t* device,
                                           iree_string_view_t results_cconv,
                                           iree_vm_list_t* results,
                                           iree_io_stream_t* stream,
                                           iree_allocator_t host_allocator,
                                           int* out_exit_code) {
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

iree_status_t iree_tooling_process_results_and_print(
    iree_hal_device_t* device, iree_string_view_t results_cconv,
    iree_vm_list_t* results, iree_allocator_t host_allocator,
    int* out_exit_code) {
  // Wrap stdout for printing results.
  iree_io_stream_t* stdout_stream = NULL;
  iree_status_t status = iree_status_annotate_f(
      iree_io_stdio_stream_wrap(IREE_IO_STREAM_MODE_WRITABLE, stdout,
                                /*owns_handle=*/false, host_allocator,
                                &stdout_stream),
      "opening stdout stream");

  // Handle either printing/writing the outputs or checking them against
  // expected values (basic pass/fail testing).
  if (iree_status_is_ok(status)) {
    status = iree_status_annotate_f(
        iree_tooling_process_results(device, results_cconv, results,
                                     stdout_stream, host_allocator,
                                     out_exit_code),
        "processing function outputs");
  }

  iree_io_stream_release(stdout_stream);
  fflush(stdout);

  return status;
}

// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Assembles textual VM assembly into a VM bytecode module.
//
// $ iree-as-module --output=module.vmfb module.vmasm

#include <stdio.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"
#include "iree/io/file_contents.h"
#include "iree/vm/bytecode/assembler/assembler.h"

IREE_FLAG(string, output, "", "Output .vmfb file path.");

int main(int argc, char** argv) {
  IREE_TRACE_APP_ENTER();

  iree_allocator_t host_allocator = iree_allocator_system();
  int exit_code = EXIT_SUCCESS;

  iree_flags_set_usage(
      "iree-as-module",
      "Assembles textual VM assembly into an IREE VM bytecode module.\n"
      "$ iree-as-module --output=module.vmfb module.vmasm\n"
      "$ iree-as-module --output=module.vmfb -\n");
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);

  iree_status_t status = iree_ok_status();
  if (argc != 2 || strlen(FLAG_output) == 0) {
    fprintf(stderr,
            "Syntax: iree-as-module --output=module.vmfb module.vmasm\n");
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "expected one input file and --output");
  }

  iree_io_file_contents_t* input_contents = NULL;
  if (iree_status_is_ok(status)) {
    if (strcmp(argv[1], "-") == 0) {
      status =
          iree_io_file_contents_read_stdin(host_allocator, &input_contents);
    } else {
      status = iree_io_file_contents_read(iree_make_cstring_view(argv[1]),
                                          host_allocator, &input_contents);
    }
  }

  iree_byte_span_t archive = iree_byte_span_empty();
  if (iree_status_is_ok(status)) {
    iree_string_view_t source =
        iree_make_string_view((const char*)input_contents->const_buffer.data,
                              input_contents->const_buffer.data_length);
    status =
        iree_vm_bytecode_assembler_assemble(source, host_allocator, &archive);
  }

  if (iree_status_is_ok(status)) {
    status = iree_io_file_contents_write(iree_make_cstring_view(FLAG_output),
                                         iree_const_cast_byte_span(archive),
                                         host_allocator);
  }

  iree_allocator_free(host_allocator, archive.data);
  iree_io_file_contents_free(input_contents);

  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    exit_code = EXIT_FAILURE;
  }
  fflush(stderr);
  IREE_TRACE_APP_EXIT(exit_code);
  return exit_code;
}

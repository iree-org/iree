// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>

#include "iree/base/api.h"
#include "iree/base/internal/file_io.h"
#include "iree/schemas/bytecode_module_def_json_printer.h"

// Today we just print to JSON. We could do something more useful (size
// analysis, etc), but JSON should be enough.
//
// We could also move all of this into iree-compile (mlir -> vmfb -> json),
// though having a tiny little tool not reliant on LLVM is nice (can run this
// on a device).
int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "Syntax: iree-dump-module module.vmfb > module.json\n");
    return 1;
  }

  iree_file_contents_t* flatbuffer_contents = NULL;
  IREE_CHECK_OK(iree_file_read_contents(argv[1], iree_allocator_system(),
                                        &flatbuffer_contents));

  // Print direct to stdout.
  flatcc_json_printer_t printer;
  flatcc_json_printer_init(&printer, /*fp=*/NULL);
  flatcc_json_printer_set_skip_default(&printer, true);
  bytecode_module_def_print_json(
      &printer, (const char*)flatbuffer_contents->const_buffer.data,
      flatbuffer_contents->const_buffer.data_length);
  flatcc_json_printer_clear(&printer);

  iree_file_contents_free(flatbuffer_contents);

  return 0;
}

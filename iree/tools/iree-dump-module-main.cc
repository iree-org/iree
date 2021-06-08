// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <iostream>
#include <string>
#include <utility>

#include "iree/base/api.h"
#include "iree/base/internal/file_io.h"
#include "iree/schemas/bytecode_module_def_json_printer.h"
#include "iree/tools/utils/vm_util.h"

// Today we just print to JSON. We could do something more useful (size
// analysis, etc), but JSON should be enough.
//
// We could also move all of this into iree-translate (mlir -> vmfb -> json),
// though having a tiny little tool not reliant on LLVM is nice (can run this
// on a device).
extern "C" int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Syntax: iree-dump-module module.vmfb > module.json\n";
    return 1;
  }
  std::string module_contents;
  IREE_CHECK_OK(iree::GetFileContents(argv[1], &module_contents));

  // Print direct to stdout.
  flatcc_json_printer_t printer;
  flatcc_json_printer_init(&printer, /*fp=*/nullptr);
  flatcc_json_printer_set_skip_default(&printer, true);
  bytecode_module_def_print_json(
      &printer, reinterpret_cast<const char*>(module_contents.data()),
      module_contents.size());
  flatcc_json_printer_clear(&printer);

  return 0;
}

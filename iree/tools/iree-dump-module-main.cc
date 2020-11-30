// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <iostream>
#include <string>
#include <utility>

#include "iree/base/file_mapping.h"
#include "iree/schemas/bytecode_module_def_json_printer.h"

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
  auto module_mapping_or = iree::FileMapping::OpenRead(argv[1]);
  if (!module_mapping_or.ok()) {
    std::cerr << module_mapping_or.status();
    return 1;
  }
  auto module_mapping = std::move(module_mapping_or.value());
  auto module_buffer = module_mapping->data();

  // Print direct to stdout.
  flatcc_json_printer_t printer;
  flatcc_json_printer_init(&printer, /*fp=*/nullptr);
  flatcc_json_printer_set_skip_default(&printer, true);
  bytecode_module_def_print_json(
      &printer, reinterpret_cast<const char*>(module_buffer.data()),
      module_buffer.size());
  flatcc_json_printer_clear(&printer);

  return 0;
}

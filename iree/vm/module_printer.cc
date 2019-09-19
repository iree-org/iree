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

#include "iree/vm/module_printer.h"

#include "iree/vm/bytecode_printer.h"
#include "iree/vm/source_map.h"

namespace iree {
namespace vm {

Status PrintModuleToStream(OpcodeTable opcode_table, const Module& module,
                           PrintModuleFlagBitfield flags,
                           std::ostream* stream) {
  // TODO(benvanik): custom FunctionTable Function iterator.
  for (int i = 0; i < module.function_table().def().functions()->size(); ++i) {
    ASSIGN_OR_RETURN(const auto& function,
                     module.function_table().LookupFunction(i));
    if (function.def().bytecode()) {
      auto source_map_resolver =
          AllBitsSet(flags, PrintModuleFlag::kIncludeSourceMapping)
              ? SourceMapResolver::FromFunction(module.def(), i)
              : SourceMapResolver();
      BytecodePrinter printer(opcode_table, module.function_table(),
                              module.executable_table(), source_map_resolver);
      *stream << "Function " << i << ": " << function << "\n";
      RETURN_IF_ERROR(
          printer.PrintToStream(*function.def().bytecode(), stream));
      *stream << "\n";
    } else {
      *stream << "Function " << i << ": " << function.name() << " (import)\n";
    }
  }
  return OkStatus();
}

Status PrintModuleToStream(OpcodeTable opcode_table, const Module& module,
                           std::ostream* stream) {
  return PrintModuleToStream(opcode_table, module, PrintModuleFlag::kNone,
                             stream);
}

}  // namespace vm
}  // namespace iree

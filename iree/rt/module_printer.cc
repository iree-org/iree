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

#include "iree/rt/module_printer.h"

#include <iomanip>

#include "iree/rt/disassembler.h"
#include "iree/rt/source_resolver.h"

namespace iree {
namespace rt {

Status PrintModuleToStream(const Module& module, PrintModuleFlagBitfield flags,
                           std::ostream* stream) {
  *stream << "Imports:\n";
  for (int i = 0; i < module.signature().import_function_count(); ++i) {
    ASSIGN_OR_RETURN(auto function, module.LookupFunctionByOrdinal(
                                        Function::Linkage::kImport, i));
    *stream << "  " << i << ": " << function << "\n";
  }
  *stream << "Exports:\n";
  for (int i = 0; i < module.signature().export_function_count(); ++i) {
    ASSIGN_OR_RETURN(auto function, module.LookupFunctionByOrdinal(
                                        Function::Linkage::kExport, i));
    *stream << "  " << i << ": " << function << "\n";
  }
  if (module.signature().internal_function_count()) {
    *stream << "Internal:\n";
    auto* disassembler = module.disassembler();
    for (int i = 0; i < module.signature().internal_function_count(); ++i) {
      ASSIGN_OR_RETURN(auto function, module.LookupFunctionByOrdinal(
                                          Function::Linkage::kInternal, i));
      *stream << "  " << i << ": " << function << "\n";
      if (disassembler && AllBitsSet(flags, PrintModuleFlag::kDisassemble)) {
        auto instructions_or =
            disassembler->DisassembleInstructions(function, 0);
        if (IsUnavailable(instructions_or.status())) continue;
        for (const auto& instruction : instructions_or.ValueOrDie()) {
          *stream << "    " << std::setw(6) << instruction.offset << ": "
                  << instruction.long_text << "\n";
        }
      }
    }
  }
  return OkStatus();
}

Status PrintModuleToStream(const Module& module, std::ostream* stream) {
  return PrintModuleToStream(module, PrintModuleFlag::kNone, stream);
}

}  // namespace rt
}  // namespace iree

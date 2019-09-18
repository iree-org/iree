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

#ifndef THIRD_PARTY_MLIR_EDGE_IREE_VM_MODULE_PRINTER_H_
#define THIRD_PARTY_MLIR_EDGE_IREE_VM_MODULE_PRINTER_H_

#include <ostream>

#include "third_party/mlir_edge/iree/base/bitfield.h"
#include "third_party/mlir_edge/iree/base/status.h"
#include "third_party/mlir_edge/iree/vm/module.h"
#include "third_party/mlir_edge/iree/vm/opcode_info.h"

namespace iree {
namespace vm {

enum class PrintModuleFlag {
  kNone = 0,
  kIncludeSourceMapping = 1,
};
IREE_BITFIELD(PrintModuleFlag);
using PrintModuleFlagBitfield = PrintModuleFlag;

// Prints all functions within the module to the given |stream|.
Status PrintModuleToStream(OpcodeTable opcode_table, const Module& module,
                           std::ostream* stream);
Status PrintModuleToStream(OpcodeTable opcode_table, const Module& module,
                           PrintModuleFlagBitfield flags, std::ostream* stream);

}  // namespace vm
}  // namespace iree

#endif  // THIRD_PARTY_MLIR_EDGE_IREE_VM_MODULE_PRINTER_H_

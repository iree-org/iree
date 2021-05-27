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

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"
#include "mlir/TableGen/Attribute.h"
#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/Operator.h"

namespace mlir {
namespace iree_compiler {
namespace {

using ::llvm::format_hex;
using ::llvm::formatv;
using ::llvm::Record;

void emitOpTable(const llvm::RecordKeeper &recordKeeper, const Record &tableDef,
                 raw_ostream &os) {
  std::vector<const Record *> opEncodings(256);
  for (auto *opcodeDef : tableDef.getValueAsListOfDefs("enumerants")) {
    opEncodings[opcodeDef->getValueAsInt("value")] = opcodeDef;
  }

  os << "typedef enum {\n";
  for (int i = 0; i < 256; ++i) {
    if (auto *opcode = opEncodings[i]) {
      os << formatv("  IREE_VM_OP_{0}_{1} = {2}",
                    tableDef.getValueAsString("opcodeEnumTag"),
                    opcode->getValueAsString("symbol"), format_hex(i, 4, true));
    } else {
      os << formatv("  IREE_VM_OP_{0}_RSV_{1}",
                    tableDef.getValueAsString("opcodeEnumTag"),
                    format_hex(i, 4, true));
    }
    os << ",\n";
  }
  os << "} " << tableDef.getValueAsString("opcodeEnumName") << ";\n";
  os << "\n";

  os << formatv("#define IREE_VM_OP_{0}_TABLE(OPC, RSV) \\\n",
                tableDef.getValueAsString("opcodeEnumTag"));
  for (int i = 0; i < 256; ++i) {
    if (auto *opcode = opEncodings[i]) {
      os << formatv("    OPC({0}, {1})", format_hex(i, 4, true),
                    opcode->getValueAsString("symbol"));
    } else {
      os << formatv("    RSV({0})", format_hex(i, 4, true));
    }
    if (i != 255) {
      os << " \\\n";
    }
  }
  os << "\n\n";
}

// Finds all opcode tables in VMBase.td and emits a enum and template table for
// their opcode and name.
bool emitOpTableDefs(const llvm::RecordKeeper &recordKeeper, raw_ostream &os) {
  llvm::emitSourceFileHeader("IREE VM Operation Tables", os);

  auto defs = recordKeeper.getAllDerivedDefinitions("VM_OPC_EnumAttr");
  for (const auto *def : defs) {
    emitOpTable(recordKeeper, *def, os);
  }

  return false;
}

static GenRegistration genVMOpDispatcherDefs(
    "gen-iree-vm-op-table-defs",
    "Generates IREE VM operation table macros for runtime use",
    [](const llvm::RecordKeeper &records, raw_ostream &os) {
      return emitOpTableDefs(records, os);
    });

}  // namespace
}  // namespace iree_compiler
}  // namespace mlir

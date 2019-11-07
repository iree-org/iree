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

#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"
#include "mlir/Support/STLExtras.h"
#include "mlir/TableGen/Attribute.h"
#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/Operator.h"

namespace mlir {
namespace iree_compiler {
namespace {

using ::llvm::format_hex;
using ::llvm::formatv;
using ::llvm::Record;

// Finds all serializable ops and emits a enum and template table for their
// opcode and name.
bool emitOpTableDefs(const llvm::RecordKeeper &recordKeeper, raw_ostream &os) {
  llvm::emitSourceFileHeader("IREE VM Operation Tables", os);

  std::vector<const Record *> opRecords(256);
  auto defs = recordKeeper.getAllDerivedDefinitions("VM_Op");
  for (const auto *def : defs) {
    if (def->isValueUnset("encoding")) continue;
    auto encodingExprs = def->getValueAsListOfDefs("encoding");
    for (auto encodingExpr : encodingExprs) {
      if (encodingExpr->getType()->getAsString() == "VM_EncOpcode") {
        auto *opcode = encodingExpr->getValueAsDef("opcode");
        opRecords[opcode->getValueAsInt("value")] = def;
        break;
      }
    }
  }

  os << "typedef enum {\n";
  for (int i = 0; i < 256; ++i) {
    auto *def = opRecords[i];
    if (def) {
      auto encodingExprs = def->getValueAsListOfDefs("encoding");
      auto *opcode = encodingExprs.front()->getValueAsDef("opcode");
      os << formatv("  IREE_VM_OP_{0} = {1}",
                    opcode->getValueAsString("symbol"), format_hex(i, 4, true));
    } else {
      os << formatv("  IREE_VM_OP_RSV_{0}", format_hex(i, 4, true));
    }
    os << ",\n";
  }
  os << "} iree_vm_op_t;\n";
  os << "\n";

  os << "#define IREE_VM_OP_TABLE(OPC, RSV) \\\n";
  for (int i = 0; i < 256; ++i) {
    auto *def = opRecords[i];
    if (def) {
      auto encodingExprs = def->getValueAsListOfDefs("encoding");
      auto *opcode = encodingExprs.front()->getValueAsDef("opcode");
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

  return false;
}

static GenRegistration genVMOpDispatcherDefs(
    "gen-vm-op-table-defs",
    "Generates IREE VM operation table macros for runtime use",
    [](const llvm::RecordKeeper &records, raw_ostream &os) {
      return emitOpTableDefs(records, os);
    });

}  // namespace
}  // namespace iree_compiler
}  // namespace mlir

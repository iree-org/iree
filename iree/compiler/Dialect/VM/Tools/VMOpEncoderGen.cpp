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
#include "llvm/Support/FormatVariadic.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"
#include "mlir/TableGen/Attribute.h"
#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/Operator.h"

namespace mlir {
namespace iree_compiler {
namespace {

using ::llvm::formatv;
using ::llvm::Record;
using ::mlir::tblgen::Operator;

bool emitEncodeFnDefs(const llvm::RecordKeeper &recordKeeper, raw_ostream &os) {
  llvm::emitSourceFileHeader("IREE VM Operation Encoder Definitions", os);

  // Gather prefix opcodes:
  DenseMap<StringRef, int> prefixOpcodes;
  auto opcodes = recordKeeper.getAllDerivedDefinitions("VM_OPC");
  for (const auto *opcode : opcodes) {
    auto symbol = opcode->getValueAsString("symbol");
    if (symbol.startswith("Prefix")) {
      prefixOpcodes[symbol] = opcode->getValueAsInt("value");
    }
  }

  auto defs = recordKeeper.getAllDerivedDefinitions("VM_Op");
  for (const auto *def : defs) {
    if (def->isValueUnset("encoding")) continue;
    auto encodingExprs = def->getValueAsListOfDefs("encoding");
    if (encodingExprs.empty()) continue;

    Operator op(def);
    os << formatv(
        "LogicalResult {0}::encode(SymbolTable &syms, VMFuncEncoder &e) {{\n",
        op.getQualCppClassName());

    for (auto &pair : prefixOpcodes) {
      std::string traitName = (StringRef("OpTrait::IREE::VM::") +
                               pair.first.substr(strlen("Prefix")))
                                  .str();
      if (op.getTrait(traitName)) {
        os << formatv(
            "  if (failed(e.encodeOpcode(\"{0}\", {1}))) return emitOpError() "
            "<< "
            "\"failed to encode op prefix\";\n",
            pair.first, pair.second);
      }
    }

    os << "  if (";
    interleave(
        encodingExprs, os,
        [&](Record *encodingExpr) {
          os << formatv("failed({0})", encodingExpr->getValueAsString("expr"));
        },
        " ||\n      ");
    os << ") {\n";
    os << "    return emitOpError() << \"failed to encode (internal)\";\n";
    os << "  }\n";

    os << "  return success();\n";
    os << "}\n\n";
  }

  return false;
}

static GenRegistration genVMOpEncoderDefs(
    "gen-iree-vm-op-encoder-defs",
    "Generates IREE VM operation encoder definitions (.cpp)",
    [](const llvm::RecordKeeper &records, raw_ostream &os) {
      return emitEncodeFnDefs(records, os);
    });

}  // namespace
}  // namespace iree_compiler
}  // namespace mlir

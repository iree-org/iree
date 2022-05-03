// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"
#include "mlir/TableGen/Attribute.h"
#include "mlir/TableGen/CodeGenHelpers.h"
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
    tblgen::NamespaceEmitter emitter(os, op.getDialect());
    os << formatv(
        "LogicalResult {0}::encode(SymbolTable &syms, VMFuncEncoder &e) {{\n",
        op.getCppClassName());

    for (auto &pair : prefixOpcodes) {
      std::string traitName = (StringRef("::mlir::OpTrait::IREE::VM::") +
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

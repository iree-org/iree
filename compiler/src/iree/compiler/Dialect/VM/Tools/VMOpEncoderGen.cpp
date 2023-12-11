// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"
#include "mlir/TableGen/CodeGenHelpers.h"
#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/Operator.h"

namespace mlir::iree_compiler {

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
    if (def->isValueUnset("encoding"))
      continue;
    auto encodingExprs = def->getValueAsListOfDefs("encoding");
    if (encodingExprs.empty())
      continue;

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
    auto printOneCondition = [&](Record *encodingExpr) {
      StringRef expr = encodingExpr->getValueAsString("expr");
      std::vector<StringRef> params =
          encodingExpr->getValueAsListOfStrings("params");
      assert(params.size() <= 1);

      // Note the following relies on the fact that only encoding expressions
      // involving operands/results have one parameter. It's a bit inflexible,
      // but it works for now and we can change when the extra flexibility is
      // really needed.
      std::string param;
      if (params.size() == 1) {
        param = "get" + llvm::convertToCamelFromSnakeCase(params.front(), true);
      } else {
        param = expr;
      }
      os << formatv("failed({0})", formatv(expr.data(), param));
    };
    interleave(encodingExprs, os, printOneCondition, " ||\n      ");
    os << ") {\n";
    os << "    return emitOpError() << \"failed to encode (internal)\";\n";
    os << "  }\n";

    os << "  return success();\n";
    os << "}\n\n";
  }

  return false;
}

static GenRegistration
    genVMOpEncoderDefs("gen-iree-vm-op-encoder-defs",
                       "Generates IREE VM operation encoder definitions (.cpp)",
                       [](const llvm::RecordKeeper &records, raw_ostream &os) {
                         return emitEncodeFnDefs(records, os);
                       });

} // namespace

} // namespace mlir::iree_compiler

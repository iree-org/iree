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

#include "compiler/IR/StructureOps.h"

#include "compiler/IR/Types.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/STLExtras.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {

//===----------------------------------------------------------------------===//
// Generic printers and parsers.
//===----------------------------------------------------------------------===//

// Parses an op that has no inputs and no outputs.
static ParseResult parseNoIOOp(OpAsmParser &parser, OperationState &state) {
  if (failed(parser.parseOptionalAttributeDict(state.attributes))) {
    return failure();
  }
  return success();
}

// Prints an op that has no inputs and no outputs.
static void printNoIOOp(Operation *op, OpAsmPrinter &printer) {
  printer << op->getName();
  printer.printOptionalAttrDict(op->getAttrs());
}

//===----------------------------------------------------------------------===//
// iree.module
//===----------------------------------------------------------------------===//

void ModuleOp::build(Builder *builder, OperationState &state) {
  ensureTerminator(*state.addRegion(), *builder, state.location);
}

static ParseResult parseModuleOp(OpAsmParser &parser, OperationState &state) {
  Region *body = state.addRegion();
  if (parser.parseRegion(*body, /*arguments=*/{}, /*argTypes=*/{})) {
    return failure();
  }
  if (parser.parseOptionalAttributeDict(state.attributes)) {
    return failure();
  }
  ModuleOp::ensureTerminator(*body, parser.getBuilder(), state.location);
  return success();
}

static void printModuleOp(OpAsmPrinter &printer, Operation *op) {
  printer << op->getName();
  printer.printRegion(op->getRegion(0), /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/false);
  printer.printOptionalAttrDict(op->getAttrs());
}

//===----------------------------------------------------------------------===//
// iree.multi_arch_executable
//===----------------------------------------------------------------------===//

void MultiArchExecutableOp::build(Builder *builder, OperationState &state,
                                  StringRef name) {
  state.addAttribute(SymbolTable::getSymbolAttrName(),
                     builder->getStringAttr(name));
  ensureTerminator(*state.addRegion(), *builder, state.location);
}

static ParseResult parseMultiArchExecutableOp(OpAsmParser &parser,
                                              OperationState &state) {
  auto &builder = parser.getBuilder();

  // Parse the name as a symbol reference attr and then convert to a string.
  SymbolRefAttr nameAttr;
  if (failed(parser.parseAttribute(nameAttr, SymbolTable::getSymbolAttrName(),
                                   state.attributes))) {
    return failure();
  }
  state.attributes.back().second = builder.getStringAttr(nameAttr.getValue());

  if (succeeded(parser.parseOptionalLSquare())) {
    IntegerAttr ordinalAttr;
    if (failed(parser.parseAttribute(ordinalAttr, builder.getIntegerType(32),
                                     "iree.ordinal", state.attributes)) ||
        failed(parser.parseRSquare())) {
      return failure();
    }
  }

  if (failed(parser.parseLParen()) || failed(parser.parseRParen())) {
    return failure();
  }

  Region *body = state.addRegion();
  if (failed(parser.parseRegion(*body, /*arguments=*/{}, /*argTypes=*/{}))) {
    return failure();
  }
  if (succeeded(parser.parseOptionalKeyword("attributes"))) {
    if (failed(parser.parseOptionalAttributeDict(state.attributes))) {
      return failure();
    }
  }

  MultiArchExecutableOp::ensureTerminator(*body, builder, state.location);

  return success();
}

static void printMultiArchExecutableOp(OpAsmPrinter &printer,
                                       MultiArchExecutableOp op) {
  printer << op.getOperationName() << " @" << op.sym_name();
  if (auto ordinalAttr =
          op.getAttr("iree.ordinal").dyn_cast_or_null<IntegerAttr>()) {
    printer << "[" << ordinalAttr.getInt() << "]";
  }
  printer << "()";

  printer.printRegion(op.body(), /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/false);

  // Print out executable attributes, if present.
  SmallVector<StringRef, 2> ignoredAttrs = {
      SymbolTable::getSymbolAttrName(),
      "iree.ordinal",
  };
  SmallVector<NamedAttribute, 4> attrs(
      llvm::make_filter_range(op.getAttrs(), [&](const NamedAttribute &attr) {
        return llvm::count(ignoredAttrs, attr.first) == 0;
      }));
  if (!attrs.empty()) {
    printer << "\n    attributes ";
    printer.printOptionalAttrDict(attrs);
  }
}

//===----------------------------------------------------------------------===//
// iree.executable
//===----------------------------------------------------------------------===//

void ExecutableOp::build(Builder *builder, OperationState &state,
                         IREE::ExecutableFormat format) {
  state.addAttribute("format",
                     builder->getI32IntegerAttr(static_cast<uint32_t>(format)));
  ensureTerminator(*state.addRegion(), *builder, state.location);
}

static ParseResult parseExecutableOp(OpAsmParser &parser,
                                     OperationState &state) {
  auto &builder = parser.getBuilder();

  if (succeeded(parser.parseOptionalLSquare())) {
    IntegerAttr ordinalAttr;
    if (failed(parser.parseAttribute(ordinalAttr, builder.getIntegerType(32),
                                     "iree.ordinal", state.attributes)) ||
        failed(parser.parseRSquare())) {
      return failure();
    }
  }

  IntegerAttr executableOrdinalAttr;
  StringAttr formatAttr;
  llvm::SMLoc formatLoc;
  if (failed(parser.parseLParen()) ||
      failed(parser.getCurrentLocation(&formatLoc)) ||
      failed(parser.parseAttribute(formatAttr, "format", state.attributes))) {
    return failure();
  }
  auto format = symbolizeExecutableFormat(formatAttr.getValue());
  if (!format.hasValue()) {
    return parser.emitError(formatLoc)
           << "Unknown executable format " << formatAttr.getValue();
  }
  state.attributes.back().second =
      builder.getI32IntegerAttr(static_cast<int32_t>(format.getValue()));

  Region *body = state.addRegion();
  if (failed(parser.parseRegion(*body, /*arguments=*/{}, /*argTypes=*/{}))) {
    return failure();
  }
  if (succeeded(parser.parseOptionalKeyword("attributes"))) {
    if (failed(parser.parseOptionalAttributeDict(state.attributes))) {
      return failure();
    }
  }

  ExecutableOp::ensureTerminator(*body, parser.getBuilder(), state.location);

  return success();
}

static void printExecutableOp(OpAsmPrinter &printer, ExecutableOp op) {
  printer << op.getOperationName();
  if (auto ordinalAttr =
          op.getAttr("iree.ordinal").dyn_cast_or_null<IntegerAttr>()) {
    printer << "[" << ordinalAttr.getInt() << "]";
  }
  printer << "(";
  auto format = symbolizeExecutableFormat(op.format());
  if (format.hasValue()) {
    printer << stringifyExecutableFormat(format.getValue());
  } else {
    printer << "INVALID FORMAT";
  }
  printer << ")";

  printer.printRegion(op.body(), /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/false);

  // Print out executable attributes, if present.
  SmallVector<StringRef, 2> ignoredAttrs = {
      "iree.ordinal",
      "format",
  };
  SmallVector<NamedAttribute, 4> attrs(
      llvm::make_filter_range(op.getAttrs(), [&](const NamedAttribute &attr) {
        return llvm::count(ignoredAttrs, attr.first) == 0;
      }));
  if (!attrs.empty()) {
    printer << "\n      attributes ";
    printer.printOptionalAttrDict(attrs);
  }
}

#define GET_OP_CLASSES
#include "compiler/IR/StructureOps.cpp.inc"

}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

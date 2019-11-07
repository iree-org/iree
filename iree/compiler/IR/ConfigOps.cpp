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

#include "iree/compiler/IR/ConfigOps.h"

#include "iree/compiler/IR/StructureOps.h"
#include "iree/compiler/IR/Types.h"
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
  if (failed(parser.parseOptionalAttrDict(state.attributes))) {
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
// iree.target_config
//===----------------------------------------------------------------------===//

void ExecutableTargetConfigOp::build(Builder *builder, OperationState &state,
                                     std::string backend) {
  state.addAttribute("backend", builder->getStringAttr(backend));
  ensureTerminator(*state.addRegion(), *builder, state.location);
}

static ParseResult parseExecutableTargetConfigOp(OpAsmParser &parser,
                                                 OperationState &state) {
  llvm::SMLoc backendLoc;
  StringAttr backendAttr;
  if (failed(parser.parseLParen()) ||
      failed(parser.getCurrentLocation(&backendLoc)) ||
      failed(parser.parseAttribute(backendAttr, "backend", state.attributes))) {
    return failure();
  }

  Region *body = state.addRegion();
  if (failed(parser.parseRegion(*body, /*arguments=*/{}, /*argTypes=*/{}))) {
    return failure();
  }
  if (failed(parser.parseOptionalAttrDictWithKeyword(state.attributes))) {
    return failure();
  }

  ExecutableTargetConfigOp::ensureTerminator(*body, parser.getBuilder(),
                                             state.location);

  return success();
}

static void printExecutableTargetConfigOp(OpAsmPrinter &printer,
                                          ExecutableTargetConfigOp op) {
  printer << op.getOperationName() << "(" << op.backend() << ")";

  printer.printRegion(op.body(), /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/false);

  printer.printOptionalAttrDictWithKeyword(op.getAttrs(),
                                           /*elidedAttrs=*/{"backend"});
}

#define GET_OP_CLASSES
#include "iree/compiler/IR/ConfigOps.cpp.inc"

}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

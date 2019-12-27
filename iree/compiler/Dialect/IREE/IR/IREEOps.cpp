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

#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/SMLoc.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/STLExtras.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {

//===----------------------------------------------------------------------===//
// iree.do_not_optimize
//===----------------------------------------------------------------------===//

ParseResult parseDoNotOptimizeOp(OpAsmParser &parser, OperationState &state) {
  SmallVector<OpAsmParser::OperandType, 2> args;

  if (failed(parser.parseLParen()) || failed(parser.parseOperandList(args)) ||
      failed(parser.parseRParen()) ||
      failed(parser.parseOptionalAttrDict(state.attributes)) ||
      failed(parser.parseOptionalColonTypeList(state.types))) {
    return failure();
  }

  // Operands and results have the same types.
  auto &operandTypes = state.types;
  parser.resolveOperands(args, operandTypes, parser.getCurrentLocation(),
                         state.operands);

  return success();
}

void printDoNotOptimizeOp(OpAsmPrinter &p, Operation *op) {
  p << "ireex.do_not_optimize";
  p << "(";
  p.printOperands(op->getOperands());
  p << ")";
  p.printOptionalAttrDict(op->getAttrs());

  if (op->getNumOperands() != 0) {
    p << " : ";
    interleaveComma(op->getOperandTypes(), p);
  }
}

static LogicalResult verifyDoNotOptimizeOp(DoNotOptimizeOp op) {
  if (op.getNumOperands() != op.getNumResults()) {
    return op.emitOpError()
           << "must have same number of operands and results, but has "
           << op.getNumOperands() << " and " << op.getNumResults()
           << ", respectively";
  }

  for (int i = 0, e = op.getNumOperands(); i < e; ++i) {
    if (op.getOperand(i).getType() != op.getResult(i).getType()) {
      op.emitOpError() << "must have same operand and result types, but they "
                          "differ at index "
                       << i;
    }
  }

  return success();
}

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/IREE/IR/IREEOps.cpp.inc"

}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

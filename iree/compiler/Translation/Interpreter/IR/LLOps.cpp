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

#include "iree/compiler/Translation/Interpreter/IR/LLOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/OpImplementation.h"

namespace mlir {
namespace iree_compiler {
namespace IREEInterp {
namespace LL {

//===----------------------------------------------------------------------===//
// iree_ll_interp.call
//===----------------------------------------------------------------------===//

FunctionType CallOp::getCalleeType() {
  SmallVector<Type, 8> argTypes(getOperandTypes());
  return FunctionType::get(argTypes, getResultTypes(), getContext());
}

//===----------------------------------------------------------------------===//
// iree_ll_interp.br
//===----------------------------------------------------------------------===//

Block *BranchOp::getDest() { return getOperation()->getSuccessor(0); }

void BranchOp::setDest(Block *block) {
  return getOperation()->setSuccessor(block, 0);
}

void BranchOp::eraseOperand(unsigned index) {
  getOperation()->eraseOperand(index);
}

Optional<OperandRange> BranchOp::getSuccessorOperands(unsigned index) {
  assert(index == 0 && "invalid successor index");
  return getOperands();
}

bool BranchOp::canEraseSuccessorOperand() { return true; }

//===----------------------------------------------------------------------===//
// iree_ll_interp.cond_br
//===----------------------------------------------------------------------===//

static ParseResult parseCondBranchOp(OpAsmParser &parser,
                                     OperationState &result) {
  SmallVector<Value, 4> destOperands;
  Block *dest;
  OpAsmParser::OperandType condInfo;

  // Parse the condition.
  Type int1Ty = parser.getBuilder().getI1Type();
  if (parser.parseOperand(condInfo) || parser.parseComma() ||
      parser.resolveOperand(condInfo, int1Ty, result.operands)) {
    return parser.emitError(parser.getNameLoc(),
                            "expected condition type was boolean (i1)");
  }

  // Parse the true successor.
  if (parser.parseSuccessorAndUseList(dest, destOperands)) return failure();
  result.addSuccessors(dest);
  result.addOperands(destOperands);

  // Parse the false successor.
  destOperands.clear();
  if (parser.parseComma() ||
      parser.parseSuccessorAndUseList(dest, destOperands))
    return failure();
  result.addSuccessors(dest);
  result.addOperands(destOperands);

  return success();
}

static void printCondBranchOp(OpAsmPrinter &p, CondBranchOp op) {
  p << "iree_ll_interp.cond_br ";
  p.printOperand(op.getCondition());
  p << ", ";
  p.printSuccessorAndUseList(op.trueDest(), op.trueDestOperands());
  p << ", ";
  p.printSuccessorAndUseList(op.falseDest(), op.falseDestOperands());
}

Optional<OperandRange> CondBranchOp::getSuccessorOperands(unsigned index) {
  assert(index < getNumSuccessors() && "invalid successor index");
  return index == trueIndex ? getTrueOperands() : getFalseOperands();
}

bool CondBranchOp::canEraseSuccessorOperand() { return true; }

#define GET_OP_CLASSES
#include "iree/compiler/Translation/Interpreter/IR/LLOps.cpp.inc"

}  // namespace LL
}  // namespace IREEInterp
}  // namespace iree_compiler
}  // namespace mlir

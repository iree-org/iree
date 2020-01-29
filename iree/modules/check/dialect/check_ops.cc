// Copyright 2020 Google LLC
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

#include "iree/modules/check/dialect/check_ops.h"

#include "mlir/IR/OpImplementation.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Check {

void printExpectTrueOp(OpAsmPrinter &p, ExpectTrueOp op) {
  p << "check.expect_true";
  p << "(";
  p.printOperand(op.getOperand());
  p << ")";
  p.printOptionalAttrDict(op.getAttrs());
  p << " : ";
  p.printType(op.getOperand().getType());
}

ParseResult parseExpectTrueOp(OpAsmParser &parser, OperationState &state) {
  OpAsmParser::OperandType arg;
  Type type;
  return failure(parser.parseLParen() || parser.parseOperand(arg) ||
                 parser.parseRParen() ||
                 parser.parseOptionalAttrDict(state.attributes) ||
                 parser.parseColonType(type) ||
                 parser.resolveOperand(arg, type, state.operands));
}

#define GET_OP_CLASSES
#include "iree/modules/check/dialect/check_ops.cc.inc"

}  // namespace Check
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

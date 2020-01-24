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

#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"

#include "iree/compiler/Dialect/Shape/IR/ShapeTypes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/SMLoc.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/STLExtras.h"

namespace mlir {
namespace iree_compiler {
namespace Shape {

//===----------------------------------------------------------------------===//
// Canonicalization
//===----------------------------------------------------------------------===//

class ElideTiedGetRankedShapePattern
    : public OpRewritePattern<GetRankedShapeOp> {
  using OpRewritePattern::OpRewritePattern;
  PatternMatchResult matchAndRewrite(GetRankedShapeOp op,
                                     PatternRewriter &rewriter) const override {
    // If the immediate predecessor is a TieShapeOp, then this op can be
    // erased in favor of the input to the tie op.
    if (!matchPattern(op.operand(), m_Op<TieShapeOp>())) {
      return matchFailure();
    }

    auto tieOp = cast<TieShapeOp>(op.operand().getDefiningOp());
    rewriter.replaceOp(op, tieOp.shape(), op.operand());

    return matchSuccess();
  }
};

//===----------------------------------------------------------------------===//
// iree.tie_shape
//===----------------------------------------------------------------------===//

static ParseResult parseTieShapeOp(OpAsmParser &parser, OperationState &state) {
  SmallVector<OpAsmParser::OperandType, 2> operands;
  SmallVector<Type, 2> operandTypes;
  if (parser.parseOperandList(operands) ||
      parser.parseColonTypeList(operandTypes) ||
      parser.parseOptionalAttrDict(state.attributes) ||
      parser.resolveOperands(operands, operandTypes, parser.getNameLoc(),
                             state.operands)) {
    return failure();
  }

  // The result type is the same as the first operand.
  if (state.operands.empty()) return failure();
  state.types.push_back(state.operands.front().getType());
  return success();
}

static void printTieShapeOp(OpAsmPrinter &p, TieShapeOp op) {
  p << op.getOperationName() << " ";
  p.printOperands(op.getOperands());
  p << " : ";
  interleaveComma(op.getOperandTypes(), p);
  p.printOptionalAttrDict(op.getOperation()->getAttrs());
}

static LogicalResult verifyTieShapeOp(TieShapeOp op) {
  if (op.operand().getType() != op.result().getType()) {
    return op.emitOpError("operand and result must be the same type");
  }

  // tie_shape currently only supports ranked tensors.
  auto rankedTensorType = op.operand().getType().dyn_cast<RankedTensorType>();
  auto rsType = op.shape().getType().dyn_cast<RankedShapeType>();
  if (!rankedTensorType || !rsType) {
    return op.emitOpError("currently only ranked tensors are supported");
  }

  SmallVector<int64_t, 4> rsDims;
  rsType.getAllDims(rsDims);
  if (!rankedTensorType.getShape().equals(rsDims)) {
    return op.emitOpError("dims must match between tensor and shape");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// iree.get_ranked_shape
//===----------------------------------------------------------------------===//

static ParseResult parseGetRankedShapeOp(OpAsmParser &parser,
                                         OperationState &state) {
  OpAsmParser::OperandType operandType;
  Type resultType;
  return failure(
      parser.parseOperand(operandType) || parser.parseColonType(resultType) ||
      parser.parseOptionalArrowTypeList(state.types) ||
      parser.resolveOperand(operandType, resultType, state.operands));
}

static void printGetRankedShapeOp(OpAsmPrinter &p, GetRankedShapeOp op) {
  p << op.getOperationName() << " ";
  p.printOperand(op.operand());
  p << " : ";
  p.printType(op.operand().getType());
  p << " -> ";
  p.printType(op.shape().getType());
}

static LogicalResult verifyGetRankedShapeOp(GetRankedShapeOp op) {
  auto tensorType = op.operand().getType().cast<TensorType>();
  auto rsType = op.shape().getType().cast<RankedShapeType>();
  if (tensorType.getRank() != rsType.getRank()) {
    return op.emitOpError("operand and result must be of same rank");
  }
  SmallVector<int64_t, 4> rsDims;
  rsType.getAllDims(rsDims);
  if (!std::equal(rsDims.begin(), rsDims.end(),
                  tensorType.getShape().begin())) {
    return op.emitOpError("operand tensor and result shape must be equal");
  }
  return success();
}

void GetRankedShapeOp::getCanonicalizationPatterns(
    OwningRewritePatternList &patterns, MLIRContext *context) {
  patterns.insert<ElideTiedGetRankedShapePattern>(context);
}

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.cpp.inc"

}  // namespace Shape
}  // namespace iree_compiler
}  // namespace mlir

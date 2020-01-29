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
#include "mlir/IR/StandardTypes.h"
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

class ElideDuplicateGetRankedShapePattern
    : public OpRewritePattern<GetRankedShapeOp> {
  using OpRewritePattern::OpRewritePattern;
  PatternMatchResult matchAndRewrite(GetRankedShapeOp op,
                                     PatternRewriter &rewriter) const override {
    // If the immediate predecessor is a GetRankedShapeOp, then this op can be
    // erased in favor of the input to the tie op.
    if (!matchPattern(op.operand(), m_Op<GetRankedShapeOp>())) {
      return matchFailure();
    }

    auto precedingGetRankedShapeOp =
        cast<GetRankedShapeOp>(op.operand().getDefiningOp());
    rewriter.replaceOp(op, precedingGetRankedShapeOp.shape(), op.operand());
    return matchSuccess();
  }
};

class ElideStaticGetRankedShapePattern
    : public OpRewritePattern<GetRankedShapeOp> {
  using OpRewritePattern::OpRewritePattern;
  PatternMatchResult matchAndRewrite(GetRankedShapeOp op,
                                     PatternRewriter &rewriter) const override {
    auto operandType = op.operand().getType().dyn_cast<RankedTensorType>();
    auto shapeType = op.shape().getType().dyn_cast<RankedShapeType>();
    if (!operandType || !shapeType || !operandType.hasStaticShape()) {
      return matchFailure();
    }

    rewriter.replaceOpWithNewOp<ConstRankedShapeOp>(op, shapeType);
    return matchSuccess();
  }
};

class SafeCastCompatibleShapePattern
    : public OpRewritePattern<CastCompatibleShapeOp> {
  using OpRewritePattern::OpRewritePattern;
  PatternMatchResult matchAndRewrite(CastCompatibleShapeOp op,
                                     PatternRewriter &rewriter) const override {
    // TODO(laurenzo): This is just eliding if everything is the same. Make
    // it generic.
    auto resultRs = op.result().getType().dyn_cast<RankedShapeType>();
    if (resultRs) {
      // Casting to a ranked shape.
      for (auto operandType : op.getOperandTypes()) {
        auto operandRs = operandType.dyn_cast<RankedShapeType>();
        if (!operandRs || operandRs != resultRs) {
          return matchFailure();
        }
      }
      rewriter.replaceOp(op, op.operands()[0]);
      return matchSuccess();
    }

    return matchFailure();
  }
};

//===----------------------------------------------------------------------===//
// shape.tie_shape
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
// shape.cast_compatible_shape
//===----------------------------------------------------------------------===//

static ParseResult parseCastCompatibleShapeOp(OpAsmParser &parser,
                                              OperationState &state) {
  SmallVector<OpAsmParser::OperandType, 2> operands;
  SmallVector<Type, 2> operandTypes;
  if (parser.parseOperandList(operands) ||
      parser.parseColonTypeList(operandTypes) ||
      parser.parseOptionalArrowTypeList(state.types) ||
      parser.parseOptionalAttrDict(state.attributes) ||
      parser.resolveOperands(operands, operandTypes, parser.getNameLoc(),
                             state.operands)) {
    return failure();
  }

  return success();
}

static void printCastCompatibleShapeOp(OpAsmPrinter &p,
                                       CastCompatibleShapeOp op) {
  p << op.getOperationName() << " ";
  p.printOperands(op.operands());
  p << " : ";
  interleaveComma(op.getOperandTypes(), p);
  p << " -> ";
  p.printType(op.result().getType());
  p.printOptionalAttrDict(op.getOperation()->getAttrs());
}

static LogicalResult verifyCastCompatibleShapeOp(CastCompatibleShapeOp op) {
  if (op.operands().empty()) {
    return op.emitOpError() << "Must have at least one operand";
  }

  auto resultRs = op.result().getType().dyn_cast<RankedShapeType>();
  if (resultRs) {
    // TODO(laurenzo): Expand this to check true compatibility instead of
    // just equality.
    // Casting to a ranked shape.
    for (auto operandType : op.getOperandTypes()) {
      auto operandRs = operandType.dyn_cast<RankedShapeType>();
      if (!operandRs || operandRs != resultRs) {
        return op.emitOpError()
               << "Incompatible static shape cast: " << operandRs << " -> "
               << resultRs;
      }
    }
    return success();
  }

  return failure();
}

void CastCompatibleShapeOp::getCanonicalizationPatterns(
    OwningRewritePatternList &patterns, MLIRContext *context) {
  patterns.insert<SafeCastCompatibleShapePattern>(context);
}

//===----------------------------------------------------------------------===//
// shape.get_ranked_shape
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
  patterns
      .insert<ElideTiedGetRankedShapePattern, ElideStaticGetRankedShapePattern,
              ElideDuplicateGetRankedShapePattern>(context);
}

//===----------------------------------------------------------------------===//
// shape.const_ranked_shape
//===----------------------------------------------------------------------===//

void ConstRankedShapeOp::build(Builder *builder, OperationState &result,
                               Type type) {
  result.addAttribute("value", UnitAttr::get(builder->getContext()));
  result.types.push_back(type);
}

static ParseResult parseConstRankedShapeOp(OpAsmParser &parser,
                                           OperationState &state) {
  Type resultType;
  if (parser.parseColonType(resultType)) {
    return failure();
  }
  state.types.push_back(resultType);
  return success();
}

static void printConstRankedShapeOp(OpAsmPrinter &p, ConstRankedShapeOp op) {
  p << op.getOperationName() << " : ";
  p.printType(op.result().getType());
}

static LogicalResult verifyConstRankedShapeOp(ConstRankedShapeOp op) {
  auto rsType = op.result().getType().dyn_cast<RankedShapeType>();
  if (!rsType || !rsType.isFullyStatic()) {
    return op.emitOpError("must be a fully static ranked_shape");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// shape.ranked_dim
//===----------------------------------------------------------------------===//

ParseResult parseRankedDimOp(OpAsmParser &parser, OperationState &state) {
  OpAsmParser::OperandType operand;
  Type operandType;
  IntegerAttr indexAttr;
  Type indexType = parser.getBuilder().getIndexType();
  if (parser.parseOperand(operand) || parser.parseLSquare() ||
      parser.parseAttribute(indexAttr, indexType, "index", state.attributes) ||
      parser.parseRSquare() || parser.parseColonType(operandType) ||
      parser.resolveOperand(operand, operandType, state.operands)) {
    return failure();
  }

  auto rsType = operandType.dyn_cast<RankedShapeType>();
  if (!rsType) {
    return parser.emitError(parser.getNameLoc());
  }
  state.types.push_back(rsType.getDimType());
  return success();
}

static void printRankedDimOp(OpAsmPrinter &p, RankedDimOp op) {
  p << op.getOperationName() << " ";
  p.printOperand(op.shape());
  p << "[" << op.getIndex() << "]";
  p << " : ";
  p.printType(op.shape().getType());
}

static LogicalResult verifyRankedDimOp(RankedDimOp op) {
  auto resultType = op.result().getType();
  auto rsType = op.shape().getType().dyn_cast<RankedShapeType>();
  if (resultType != rsType.getDimType()) {
    return op.emitOpError()
           << "expected result of type " << rsType.getDimType();
  }
  auto index = static_cast<int64_t>(op.getIndex());
  if (index < 0 || index >= rsType.getRank()) {
    return op.emitOpError() << "index out of bounds of shape";
  }
  return success();
}

OpFoldResult RankedDimOp::fold(ArrayRef<Attribute> operand) {
  auto rsType = shape().getType().cast<RankedShapeType>();
  int index = getIndex();
  if (!rsType.isDimDynamic(index)) {
    auto dimSize = rsType.getStaticDim(index);
    return IntegerAttr::get(rsType.getDimType(), dimSize);
  }

  return {};
}

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.cpp.inc"

}  // namespace Shape
}  // namespace iree_compiler
}  // namespace mlir

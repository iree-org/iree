// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"

#include "iree/compiler/Dialect/Shape/IR/ShapeTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/SMLoc.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace iree_compiler {
namespace Shape {

//===----------------------------------------------------------------------===//
// shapex.const_ranked_shape
//===----------------------------------------------------------------------===//

void ConstRankedShapeOp::build(OpBuilder &builder, OperationState &result,
                               Type type) {
  assert(type.cast<RankedShapeType>().isFullyStatic());
  result.types.push_back(type);
}

static LogicalResult verifyConstRankedShapeOp(ConstRankedShapeOp op) {
  auto rsType = op.result().getType().dyn_cast<RankedShapeType>();
  if (!rsType || !rsType.isFullyStatic()) {
    return op.emitOpError("must be a fully static ranked_shape");
  }
  return success();
}

void ConstRankedShapeOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  auto rankedShape = result().getType().cast<RankedShapeType>();
  SmallString<32> buffer;
  llvm::raw_svector_ostream os(buffer);
  os << "rs";
  interleave(
      rankedShape.getAllDims(), os, [&](int64_t dim) { os << dim; }, "_");
  setNameFn(getResult(), os.str());
}

//===----------------------------------------------------------------------===//
// shapex.make_ranked_shape
//===----------------------------------------------------------------------===//

static LogicalResult verifyMakeRankedShapeOp(MakeRankedShapeOp op) {
  if (op.getRankedShapeType().getNumDynamicDims() != op.getNumOperands()) {
    return op.emitError()
           << "number of dynamic dims doesn't match number of operands";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// shapex.ranked_dim
//===----------------------------------------------------------------------===//

void RankedDimOp::build(OpBuilder &builder, OperationState &result,
                        Type dimType, Value shape, int index) {
  result.addOperands(shape);
  result.addAttribute("index",
                      builder.getIntegerAttr(builder.getIndexType(), index));
  result.addTypes(dimType);
}

void RankedDimOp::build(OpBuilder &builder, OperationState &result, Value shape,
                        int index) {
  RankedDimOp::build(builder, result, builder.getIndexType(), shape, index);
}

ParseResult parseRankedDimOp(OpAsmParser &parser, OperationState &state) {
  OpAsmParser::OperandType operand;
  Type operandType;
  IntegerAttr indexAttr;
  Type indexType = parser.getBuilder().getIndexType();
  SmallVector<Type, 1> resultTypes;
  if (parser.parseOperand(operand) || parser.parseLSquare() ||
      parser.parseAttribute(indexAttr, indexType, "index", state.attributes) ||
      parser.parseRSquare() || parser.parseColonType(operandType) ||
      parser.parseArrowTypeList(resultTypes) || resultTypes.empty() ||
      parser.resolveOperand(operand, operandType, state.operands)) {
    return failure();
  }

  auto rsType = operandType.dyn_cast<RankedShapeType>();
  if (!rsType) {
    return parser.emitError(parser.getNameLoc());
  }
  state.types.push_back(resultTypes[0]);
  return success();
}

static void printRankedDimOp(OpAsmPrinter &p, RankedDimOp op) {
  p << " ";
  p.printOperand(op.shape());
  p << "[" << op.getIndex() << "]";
  p << " : ";
  p.printType(op.shape().getType());
  p << " -> ";
  p.printType(op.getType());
}

static LogicalResult verifyRankedDimOp(RankedDimOp op) {
  auto rsType = op.shape().getType().dyn_cast<RankedShapeType>();
  auto index = static_cast<int64_t>(op.getIndex());
  if (index < 0 || index >= rsType.getRank()) {
    return op.emitOpError() << "index out of bounds of shape";
  }
  return success();
}

}  // namespace Shape
}  // namespace iree_compiler
}  // namespace mlir

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.cpp.inc"

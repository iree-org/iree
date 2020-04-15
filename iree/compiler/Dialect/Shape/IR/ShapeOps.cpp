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
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/SMLoc.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
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

namespace mlir {
namespace iree_compiler {
namespace Shape {

//===----------------------------------------------------------------------===//
// shape.tie_shape
//===----------------------------------------------------------------------===//

void TieShapeOp::build(Builder *builder, OperationState &result, Value operand,
                       Value shape) {
  result.types.push_back(operand.getType());
  result.addOperands({operand, shape});
}

static LogicalResult verifyTieShapeOp(TieShapeOp op) {
  if (op.operand().getType() != op.result().getType()) {
    return op.emitOpError("must have the same operand and result type");
  }

  // Validate RankedTensorType and ranked_shape_type conservatively in this
  // case (tie_shape supports arbitrary operand() but we constrain it if
  // it is specific enough.
  auto rankedTensorType = op.operand().getType().dyn_cast<RankedTensorType>();
  auto rsType = op.shape().getType().dyn_cast<RankedShapeType>();
  if (rankedTensorType && rsType) {
    if (!rankedTensorType.getShape().equals(rsType.getAllDims())) {
      return op.emitOpError("dims must match between tensor and shape");
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// shape.cast_compatible_shape
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// shape.get_ranked_shape
//===----------------------------------------------------------------------===//

void GetRankedShapeOp::build(Builder *builder, OperationState &result,
                             Value operand) {
  auto rankedOperandType = operand.getType().dyn_cast<RankedTensorType>();
  if (rankedOperandType) {
    result.types.push_back(RankedShapeType::get(rankedOperandType.getShape(),
                                                builder->getContext()));
  }
  result.addOperands(operand);
}

static LogicalResult verifyGetRankedShapeOp(GetRankedShapeOp op) {
  auto tensorType = op.operand().getType().cast<TensorType>();
  auto rsType = op.shape().getType().cast<RankedShapeType>();
  if (tensorType.getRank() != rsType.getRank()) {
    return op.emitOpError("operand and result must be of same rank");
  }
  auto rsDims = rsType.getAllDims();
  if (!std::equal(rsDims.begin(), rsDims.end(),
                  tensorType.getShape().begin())) {
    return op.emitOpError("operand tensor and result shape must be equal");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// shape.const_ranked_shape
//===----------------------------------------------------------------------===//

void ConstRankedShapeOp::build(Builder *builder, OperationState &result,
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
// shape.make_ranked_shape
//===----------------------------------------------------------------------===//

static LogicalResult verifyMakeRankedShapeOp(MakeRankedShapeOp op) {
  if (op.getRankedShapeType().getNumDynamicDims() != op.getNumOperands()) {
    return op.emitError()
           << "number of dynamic dims doesn't match number of operands";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// shape.ranked_dim
//===----------------------------------------------------------------------===//

void RankedDimOp::build(Builder *builder, OperationState &result, Type dimType,
                        Value shape, int index) {
  result.addOperands(shape);
  result.addAttribute("index",
                      builder->getIntegerAttr(builder->getIndexType(), index));
  result.addTypes(dimType);
}

void RankedDimOp::build(Builder *builder, OperationState &result, Value shape,
                        int index) {
  RankedDimOp::build(builder, result, builder->getIndexType(), shape, index);
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
  p << op.getOperationName() << " ";
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

//===----------------------------------------------------------------------===//
// shape.ranked_dims
//===----------------------------------------------------------------------===//

void RankedDimsOp::build(Builder *builder, OperationState &result, Type dimType,
                         Value shape) {
  result.addOperands(shape);
  auto rankedShapeType = shape.getType().cast<RankedShapeType>();
  for (int i = 0; i < rankedShapeType.getRank(); ++i) {
    result.types.push_back(dimType);
  }
}

void RankedDimsOp::build(Builder *builder, OperationState &result,
                         Value shape) {
  RankedDimsOp::build(builder, result, builder->getIndexType(), shape);
}

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.cpp.inc"

}  // namespace Shape
}  // namespace iree_compiler
}  // namespace mlir

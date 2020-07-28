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
// shapex.tie_shape
//===----------------------------------------------------------------------===//

static LogicalResult verifyTieShapeOp(TieShapeOp op) {
  // Validate shapedType and ranked_shape_type conservatively in this
  // case (tie_shape supports arbitrary operand() but we constrain it if
  // it is specific enough.
  auto shapedType = op.operand().getType().dyn_cast<ShapedType>();
  auto rsType = op.shape().getType().dyn_cast<RankedShapeType>();
  if (shapedType && shapedType.hasRank() && rsType) {
    if (!shapedType.getShape().equals(rsType.getAllDims())) {
      return op.emitOpError("dims must match between tensor and shape");
    }
  }

  return success();
}

Value TieShapeOp::getViewSource() { return operand(); }

//===----------------------------------------------------------------------===//
// shapex.cast_compatible_shape
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
// shapex.get_ranked_shape
//===----------------------------------------------------------------------===//

void GetRankedShapeOp::build(OpBuilder &builder, OperationState &result,
                             Value operand) {
  auto rankedOperandType = operand.getType().dyn_cast<RankedTensorType>();
  if (rankedOperandType) {
    result.types.push_back(RankedShapeType::get(rankedOperandType.getShape(),
                                                builder.getContext()));
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
// shapex.ranked_dims
//===----------------------------------------------------------------------===//

void RankedDimsOp::build(OpBuilder &builder, OperationState &result,
                         Type dimType, Value shape) {
  result.addOperands(shape);
  auto rankedShapeType = shape.getType().cast<RankedShapeType>();
  for (int i = 0; i < rankedShapeType.getRank(); ++i) {
    result.types.push_back(dimType);
  }
}

void RankedDimsOp::build(OpBuilder &builder, OperationState &result,
                         Value shape) {
  RankedDimsOp::build(builder, result, builder.getIndexType(), shape);
}

//===----------------------------------------------------------------------===//
// shapex.gather_extents
//===----------------------------------------------------------------------===//

// Helper for accessing attributes for inferReturnTypes callback.
// That helper gives the attributes as an `ArrayRef<NamedAttribute>` which isn't
// the nicest form.
template <typename Attr>
static Attr getRequiredAttr(DictionaryAttr attributes, StringRef name) {
  return attributes.get(name).cast<Attr>();
}

/*static*/ SmallVector<int64_t, 6> GatherExtentsOp::getConcatenatedExtents(
    ValueRange values) {
  SmallVector<int64_t, 6> ret;
  for (auto type : values.getTypes()) {
    auto rankedShape = type.cast<RankedShapeType>();
    ret.append(rankedShape.getAllDims().begin(),
               rankedShape.getAllDims().end());
  }
  return ret;
}

static LogicalResult verifyGatherExtentsOp(GatherExtentsOp op) {
  int64_t totalExtents = 0;
  for (Type type : op.shapes().getTypes()) {
    totalExtents += type.cast<RankedShapeType>().getRank();
  }

  for (int64_t index : op.indices().getValues<int64_t>()) {
    if (index >= totalExtents) {
      return op.emitError() << "index " << index
                            << " exceeds total number of extents of operands ("
                            << totalExtents << ")";
    }
  }

  return success();
}

LogicalResult GatherExtentsOp::inferReturnTypes(
    MLIRContext *context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  // We can't infer the DimType of the result if there are no operands.
  // If a user requires this, then they should manually specify the return type.
  // We could in theory use an index type here (the default).
  assert(!operands.empty() && "inferring return type for empty operands");
  auto indices = getRequiredAttr<DenseIntElementsAttr>(attributes, "indices")
                     .getValues<int64_t>();
  auto inputExtents = getConcatenatedExtents(operands);
  SmallVector<int64_t, 6> resultExtents;
  for (auto index : indices) {
    resultExtents.push_back(inputExtents[index]);
  }
  inferredReturnTypes.push_back(RankedShapeType::get(resultExtents, context));
  return success();
}

//===----------------------------------------------------------------------===//
// shapex.to_extent_tensor
//===----------------------------------------------------------------------===//

LogicalResult ToExtentTensorOp::inferReturnTypes(
    MLIRContext *context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  auto inputType = operands[0].getType().cast<RankedShapeType>();
  inferredReturnTypes.push_back(
      RankedTensorType::get({inputType.getRank()}, IndexType::get(context)));
  return success();
}

//===----------------------------------------------------------------------===//
// shapex.from_extent_tensor
//===----------------------------------------------------------------------===//

static bool isValidTensorOfExtents(RankedTensorType type) {
  // If the tensor of extents is not static shapes, that would imply that the
  // tensor whose shape it is describing is unranked.
  return type.getRank() == 1 && type.hasStaticShape();
}

LogicalResult FromExtentTensorOp::inferReturnTypes(
    MLIRContext *context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  auto inputType = operands[0].getType().dyn_cast<RankedTensorType>();
  if (!inputType || !isValidTensorOfExtents(inputType)) {
    return emitOptionalError(location, "Invalid input type, ",
                             operands[0].getType(),
                             ", for from_extent_tensor op");
  }
  SmallVector<int64_t, 6> extents(inputType.getDimSize(0),
                                  static_cast<int64_t>(-1));
  inferredReturnTypes.push_back(RankedShapeType::get(extents, context));
  return success();
}

bool FromExtentTensorOp::isCompatibleReturnTypes(ArrayRef<Type> lhs,
                                                 ArrayRef<Type> rhs) {
  auto lhsRs = lhs[0].cast<RankedShapeType>();
  auto rhsRs = rhs[0].cast<RankedShapeType>();
  return succeeded(
      verifyCompatibleShape(lhsRs.getAllDims(), rhsRs.getAllDims()));
}

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.cpp.inc"

}  // namespace Shape
}  // namespace iree_compiler
}  // namespace mlir

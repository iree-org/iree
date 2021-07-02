// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Shape/IR/Builders.h"

#include "iree/compiler/Dialect/Shape/IR/ShapeDialect.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeTypes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Diagnostics.h"

namespace mlir {
namespace iree_compiler {
namespace Shape {

static Value getRankedShapeFromOpResult(Operation *op, Value resultValue,
                                        OpBuilder &builder) {
  if (!op) return nullptr;
  if (auto carryingOp = dyn_cast<ShapeCarryingInterface>(op)) {
    return carryingOp.buildResultValueRankedShape(resultValue, builder);
  } else {
    return nullptr;
  }
}

static Value getRankedShapeFromOpOperand(Operation *op, unsigned idx,
                                         OpBuilder &builder) {
  auto carryingOp = dyn_cast_or_null<ShapeCarryingInterface>(op);
  if (!carryingOp) {
    auto value = op->getOperand(idx);
    auto definingOp = value.getDefiningOp();
    if (!definingOp) return nullptr;
    return getRankedShapeFromOpResult(definingOp, value, builder);
  }
  return carryingOp.buildOperandRankedShape(idx, builder);
}

static Value findRankedShapeFromUse(Value value, OpBuilder &builder) {
  Value rs = getRankedShapeFromOpResult(value.getDefiningOp(), value, builder);
  if (rs) return rs;
  for (auto &use : value.getUses()) {
    rs = getRankedShapeFromOpOperand(use.getOwner(), use.getOperandNumber(),
                                     builder);
    if (rs) return rs;
  }
  return nullptr;
}

Value buildRankedShapeForValue(Location loc, Value shapedValue,
                               ValueRange dynamicDims, OpBuilder &builder) {
  auto shapedType = shapedValue.getType().dyn_cast<ShapedType>();
  assert(shapedType && "only valid to call on shaped types");
  return builder.createOrFold<Shape::MakeRankedShapeOp>(
      loc, Shape::RankedShapeType::get(shapedType), dynamicDims);
}

// Slices out a range of |dynamicDims| corresponding to the value at |index|.
static ValueRange sliceDynamicDims(unsigned index, ValueRange values,
                                   ValueRange dynamicDims) {
  auto valueType = values[index].getType().dyn_cast<ShapedType>();
  assert(valueType && "must be a shaped type to get dims");
  unsigned dimsIndex = 0;
  for (unsigned i = 0; i < index; ++i) {
    if (auto shapedType = values[i].getType().dyn_cast<ShapedType>()) {
      dimsIndex += shapedType.getNumDynamicDims();
    }
  }
  return dynamicDims.slice(dimsIndex, valueType.getNumDynamicDims());
}

Value buildRankedShapeForValueInList(Location loc, unsigned index,
                                     ValueRange flatValues,
                                     ValueRange flatDynamicDims,
                                     OpBuilder &builder) {
  auto dynamicDims = sliceDynamicDims(index, flatValues, flatDynamicDims);
  return buildRankedShapeForValue(loc, flatValues[index], dynamicDims, builder);
}

Value buildCastInputsToResultShape(Location loc,
                                   RankedShapeType resultShapeType,
                                   ArrayRef<Value> inputs, OpBuilder &builder) {
  llvm::SmallVector<Value, 4> inputShapes;
  for (auto inputOperand : inputs) {
    auto inputOperandType = inputOperand.getType().dyn_cast<RankedTensorType>();
    RankedShapeType inputOperandShape = RankedShapeType::getChecked(
        inputOperandType.getShape(), inputOperand.getLoc());
    if (!inputOperandShape) return nullptr;

    inputShapes.push_back(
        builder.create<GetRankedShapeOp>(loc, inputOperandShape, inputOperand));
  }

  // Assert compatible.
  return builder.create<CastCompatibleShapeOp>(loc, resultShapeType,
                                               inputShapes);
}

Value buildDegenerateBroadcastRankedShape(
    Value srcShape, int dstRank, SmallVectorImpl<int64_t> &broadcastDims,
    OpBuilder &builder) {
  RankedShapeType srcRsType = srcShape.getType().dyn_cast<RankedShapeType>();
  if (!srcRsType) {
    return nullptr;
  }

  // Map output dims to input dims.
  SmallVector<int, 4> outputDimMap;  // Input dimension or -1 for expand.
  outputDimMap.resize(dstRank, -1);
  if (broadcastDims.empty()) {
    // Right align the broadcast dims.
    int leftPadding = dstRank - srcRsType.getRank();
    assert(leftPadding >= 0);
    for (int i = 0, e = srcRsType.getRank(); i < e; ++i) {
      outputDimMap[leftPadding + i] = i;
    }
  } else {
    // Explicitly provided broadcast dimensions.
    assert(broadcastDims.size() == srcRsType.getRank());
    for (int i = 0, e = broadcastDims.size(); i < e; ++i) {
      auto outputDimIndex = broadcastDims[i];
      assert(outputDimIndex < outputDimMap.size());
      outputDimMap[outputDimIndex] = i;
    }
  }

  // Compute dims for the new output ranked shape.
  SmallVector<int64_t, 4> outputAllDims;
  SmallVector<Value, 4> outputDynamicDims;
  for (int i = 0, e = outputDimMap.size(); i < e; ++i) {
    int inputDimIndex = outputDimMap[i];
    if (inputDimIndex < 0) {
      // Expand with 1-dim.
      outputAllDims.push_back(1);
    } else if (srcRsType.isDimDynamic(inputDimIndex)) {
      // Append dynamic source dim.
      outputAllDims.push_back(-1);
      auto dim = builder.create<RankedDimOp>(
          srcShape.getLoc(), builder.getIndexType(), srcShape, inputDimIndex);
      outputDynamicDims.push_back(dim);
    } else {
      // Append static source dim.
      outputAllDims.push_back(srcRsType.getStaticDim(inputDimIndex));
    }
  }

  auto dstRsType = RankedShapeType::get(outputAllDims, srcRsType.getContext());
  if (outputDynamicDims.empty()) {
    return builder.create<ConstRankedShapeOp>(srcShape.getLoc(), dstRsType);
  } else {
    return builder.create<MakeRankedShapeOp>(srcShape.getLoc(), dstRsType,
                                             outputDynamicDims);
  }
}

LogicalResult getRankedDimsFromRankedShape(Location loc, Value rsValue,
                                           bool createIntermediateOps,
                                           SmallVectorImpl<Value> &outDims,
                                           OpBuilder &builder) {
  Operation *op = rsValue.getDefiningOp();
  if (op &&
      (llvm::isa<MakeRankedShapeOp>(op) || llvm::isa<ConstRankedShapeOp>(op))) {
    unsigned dynamicDimIndex = 0;
    auto rsType = rsValue.getType().cast<RankedShapeType>();
    for (int i = 0, e = rsType.getRank(); i < e; ++i) {
      if (rsType.isDimDynamic(i)) {
        if (dynamicDimIndex >= op->getNumOperands()) {
          return emitError(loc, "mismatched dynamic dimensions");
        }
        Value dimValue = op->getOperand(dynamicDimIndex++);
        if (!dimValue) {
          return emitError(
              loc, "unable to find remapped value for ranked dim value");
        }
        outDims.push_back(dimValue);
      } else {
        outDims.push_back(
            builder.create<ConstantIndexOp>(loc, rsType.getStaticDim(i)));
      }
    }
  } else if (createIntermediateOps) {
    auto dimsOp = builder.create<Shape::RankedDimsOp>(loc, rsValue);
    outDims.resize(dimsOp.result().size());
    std::copy(dimsOp.result().begin(), dimsOp.result().end(), outDims.begin());
  } else {
    return emitError(loc,
                     "could not resolve ranked dimensions from metadata ops");
  }
  return success();
}

Value buildOrFindRankedShapeForValue(Location loc, Value value, Type dimType,
                                     OpBuilder &builder) {
  auto valueSt = value.getType().dyn_cast<ShapedType>();
  if (!valueSt) {
    builder.getContext()->getDiagEngine().emit(loc, DiagnosticSeverity::Error)
        << "cannot construct shape for non shaped value: " << value.getType();
    return nullptr;
  }
  if (valueSt.hasStaticShape()) {
    auto rsType =
        RankedShapeType::get(valueSt.getShape(), builder.getContext());
    return builder.createOrFold<ConstRankedShapeOp>(loc, rsType);
  }

  // Dynamic - walk the uses to find a tie_shape op (either this op or an
  // immediate use).
  Value rs = findRankedShapeFromUse(value, builder);
  if (!rs) {
    builder.getContext()->getDiagEngine().emit(loc, DiagnosticSeverity::Error)
        << "dynamically shaped value is missing a shape association via "
        << "tie_shape";
    return nullptr;
  }

  auto rsType = rs.getType().dyn_cast<RankedShapeType>();
  if (!rsType) {
    builder.getContext()->getDiagEngine().emit(loc, DiagnosticSeverity::Error)
        << "dynamically shaped value is not ranked (which is not yet "
        << "supported)";
    return nullptr;
  }
  return rs;
}

SmallVector<Value, 4> buildOrFindDynamicDimsForValue(Location loc, Value value,
                                                     OpBuilder &builder) {
  auto valueSt = value.getType().dyn_cast<ShapedType>();
  if (!valueSt) {
    builder.getContext()->getDiagEngine().emit(loc, DiagnosticSeverity::Error)
        << "cannot construct shape for non shaped value: " << value.getType();
    return {};
  }

  // Bail if all dimensions are static.
  if (valueSt.hasStaticShape()) {
    return {};
  }

  // Dynamic - walk the uses to find a tie_shape op (either this op or an
  // immediate use).
  SmallVector<Value, 4> result;
  Value rs = findRankedShapeFromUse(value, builder);
  if (rs) {
    auto rsType = rs.getType().dyn_cast<RankedShapeType>();
    if (!rsType) {
      builder.getContext()->getDiagEngine().emit(loc, DiagnosticSeverity::Error)
          << "dynamically shaped value is not ranked (which is not yet "
          << "supported)";
      return {};
    }
    for (unsigned i = 0; i < rsType.getRank(); ++i) {
      if (rsType.isDimDynamic(i)) {
        result.push_back(builder.createOrFold<Shape::RankedDimOp>(loc, rs, i));
      }
    }
  } else {
    // No tie information - insert std.dim ops that may later be used and
    // hopefully converted to ranked shape types.
    for (unsigned i = 0; i < valueSt.getRank(); ++i) {
      if (valueSt.isDynamicDim(i)) {
        result.push_back(builder.createOrFold<tensor::DimOp>(loc, value, i));
      }
    }
  }
  return result;
}

}  // namespace Shape
}  // namespace iree_compiler
}  // namespace mlir

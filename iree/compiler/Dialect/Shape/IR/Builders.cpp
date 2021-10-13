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

Optional<SmallVector<Value, 4>> buildOrFindDimsForValue(Location loc,
                                                        Value value,
                                                        OpBuilder &builder) {
  auto valueSt = value.getType().dyn_cast<ShapedType>();
  if (!valueSt) {
    builder.getContext()->getDiagEngine().emit(loc, DiagnosticSeverity::Error)
        << "cannot construct shape for non shaped value: " << value.getType();
    return llvm::None;
  }

  // Walk the uses to find a tie_shape op (either this op or an immediate use).
  SmallVector<Value, 4> result;
  Value rs = findRankedShapeFromUse(value, builder);
  if (rs) {
    auto rsType = rs.getType().dyn_cast<RankedShapeType>();
    if (!rsType) {
      builder.getContext()->getDiagEngine().emit(loc, DiagnosticSeverity::Error)
          << "dynamically shaped value is not ranked (which is not yet "
          << "supported)";
      return llvm::None;
    }
    for (unsigned i = 0; i < rsType.getRank(); ++i) {
      if (rsType.isDimDynamic(i)) {
        result.push_back(builder.createOrFold<Shape::RankedDimOp>(loc, rs, i));
      } else {
        result.push_back(builder.create<arith::ConstantIndexOp>(
            loc, rsType.getStaticDim(i)));
      }
    }
  } else {
    // No tie information - insert std.dim ops that may later be used and
    // hopefully converted to ranked shape types.
    for (unsigned i = 0; i < valueSt.getRank(); ++i) {
      if (valueSt.isDynamicDim(i)) {
        result.push_back(builder.createOrFold<tensor::DimOp>(loc, value, i));
      } else {
        result.push_back(
            builder.create<arith::ConstantIndexOp>(loc, valueSt.getDimSize(i)));
      }
    }
  }
  return result;
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

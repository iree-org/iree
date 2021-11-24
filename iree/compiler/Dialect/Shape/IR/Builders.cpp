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

}  // namespace Shape
}  // namespace iree_compiler
}  // namespace mlir

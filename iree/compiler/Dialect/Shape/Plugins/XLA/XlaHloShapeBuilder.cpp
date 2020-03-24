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

#include "iree/compiler/Dialect/Shape/Plugins/XLA/XlaHloShapeBuilder.h"

#include "iree/compiler/Dialect/Shape/IR/Builders.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeInterface.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "llvm/ADT/Optional.h"
#include "mlir/IR/Value.h"
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"

using llvm::None;
using llvm::Optional;
using llvm::SmallVector;

using namespace mlir::iree_compiler::Shape;

namespace mlir {
namespace xla_hlo {
namespace {

template <typename HloOp>
Value rewriteXlaBinaryElementwiseOpShape(RankedShapeType resultShape, HloOp op,
                                         OpBuilder &builder) {
  if (!op) return nullptr;

  if (op.broadcast_dimensions()) {
    // Has implicit broadcast - ignore for now.
    return nullptr;
  }

  // No implicit broadcast. Tread as same element type.
  SmallVector<Value, 4> inputOperands(op.getOperands());
  return buildCastInputsToResultShape(op.getLoc(), resultShape, inputOperands,
                                      builder);
}

Value rewriteXlaDotOpShape(RankedShapeType resultRs, DotOp dotOp,
                           OpBuilder &builder) {
  auto lhsType = dotOp.lhs().getType().dyn_cast<RankedTensorType>();
  auto rhsType = dotOp.rhs().getType().dyn_cast<RankedTensorType>();
  auto resultType = dotOp.getResult().getType().dyn_cast<RankedTensorType>();
  if (!lhsType || !rhsType || !resultType) return nullptr;

  // Shape transfer function:
  //  [n] dot [n] -> scalar
  //  [m x k] dot [k] -> [m]
  //  [m x k] dot [k x n] -> [m x n]
  auto lhsRank = lhsType.getRank();
  auto rhsRank = rhsType.getRank();
  auto resultRank = resultType.getRank();
  auto loc = dotOp.getLoc();
  if (lhsRank == 1 && rhsRank == 1 && resultRank == 0) {
    auto scalarShape = RankedShapeType::getChecked({}, loc);
    return builder.create<ConstRankedShapeOp>(dotOp.getLoc(), scalarShape);
  } else if (lhsRank == 2 && rhsRank == 1 && resultRank == 1) {
    SmallVector<Value, 1> dynamicDims;
    if (resultRs.isDimDynamic(0)) {
      auto lhsGetShape = builder.create<GetRankedShapeOp>(
          loc, RankedShapeType::get(lhsType.getShape(), builder.getContext()),
          dotOp.lhs());
      auto getMDim =
          builder.create<RankedDimOp>(loc, builder.getIndexType(), lhsGetShape,
                                      builder.getI64IntegerAttr(0));
      dynamicDims.push_back(getMDim);
    }
    return builder.create<MakeRankedShapeOp>(loc, resultRs, dynamicDims);
  } else if (lhsRank == 2 && rhsRank == 2 && resultRank == 2) {
    SmallVector<Value, 2> dynamicDims;
    if (resultRs.isDimDynamic(0)) {
      auto lhsGetShape = builder.create<GetRankedShapeOp>(
          loc, RankedShapeType::get(lhsType.getShape(), builder.getContext()),
          dotOp.lhs());
      auto getMDim =
          builder.create<RankedDimOp>(loc, builder.getIndexType(), lhsGetShape,
                                      builder.getI64IntegerAttr(0));
      dynamicDims.push_back(getMDim);
    }
    if (resultRs.isDimDynamic(1)) {
      auto rhsGetShape = builder.create<GetRankedShapeOp>(
          loc, RankedShapeType::get(rhsType.getShape(), builder.getContext()),
          dotOp.rhs());
      auto getNDim =
          builder.create<RankedDimOp>(loc, builder.getIndexType(), rhsGetShape,
                                      builder.getI64IntegerAttr(1));
      dynamicDims.push_back(getNDim);
    }
    return builder.create<MakeRankedShapeOp>(loc, resultRs, dynamicDims);
  } else {
    return nullptr;
  }
}

Value rewriteReduce(RankedShapeType resultShape, ReduceOp reduceOp,
                    OpBuilder &builder) {
  Location loc = reduceOp.getLoc();

  // Get a common operand shape.
  Value operandShape;
  SmallVector<Value, 4> operandShapes;
  RankedShapeType operandRs;
  for (auto operand : reduceOp.operands()) {
    auto shape = builder.create<GetRankedShapeOp>(loc, operand);
    operandRs = shape.getRankedShape();
    operandShapes.push_back(shape);
  }
  assert(!operandShapes.empty());
  if (operandShapes.size() == 1) {
    // Single operand.
    operandShape = operandShapes.front();
  } else {
    // Multiple operands must be compatible.
    operandShape =
        builder.create<CastCompatibleShapeOp>(loc, operandRs, operandShapes);
  }

  // Map reduction dims onto operand dimensions.
  SmallVector<bool, 4> isDimReduced;
  isDimReduced.resize(operandRs.getRank());
  for (auto apIntValue : reduceOp.dimensions().getIntValues()) {
    auto intValue = apIntValue.getZExtValue();
    assert(intValue < isDimReduced.size());
    isDimReduced[intValue] = true;
  }

  // Map operand -> result dynamic dims.
  assert(resultShape.getRank() ==
         (operandRs.getRank() - reduceOp.dimensions().getNumElements()));
  SmallVector<Value, 4> resultDims;
  for (unsigned operandDimIndex = 0, e = isDimReduced.size();
       operandDimIndex < e; ++operandDimIndex) {
    unsigned resultDimIndex = resultDims.size();
    // Skip reduced operand indices and non-dynamic result indices.
    if (isDimReduced[operandDimIndex] ||
        !resultShape.isDimDynamic(resultDimIndex))
      continue;
    resultDims.push_back(
        builder.create<RankedDimOp>(loc, operandShape, operandDimIndex));
  }

  return builder.create<MakeRankedShapeOp>(loc, resultShape, resultDims);
}

// NOTE: This op is an HLO interloper and is just here until a corresponding
// HLO is created. As such, it is included in this file even though not HLO
// currently.
Value rewriteShapexRankedBroadcastInDim(RankedShapeType resultShape,
                                        RankedBroadcastInDimOp bidOp,
                                        OpBuilder &builder) {
  if (!bidOp) return nullptr;
  return bidOp.result_shape();
}

Value rewriteTranspose(RankedShapeType resultShape, TransposeOp transposeOp,
                       OpBuilder &builder) {
  if (!transposeOp) return nullptr;
  auto loc = transposeOp.getLoc();

  auto operandType =
      transposeOp.operand().getType().dyn_cast<RankedTensorType>();
  if (!operandType) return nullptr;

  auto operandShapeValue = builder.create<GetRankedShapeOp>(
      loc, RankedShapeType::get(operandType.getShape(), builder.getContext()),
      transposeOp.operand());

  SmallVector<int64_t, 4> perm;
  for (auto permValue : transposeOp.permutation().getIntValues()) {
    perm.push_back(permValue.getSExtValue());
  }
  assert(perm.size() == resultShape.getRank());

  // Map the dynamic dims.
  SmallVector<Value, 4> dynamicDims;
  for (int64_t i = 0, e = resultShape.getRank(); i < e; ++i) {
    if (!resultShape.isDimDynamic(i)) continue;
    int64_t operandDim = perm[i];
    auto dimValue = builder.create<RankedDimOp>(
        loc, builder.getIndexType(), operandShapeValue,
        builder.getI64IntegerAttr(operandDim));
    dynamicDims.push_back(dimValue);
  }

  return builder.create<MakeRankedShapeOp>(loc, resultShape, dynamicDims);
}

}  // namespace

// Creates a custom op shape builder for XLA-HLO ops that are not otherwise
// supported through traits or other declarative means.
void populateXlaHloCustomOpShapeBuilder(CustomOpShapeBuilderList &builders) {
  auto &b = builders.make<CallbackCustomOpShapeBuilder>();
  // NOTE: Most of these *should not* be "custom ops". They should be coming
  // from declarative shape information, but that doesn't exist yet.
#define INSERT_EW_OP(OpTy) \
  b.insertOpRankedShapeBuilder<OpTy>(rewriteXlaBinaryElementwiseOpShape<OpTy>);
  INSERT_EW_OP(AddOp);
  INSERT_EW_OP(Atan2Op);
  INSERT_EW_OP(DivOp);
  INSERT_EW_OP(MaxOp);
  INSERT_EW_OP(MinOp);
  INSERT_EW_OP(MulOp);
  INSERT_EW_OP(PowOp);
  INSERT_EW_OP(RemOp);
  INSERT_EW_OP(ShiftLeftOp);
  INSERT_EW_OP(ShiftRightArithmeticOp);
  INSERT_EW_OP(ShiftRightLogicalOp);
  INSERT_EW_OP(SubOp);

  b.insertOpRankedShapeBuilder<DotOp>(rewriteXlaDotOpShape);
  b.insertOpRankedShapeBuilder<RankedBroadcastInDimOp>(
      rewriteShapexRankedBroadcastInDim);
  b.insertOpRankedShapeBuilder<ReduceOp>(rewriteReduce);
  b.insertOpRankedShapeBuilder<TransposeOp>(rewriteTranspose);
}

}  // namespace xla_hlo
}  // namespace mlir

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
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/Optional.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Value.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"

using llvm::None;
using llvm::Optional;
using llvm::SmallVector;

using namespace mlir::iree_compiler::Shape;

namespace mlir {
namespace mhlo {
namespace {

template <typename HloOp>
Value rewriteXlaBinaryElementwiseOpShape(RankedShapeType resultShape, HloOp op,
                                         OpBuilder &builder) {
  if (!op) return nullptr;
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
  for (const auto &apIntValue : reduceOp.dimensions().getIntValues()) {
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
  for (const auto &permValue : transposeOp.permutation().getIntValues()) {
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

// Returns a value of type `!shapex.ranked_shape` for the input value.
static Value getRankedShapeAsValue(Value v, OpBuilder &builder, Location loc) {
  assert(v.getType().isa<TensorType>());
  auto type = v.getType().dyn_cast<RankedTensorType>();
  if (!type) {
    return nullptr;
  }
  return builder.create<GetRankedShapeOp>(
      loc, RankedShapeType::get(type.getShape(), builder.getContext()), v);
}

// Returns a value representing the extent of dimension `dim`.
static Value getExtent(Value v, int64_t dim, OpBuilder &builder, Location loc) {
  return builder.create<RankedDimOp>(loc, v, dim);
}

Value rewriteDotGeneral(RankedShapeType resultShape, DotGeneralOp op,
                        OpBuilder &builder) {
  Location loc = op.getLoc();
  auto lhsShape = getRankedShapeAsValue(op.lhs(), builder, loc);
  auto rhsShape = getRankedShapeAsValue(op.rhs(), builder, loc);
  if (!lhsShape || !rhsShape) {
    return nullptr;
  }
  auto getFreeDims = [&](ArrayRef<int64_t> batchDims,
                         ArrayRef<int64_t> contractingDims, int64_t rank) {
    llvm::BitVector freeDims(rank, true);
    for (int64_t dim : batchDims) {
      freeDims.reset(dim);
    }
    for (int64_t dim : contractingDims) {
      freeDims.reset(dim);
    }
    SmallVector<int64_t, 4> result;
    for (auto bitIndex : freeDims.set_bits()) {
      result.push_back(bitIndex);
    }
    return result;
  };
  auto lhsRankedShape = lhsShape.getType().cast<RankedShapeType>();
  auto rhsRankedShape = rhsShape.getType().cast<RankedShapeType>();
  auto dotDimensions = op.dot_dimension_numbers();
  auto lhsFreeDims = getFreeDims(
      llvm::to_vector<4>(
          dotDimensions.lhs_batching_dimensions().getValues<int64_t>()),
      llvm::to_vector<4>(
          dotDimensions.lhs_contracting_dimensions().getValues<int64_t>()),
      lhsRankedShape.getRank());
  auto rhsFreeDims = getFreeDims(
      llvm::to_vector<4>(
          dotDimensions.rhs_batching_dimensions().getValues<int64_t>()),
      llvm::to_vector<4>(
          dotDimensions.rhs_contracting_dimensions().getValues<int64_t>()),
      rhsRankedShape.getRank());

  SmallVector<Value, 6> outputExtents;
  for (int64_t dim :
       dotDimensions.lhs_batching_dimensions().getValues<int64_t>()) {
    // TODO(silvasean): Add a version of MakeRankedShapeOp that takes
    // all dimensions. Having to constantly check if a dim is dynamic
    // upon construction is a waste of time, more testing burden, etc.
    // We can easily canonicalize to the more constrained one.
    if (lhsRankedShape.isDimDynamic(dim)) {
      outputExtents.push_back(getExtent(lhsShape, dim, builder, loc));
    }
  }
  for (int64_t dim : lhsFreeDims) {
    if (lhsRankedShape.isDimDynamic(dim)) {
      outputExtents.push_back(getExtent(lhsShape, dim, builder, loc));
    }
  }
  for (int64_t dim : rhsFreeDims) {
    if (rhsRankedShape.isDimDynamic(dim)) {
      outputExtents.push_back(getExtent(rhsShape, dim, builder, loc));
    }
  }
  return builder.create<MakeRankedShapeOp>(loc, resultShape, outputExtents);
}

Value rewriteDynamicReshape(RankedShapeType resultShape, DynamicReshapeOp op,
                            OpBuilder &builder) {
  return builder.create<FromExtentTensorOp>(op.getLoc(), resultShape,
                                            op.output_shape());
}

Value rewriteSelectOp(RankedShapeType resultShape, SelectOp selectOp,
                      OpBuilder &builder) {
  // TODO: broadcast case
  if (!selectOp) return nullptr;
  auto loc = selectOp.getLoc();

  auto trueType = selectOp.on_true().getType().dyn_cast<RankedTensorType>();
  auto falseType = selectOp.on_false().getType().dyn_cast<RankedTensorType>();
  auto predType = selectOp.pred().getType().dyn_cast<RankedTensorType>();
  if (!trueType || !falseType || !predType) return nullptr;

  assert(trueType.getRank() == resultShape.getRank() &&
         trueType.getRank() == falseType.getRank() &&
         trueType.getRank() == predType.getRank());

  auto operandShapeValue = builder.create<GetRankedShapeOp>(
      loc, RankedShapeType::get(trueType.getShape(), builder.getContext()),
      selectOp.on_true());

  // Map the dynamic dims.
  SmallVector<Value, 4> dynamicDims;
  for (int64_t i = 0, e = resultShape.getRank(); i < e; ++i) {
    if (!resultShape.isDimDynamic(i)) continue;
    auto dimValue = builder.create<RankedDimOp>(loc, builder.getIndexType(),
                                                operandShapeValue,
                                                builder.getI64IntegerAttr(i));
    dynamicDims.push_back(dimValue);
  }

  return builder.create<MakeRankedShapeOp>(loc, resultShape, dynamicDims);
}

Value rewriteTorchIndexSelect(RankedShapeType resultShape,
                              TorchIndexSelectOp torchIndexSelectOp,
                              OpBuilder &builder) {
  if (!torchIndexSelectOp) return nullptr;
  auto loc = torchIndexSelectOp.getLoc();

  int64_t resultShapeRank = resultShape.getRank();
  auto paramsType =
      torchIndexSelectOp.input().getType().dyn_cast<RankedTensorType>();
  auto indicesType =
      torchIndexSelectOp.index().getType().dyn_cast<RankedTensorType>();
  if (!paramsType || !indicesType) {
    return nullptr;
  }

  auto axis = torchIndexSelectOp.dim();
  auto batchDim = torchIndexSelectOp.batch_dims();
  int64_t paramsRank = paramsType.getRank();
  int64_t indicesRank = indicesType.getRank();

  std::vector<int64_t> shape(paramsType.getShape());
  int64_t axisValue = axis.getSExtValue();
  int64_t batchDimValue = batchDim.getSExtValue();

  // For neg axis values, we wrap around params,
  // e.g. axis = -1 => params[:-1]
  if (axisValue < 0) {
    axisValue += paramsRank;
  }
  if (batchDimValue < 0) {
    batchDimValue += indicesRank;
  }

  // params must be at least rank axis + 1
  if (paramsRank < axisValue + 1) {
    return nullptr;
  }

  auto paramsShapeValue = builder.create<GetRankedShapeOp>(
      loc, RankedShapeType::get(paramsType.getShape(), builder.getContext()),
      torchIndexSelectOp.input());
  auto indicesShapeValue = builder.create<GetRankedShapeOp>(
      loc, RankedShapeType::get(indicesType.getShape(), builder.getContext()),
      torchIndexSelectOp.index());

  SmallVector<Value, 4> dynamicDims;
#define GENERATE_RANKED_DIM_OP(value, index)                                   \
  do {                                                                         \
    auto dimValue = builder.create<RankedDimOp>(                               \
        loc, builder.getIndexType(), value, builder.getI64IntegerAttr(index)); \
    dynamicDims.push_back(dimValue);                                           \
  } while (0)

  if (indicesRank == 0) {
    // Scalar indices (output is rank(params) - 1).
    if (resultShapeRank != paramsRank - 1) {
      return nullptr;
    }

    // params.shape[:axis] + params.shape[axis+1:]
    for (int64_t i = 0; i < paramsRank; ++i) {
      if ((i == axisValue) || (i < axisValue && !resultShape.isDimDynamic(i)) ||
          (i > axisValue && !resultShape.isDimDynamic(i - 1)))
        continue;
      GENERATE_RANKED_DIM_OP(paramsShapeValue, i);
    }
  } else if (indicesRank == 1) {
    // Vector indices (output is rank(params)).
    // Copy indices.shape into params.shape[axis]
    if (resultShapeRank != paramsRank) {
      return nullptr;
    }

    // params.shape[:axis] + indices.shape[batch_dims:]
    //   + params.shape[indicesRank-batchDim+axisValue:]
    int resultShapeIndex = 0;
    // params.shape[:axis]
    for (int64_t i = 0; i < axisValue; ++i) {
      if (!resultShape.isDimDynamic(resultShapeIndex++)) continue;
      GENERATE_RANKED_DIM_OP(paramsShapeValue, i);
    }
    // indices.shape[:batchDim]
    for (int64_t i = batchDimValue;
         i < indicesRank && resultShapeIndex < resultShapeRank; ++i) {
      if (!resultShape.isDimDynamic(resultShapeIndex++)) continue;
      GENERATE_RANKED_DIM_OP(indicesShapeValue, i);
    }
    // params.shape[indicesRank-batchDim+axisValue:]
    // resultShapeIndex == indicesRank-batchDim+axisValue
    for (int64_t i = resultShapeIndex; i < resultShapeRank; ++i) {
      if (!resultShape.isDimDynamic(resultShapeIndex++)) continue;
      GENERATE_RANKED_DIM_OP(paramsShapeValue, i);
    }
  } else {
    // params.shape[:axis] + indices.shape[batch_dims:] + params.shape[axis +
    // 1:]
    // The expected rank is (paramsRank-1) + (indicesRank-batchDim)
    auto expectedRank = paramsRank - 1 + indicesRank - batchDimValue;
    if (resultShapeRank != expectedRank) {
      return nullptr;
    }

    int resultShapeIndex = 0;
    for (int64_t i = 0; i < axisValue; ++i) {
      if (!resultShape.isDimDynamic(resultShapeIndex++)) continue;
      GENERATE_RANKED_DIM_OP(paramsShapeValue, i);
    }

    for (int64_t i = batchDimValue; i < indicesRank; ++i) {
      if (!resultShape.isDimDynamic(resultShapeIndex++)) continue;
      GENERATE_RANKED_DIM_OP(indicesShapeValue, i);
    }

    for (int64_t i = axisValue + 1;
         i < paramsRank && resultShapeIndex < resultShapeRank; ++i) {
      if (!resultShape.isDimDynamic(resultShapeIndex++)) continue;
      GENERATE_RANKED_DIM_OP(paramsShapeValue, i);
    }
  }
#undef GENERATE_RANKED_DIM_OP

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
  INSERT_EW_OP(CompareOp);

  b.insertOpRankedShapeBuilder<SelectOp>(rewriteSelectOp);
  b.insertOpRankedShapeBuilder<DotOp>(rewriteXlaDotOpShape);
  b.insertOpRankedShapeBuilder<RankedBroadcastInDimOp>(
      rewriteShapexRankedBroadcastInDim);
  b.insertOpRankedShapeBuilder<ReduceOp>(rewriteReduce);
  b.insertOpRankedShapeBuilder<TransposeOp>(rewriteTranspose);
  b.insertOpRankedShapeBuilder<mhlo::DotGeneralOp>(rewriteDotGeneral);
  b.insertOpRankedShapeBuilder<mhlo::DynamicReshapeOp>(rewriteDynamicReshape);
  b.insertOpRankedShapeBuilder<mhlo::TorchIndexSelectOp>(
      rewriteTorchIndexSelect);
}

}  // namespace mhlo
}  // namespace mlir

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

#include "iree/compiler/Dialect/Shape/IR/Builders.h"

#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeTypes.h"

namespace mlir {
namespace iree_compiler {
namespace Shape {

Value buildCastInputsToResultShape(Location loc,
                                   RankedShapeType resultShapeType,
                                   ArrayRef<Value> inputs, OpBuilder &builder) {
  llvm::SmallVector<Value, 4> inputShapes;
  for (auto inputOperand : inputs) {
    auto inputOperandType = inputOperand.getType().dyn_cast<RankedTensorType>();
    RankedShapeType inputOperandShape = RankedShapeType::getChecked(
        inputOperandType.getShape(), resultShapeType.getDimType(),
        inputOperand.getLoc());
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
      auto dim = builder.create<RankedDimOp>(srcShape.getLoc(), srcShape,
                                             inputDimIndex);
      outputDynamicDims.push_back(dim);
    } else {
      // Append static source dim.
      outputAllDims.push_back(srcRsType.getStaticDim(inputDimIndex));
    }
  }

  auto dstRsType = RankedShapeType::get(outputAllDims, srcRsType.getDimType());
  if (outputDynamicDims.empty()) {
    return builder.create<ConstRankedShapeOp>(srcShape.getLoc(), dstRsType);
  } else {
    return builder.create<MakeRankedShapeOp>(srcShape.getLoc(), dstRsType,
                                             outputDynamicDims);
  }
}

}  // namespace Shape
}  // namespace iree_compiler
}  // namespace mlir

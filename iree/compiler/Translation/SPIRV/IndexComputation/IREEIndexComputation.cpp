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

//===- IREEIndexComputation.cpp --------------------------------*- C++//-*-===//
//
// Implementation of Index Propagation for IREE statements that are used in
// dispatch functions.
//
//===----------------------------------------------------------------------===//
#include "iree/compiler/Translation/SPIRV/IndexComputation/IREEIndexComputation.h"

namespace mlir {
namespace iree_compiler {

//===----------------------------------------------------------------------===//
// Utility functions for IREE Index propagation
//===----------------------------------------------------------------------===//

// TODO(ravishankarm) : Same logic can be used for ReturnOp. Move it there when
// IREE::StoreOutputOp and IREE::StoreReduceOp are deprecated.
static LogicalResult initIndexPropagation(Location loc, FuncOp funcOp,
                                          Value value) {
  SmallVector<int64_t, 4> valueShape;
  int64_t valueNumElements = 0;
  Type valueType = value.getType();
  if (auto valueShapedType = valueType.dyn_cast<ShapedType>()) {
    if (!valueShapedType.hasStaticShape()) {
      return emitError(loc, "can only handle tensor of static shape");
    }
    valueShape.append(valueShapedType.getShape().begin(),
                      valueShapedType.getShape().end());
    valueNumElements = valueShapedType.getNumElements();
  } else if (valueType.isIntOrFloat()) {
    valueShape.push_back(1);
    valueNumElements = 1;
  } else {
    return emitError(loc, "unhandled value type for index propagation");
  }

  SmallVector<int64_t, 3> launchSize;
  if (failed(getLegacyLaunchSize(funcOp, launchSize))) {
    return failure();
  }

  // The launch dimensions are [x, y, z] co-ordinates. The reverse of this is
  // used to determine the location of the tensor element computed by a
  // workitem. The choice is fairly arbitrary but is done to enable the common
  // case where consecutive workitems compute "logically" adjacent tensor
  // elements.
  Builder builder(funcOp.getContext());
  SmallVector<AffineExpr, 4> affineExprs;
  int64_t numElements = 1;
  for (size_t i = launchSize.size(); i > 0; --i) {
    // If launchSize along any dimension is 1, just use 0 for the index. This is
    // not just an optimization. If you have an output of type memref<f32> which
    // is lowered to !spv.ptr<!spv.struct<f32>, StorageBuffer> with launchSize
    // <1>, then spv.AccessChain requires the indices to be a constant.
    if (launchSize[i - 1] == 1) {
      affineExprs.push_back(builder.getAffineConstantExpr(0));
    } else {
      affineExprs.push_back(builder.getAffineDimExpr(i - 1));
    }
    numElements *= launchSize[i - 1];
  }
  auto launchMap = getAffineMap(funcOp, affineExprs);

  // The stored tensor can be a reshape of the launch dimension. It still
  // retains the requirement that each workitem is computing a single element
  // of the stored tensor.
  AffineMap valueMap;
  SmallVector<int64_t, 3> revLaunchSize(reverse(launchSize));
  if (numElements != valueNumElements ||
      failed(getReshapeOperandMap(funcOp, builder, launchMap, revLaunchSize,
                                  valueShape, valueMap))) {
    return emitError(
        loc,
        "unable to map from launch id to element to compute within a "
        "workitem");
  }
  return addNewIndexMapForValue(value, valueMap);
}

//===----------------------------------------------------------------------===//
// IREELoadInputOp
//===----------------------------------------------------------------------===//

LogicalResult IREELoadIndexPropagation::propagateIndexMap(
    Operation *operation, AffineMap resultIndex,
    SmallVectorImpl<AffineMap> &operandIndices) const {
  operandIndices.push_back(resultIndex);
  return success();
}

//===----------------------------------------------------------------------===//
// IREEStoreOutputOp
//===----------------------------------------------------------------------===//

LogicalResult IREEStoreIndexPropagation::propagateIndexMap(
    Operation *operation) const {
  auto storeOp = cast<IREE::StoreOutputOp>(operation);
  auto funcOp = operation->getParentOfType<FuncOp>();
  if (!funcOp) {
    return operation->emitError(
        "expected operation to be in dispatch function to get launch size");
  }
  return initIndexPropagation(storeOp.getLoc(), funcOp, storeOp.src());
}

//===----------------------------------------------------------------------===//
// IREEStoreReduceOp
//===----------------------------------------------------------------------===//

LogicalResult IREEStoreReduceIndexPropagation::propagateIndexMap(
    Operation *operation) const {
  auto storeReduceOp = cast<IREE::StoreReduceOp>(operation);
  auto funcOp = operation->getParentOfType<FuncOp>();
  if (!funcOp) {
    return operation->emitError(
        "expected operation to be in dispatch function to get launch size");
  }
  if (failed(initIndexPropagation(storeReduceOp.getLoc(), funcOp,
                                  storeReduceOp.src()))) {
    return failure();
  }

  // Set the index of the output as well based on which dimensions are reduced.
  SmallVector<AffineMap, 1> inputMap;
  getIndexMapsForValue(storeReduceOp.src(), inputMap);
  assert(inputMap.size() == 1);
  SmallVector<AffineExpr, 2> exprs;
  auto reductionDim =
      funcOp.getAttrOfType<IntegerAttr>("iree.executable.reduction.dimension")
          .getInt();
  for (auto dim : enumerate(inputMap[0].getResults())) {
    if (dim.index() == reductionDim) {
      continue;
    }
    exprs.push_back(dim.value());
  }
  if (exprs.empty()) {
    exprs.push_back(getAffineConstantExpr(0, operation->getContext()));
  }
  addNewIndexMapForValue(storeReduceOp.dst(), getAffineMap(funcOp, exprs));
  return success();
}

}  // namespace iree_compiler
}  // namespace mlir

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

//===- XLAIndexPropagation.cpp ---------------------------------*- C++//-*-===//
//
// For an IREE dispatch function in XLA-HLO dialect, compute the indices of all
// tensors needed to produce the value of the result tensors at a particlar
// index.
//
//===----------------------------------------------------------------------===//

#include "compiler/Translation/SPIRV/XLAIndexPropagation.h"

namespace mlir {
namespace iree_compiler {

//===----------------------------------------------------------------------===//
// BroadcastInDimOp
//===----------------------------------------------------------------------===//

LogicalResult XLABroadcastInDimOpIndexPropagation::propagateIndexMap(
    Operation *operation, AffineMap resultIndex,
    SmallVectorImpl<AffineMap> &indexMap) const {
  auto broadcastOp = cast<xla_hlo::BroadcastInDimOp>(operation);
  auto broadcastDim = broadcastOp.broadcast_dimensions();

  Builder builder(operation->getContext());
  if (!broadcastDim) {
    // This is a scalar. So all indices map to the same element.
    AffineMap scalarMap =
        AffineMap::get(resultIndex.getNumDims(), resultIndex.getNumSymbols(),
                       builder.getAffineConstantExpr(0));
    indexMap.push_back(scalarMap);
    return success();
  }

  // Handle non-scalar cases.
  auto dimensions = broadcastDim->getValues<int64_t>();
  SmallVector<AffineExpr, 4> exprs;
  for (auto resultExpr : enumerate(resultIndex.getResults())) {
    if (llvm::any_of(dimensions, [&resultExpr](int64_t dim) {
          return dim == resultExpr.index();
        })) {
      exprs.push_back(resultExpr.value());
    }
  }
  auto operandMap = AffineMap::get(resultIndex.getNumDims(),
                                   resultIndex.getNumSymbols(), exprs);
  indexMap.push_back(operandMap);
  return success();
}

//===----------------------------------------------------------------------===//
// BroadcastOp
//===----------------------------------------------------------------------===//

// For broadcast op, just drop the first N expressions of the resultIndex, where
// N is the number of elements in broadcast_sizes attribute.
LogicalResult XLABroadcastOpIndexPropagation::propagateIndexMap(
    Operation *operation, AffineMap resultIndex,
    SmallVectorImpl<AffineMap> &indexMap) const {
  auto broadcastOp = cast<xla_hlo::BroadcastOp>(operation);
  auto broadcastDim = broadcastOp.broadcast_sizes();

  SmallVector<AffineExpr, 4> exprs;
  for (auto i : llvm::seq<size_t>(
           broadcastDim.getType().getShape()[0],
           operation->getResult(0)->getType().cast<ShapedType>().getRank())) {
    exprs.push_back(resultIndex.getResult(i));
  }

  Builder builder(operation->getContext());
  if (exprs.empty()) {
    // The result is a scalar. Just add a constant expr 0.
    exprs.push_back(builder.getAffineConstantExpr(0));
  }
  auto operandMap = AffineMap::get(resultIndex.getNumDims(),
                                   resultIndex.getNumSymbols(), exprs);
  indexMap.push_back(operandMap);
  return success();
}

//===----------------------------------------------------------------------===//
// ReverseOp
//===----------------------------------------------------------------------===//

LogicalResult XLAReverseOpIndexPropagation::propagateIndexMap(
    Operation *op, AffineMap resultIndex,
    SmallVectorImpl<AffineMap> &indexMap) const {
  auto reverseOp = cast<xla_hlo::ReverseOp>(op);
  DenseSet<unsigned> dimensions;
  for (auto index : reverseOp.dimensions()) {
    dimensions.insert(index.getZExtValue());
  }
  return propagateIndexMapImpl(op, dimensions, resultIndex, indexMap);
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

LogicalResult XLATransposeOpIndexPropagation::propagateIndexMap(
    Operation *op, AffineMap resultIndex,
    SmallVectorImpl<AffineMap> &indexMap) const {
  auto transposeOp = cast<xla_hlo::TransposeOp>(op);
  // Compute the affine map that represents the permutation.
  SmallVector<unsigned, 4> permutation;
  for (auto index : transposeOp.permutation()) {
    permutation.push_back(index.getZExtValue());
  }
  return propagateIndexMapImpl(op, permutation, resultIndex, indexMap);
}

}  // namespace iree_compiler
}  // namespace mlir

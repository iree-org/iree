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
#include "third_party/mlir_edge/iree/compiler/Translation/SPIRV/XLAIndexPropagation.h"

namespace mlir {
namespace iree_compiler {

//===----------------------------------------------------------------------===//
// BroadcastInDimOp
//===----------------------------------------------------------------------===//

LogicalResult XLABroadcastInDimOpIndexPropagation::propagateIndexMap(
    Operation *op, AffineMap resultIndex,
    SmallVectorImpl<AffineMap> &operandIndices) const {
  auto broadcastOp = cast<xla_hlo::BroadcastInDimOp>(op);
  auto broadcastDim = broadcastOp.broadcast_dimensions();

  Builder builder(op->getContext());
  if (!broadcastDim) {
    // This is a scalar. So all indices map to the same element.
    AffineMap scalarMap = builder.getAffineMap(
        resultIndex.getNumDims(), resultIndex.getNumSymbols(),
        builder.getAffineConstantExpr(0));
    operandIndices.push_back(scalarMap);
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
  auto operandMap = builder.getAffineMap(resultIndex.getNumDims(),
                                         resultIndex.getNumSymbols(), exprs);
  operandIndices.push_back(operandMap);
  return success();
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

LogicalResult XLATransposeOpIndexPropagation::propagateIndexMap(
    Operation *op, AffineMap resultIndex,
    SmallVectorImpl<AffineMap> &operandIndices) const {
  auto transposeOp = cast<xla_hlo::TransposeOp>(op);
  // Compute the affine map that represents the permutation.
  SmallVector<unsigned, 4> permutation;
  for (auto index : transposeOp.permutation()) {
    permutation.push_back(index.getZExtValue());
  }
  return propagateIndexMapImpl(op, permutation, resultIndex, operandIndices);
}

}  // namespace iree_compiler
}  // namespace mlir

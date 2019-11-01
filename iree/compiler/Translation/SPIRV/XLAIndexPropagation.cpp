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

#include "iree/compiler/Translation/SPIRV/XLAIndexPropagation.h"

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
// ConcatenateOp
//===----------------------------------------------------------------------===//

LogicalResult XLAConcatenateOpIndexPropagation::propagateIndexMap(
    Operation *op, AffineMap resultIndex,
    SmallVectorImpl<AffineMap> &operandIndices) const {
  OpBuilder builder(op->getContext());
  auto concatenateOp = cast<xla_hlo::ConcatenateOp>(op);
  int append_dim = concatenateOp.dimension().getZExtValue();

  // For concatenate operation, the operands will be shifted along the given
  // dimension.
  int offset = 0;
  for (Value *operand : op->getOperands()) {
    auto operandType = operand->getType().cast<RankedTensorType>();
    int rank = operandType.getRank();
    SmallVector<AffineExpr, 4> exprs;
    for (int i = 0; i < rank; ++i) {
      AffineExpr e = builder.getAffineDimExpr(i);
      if (i == append_dim) e = e - offset;
      exprs.push_back(e);
    }
    offset += operandType.getDimSize(append_dim);
    AffineMap shiftedMap = AffineMap::get(resultIndex.getNumDims(),
                                          resultIndex.getNumSymbols(), exprs);
    AffineMap operandMap = shiftedMap.compose(resultIndex);
    operandIndices.push_back(operandMap);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// PadOp
//===----------------------------------------------------------------------===//

LogicalResult XLAPadOpIndexPropagation::propagateIndexMap(
    Operation *op, AffineMap resultIndex,
    SmallVectorImpl<AffineMap> &operandIndices) const {
  auto padOp = cast<xla_hlo::PadOp>(op);
  /// For pad operation, if result index at a particular dimension is d_i, then
  /// the input tensor index is computed as (d_i - edge_padding_low[i]) /
  /// (interior_padding[i]+1). Note that multiple indices of the output tensor
  /// could map to the same element of the padded tensor. The final lowering has
  /// to insert a condition to make sure that the padding_value is used when the
  /// index corresponds to a padded element.
  const auto &edge_padding_low = padOp.edge_padding_low();
  const auto &interior_padding = padOp.interior_padding();
  OpBuilder builder(op->getContext());

  /// Index for the tensor operand.
  SmallVector<AffineExpr, 4> exprs(
      padOp.operand()->getType().cast<RankedTensorType>().getRank());
  for (auto resultExpr : enumerate(resultIndex.getResults())) {
    auto i = resultExpr.index();
    int64_t padding_low = edge_padding_low.getValue<IntegerAttr>(i).getInt();
    int64_t padding_stride =
        interior_padding.getValue<IntegerAttr>(i).getInt() + 1;
    exprs[resultExpr.index()] =
        (resultExpr.value() - padding_low).floorDiv(padding_stride);
  }
  auto operandMap = AffineMap::get(resultIndex.getNumDims(),
                                   resultIndex.getNumSymbols(), exprs);
  operandIndices.push_back(operandMap);

  // Scalar operand is just mapped to a single element {0};
  auto scalarMap =
      AffineMap::get(resultIndex.getNumDims(), resultIndex.getNumSymbols(),
                     builder.getAffineConstantExpr(0));
  operandIndices.push_back(scalarMap);
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
// SliceOp
//===----------------------------------------------------------------------===//

LogicalResult XLASliceOpIndexPropagation::propagateIndexMap(
    Operation *op, AffineMap resultIndex,
    SmallVectorImpl<AffineMap> &indexMap) const {
  auto sliceOp = cast<xla_hlo::SliceOp>(op);
  SmallVector<unsigned, 4> start_indices, strides;
  for (auto index : sliceOp.start_indices()) {
    start_indices.push_back(index.getZExtValue());
  }
  for (auto stride : sliceOp.strides()) {
    strides.push_back(stride.getZExtValue());
  }
  return propagateIndexMapImpl(op, start_indices, strides, resultIndex,
                               indexMap);
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

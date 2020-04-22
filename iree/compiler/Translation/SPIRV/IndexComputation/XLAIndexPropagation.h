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

//===- XLAIndexPropagation.h -----------------------------------*- C++//-*-===//
//
// For an IREE dispatch function in XLA-HLO dialect, compute the indices of all
// tensors needed to produce the value of the result tensors at a particlar
// index.
//
//===----------------------------------------------------------------------===//
#ifndef IREE_COMPILER_TRANSLATION_SPIRV_INDEXCOMPUTATION_XLAPROPOGATION_H_
#define IREE_COMPILER_TRANSLATION_SPIRV_INDEXCOMPUTATION_XLAPROPOGATION_H_

#include "iree/compiler/Translation/SPIRV/IndexComputation/IndexComputation.h"
#include "iree/compiler/Translation/SPIRV/IndexComputation/IndexComputationAttribute.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Function.h"
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"

namespace mlir {
namespace iree_compiler {

class XLABroadcastInDimOpIndexPropagation final
    : public IndexPropagationOp<xla_hlo::BroadcastInDimOp> {
 public:
  using IndexPropagationOp<xla_hlo::BroadcastInDimOp>::IndexPropagationOp;

  LogicalResult propagateIndexMap(
      Operation *operation, AffineMap resultIndex,
      SmallVectorImpl<AffineMap> &operandIndices) const override;
};

// For broadcast op, just drop the first N expressions of the resultIndex, where
// N is the number of elements in broadcast_sizes attribute.
class XLABroadcastOpIndexPropagation final
    : public IndexPropagationOp<xla_hlo::BroadcastOp> {
 public:
  using IndexPropagationOp<xla_hlo::BroadcastOp>::IndexPropagationOp;

  LogicalResult propagateIndexMap(
      Operation *operation, AffineMap resultIndex,
      SmallVectorImpl<AffineMap> &operandIndices) const override;
};

/// For return ops, it is assumed that each thread is computing the value of one
/// element of the returned tensor.
template <typename OpTy>
class ReturnOpIndexPropagation : public IndexPropagationOp<OpTy> {
 public:
  using IndexPropagationOp<OpTy>::IndexPropagationOp;

  LogicalResult propagateIndexMap(Operation *operation) const override {
    if (operation->getNumOperands() != 1) {
      return operation->emitError("unhandled multiple return values");
    }
    auto returnValue = operation->getOperand(0);
    auto returnType = returnValue.getType().cast<RankedTensorType>();
    auto returnRank = returnType.getRank();
    if (returnRank > 3) {
      return operation->emitError("unhandled return tensor of dimension ")
             << returnType.getShape().size();
    }
    // Have as many dimensions as the rank of the input tensor. These symbols
    // map to GlobalInvocationID along the three dimensions.
    Builder builder(operation->getContext());
    SmallVector<AffineExpr, 4> affineExprs;
    for (size_t i = returnRank; i > 0; --i) {
      affineExprs.push_back(builder.getAffineDimExpr(i - 1));
    }
    return addNewIndexMapForValue(
        operation->getOperand(0),
        AffineMap::get(returnRank, 0, affineExprs, operation->getContext()));
  }
};

/// Index propogation for XLA ConcatenateOp.
class XLAConcatenateOpIndexPropagation final
    : public IndexPropagationOp<xla_hlo::ConcatenateOp> {
 public:
  using IndexPropagationOp<xla_hlo::ConcatenateOp>::IndexPropagationOp;

  LogicalResult propagateIndexMap(
      Operation *operation, AffineMap resultIndex,
      SmallVectorImpl<AffineMap> &operandIndices) const override;
};

class XLAGatherOpIndexPropagation final
    : public IndexPropagationOp<xla_hlo::GatherOp> {
 public:
  using IndexPropagationOp<xla_hlo::GatherOp>::IndexPropagationOp;

  LogicalResult propagateIndexMap(
      Operation *operation, AffineMap resultIndex,
      SmallVectorImpl<AffineMap> &operandIndices) const override;
};

/// Index propogation for XLA PadOp. If `d_i` is the index of the result
/// accessed in a workitem at dimension `i`, then set the index of the operand
/// needed as (`d_i` - `edge_padding_low`[i]) / (`interior_padding`[i] + 1)).
/// Note that multiple result indices map to a single index of the operand, with
/// only one of them needing the actual value from the operand. The rest are
/// indexing into a padding. This is handled during the lowering itself.
class XLAPadOpIndexPropagation final
    : public IndexPropagationOp<xla_hlo::PadOp> {
 public:
  using IndexPropagationOp<xla_hlo::PadOp>::IndexPropagationOp;

  LogicalResult propagateIndexMap(
      Operation *operation, AffineMap resultIndex,
      SmallVectorImpl<AffineMap> &operandIndices) const override;
};

/// Index propogation for XLA Reverse.
class XLAReverseOpIndexPropagation final
    : public ReverseOpIndexPropagation<xla_hlo::ReverseOp> {
 public:
  using ReverseOpIndexPropagation<
      xla_hlo::ReverseOp>::ReverseOpIndexPropagation;
  LogicalResult propagateIndexMap(
      Operation *op, AffineMap resultIndex,
      SmallVectorImpl<AffineMap> &indexMap) const override;
};

/// Index propogation for XLA Slice.
class XLASliceOpIndexPropagation final
    : public SliceOpIndexPropagation<xla_hlo::SliceOp> {
 public:
  using SliceOpIndexPropagation<xla_hlo::SliceOp>::SliceOpIndexPropagation;
  LogicalResult propagateIndexMap(
      Operation *op, AffineMap resultIndex,
      SmallVectorImpl<AffineMap> &indexMap) const override;
};

/// Index propogation for XLA Transpose.
class XLATransposeOpIndexPropagation final
    : public TransposeOpIndexPropagation<xla_hlo::TransposeOp> {
 public:
  using TransposeOpIndexPropagation<
      xla_hlo::TransposeOp>::TransposeOpIndexPropagation;
  LogicalResult propagateIndexMap(
      Operation *op, AffineMap resultIndex,
      SmallVectorImpl<AffineMap> &indexMap) const override;
};

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_TRANSLATION_SPIRV_INDEXCOMPUTATION_XLAPROPOGATION_H_

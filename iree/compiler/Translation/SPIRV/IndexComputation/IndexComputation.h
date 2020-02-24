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

//===- IndexComputation.h ---------------------------------------*- C++ -*-===//
//
// For an IREE dispatch function, compute the map from workitem ID to index of
// tensor computed within that workitem.
//
//===----------------------------------------------------------------------===//
#ifndef IREE_COMPILER_TRANSLATION_SPIRV_INDEXCOMPUTATION_INDEXCOMPUTATION_H
#define IREE_COMPILER_TRANSLATION_SPIRV_INDEXCOMPUTATION_INDEXCOMPUTATION_H

#include "iree/compiler/Translation/CodegenUtils/CodegenUtils.h"
#include "iree/compiler/Translation/SPIRV/IndexComputation/IndexComputationAttribute.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/StandardTypes.h"

namespace mlir {
namespace iree_compiler {

/// Base class used to construct a map from opname to the
/// computation that propagates the index map from result to
/// operands.
class IndexPropagation {
 public:
  virtual ~IndexPropagation() = default;
  virtual StringRef getOpName() const = 0;

  /// Propagates the index map from result to operands, i.e. given the index map
  /// that represents the element(s) of result(s) used within a thread, compute
  /// the indices of the operands needed. If overridden by inherited class the
  /// implementation should evaluate the indices of the operands needed for all
  /// indices of the result value needed. The default implementation only
  /// handles operations with a zero or single-return value.
  virtual LogicalResult propagateIndexMap(Operation *operation) const;

  /// Propagates the index map from result to operands for a given index of the
  /// result operand.
  virtual LogicalResult propagateIndexMap(
      Operation *operation, AffineMap resultIndex,
      SmallVectorImpl<AffineMap> &operandIndices) const {
    // By default do-nothing.
    return success();
  }
};

/// Base class that is templated on operation type to common
/// method to get the operation name.
template <typename OpTy>
class IndexPropagationOp : public IndexPropagation {
 public:
  using IndexPropagation::IndexPropagation;
  virtual ~IndexPropagationOp() = default;
  virtual StringRef getOpName() const { return OpTy::getOperationName(); }
};

// ===-------------------------------------------------------------------- ===//
// ExtractElementOp
// ===-------------------------------------------------------------------- ===//
class ExtractElementOpIndexPropagation final
    : public IndexPropagationOp<ExtractElementOp> {
 public:
  using IndexPropagationOp<ExtractElementOp>::IndexPropagationOp;

  LogicalResult propagateIndexMap(
      Operation *operation, AffineMap resultIndex,
      SmallVectorImpl<AffineMap> &operandIndices) const override;
};

// ===-------------------------------------------------------------------- ===//
// NoBroadCastPwOp
// ===-------------------------------------------------------------------- ===//

/// Propagates the index map from result to operands for
/// operations that are point-wise and the operands are not
/// implicitly broadcasted. Just copies the index maps of the
/// result to the operands.
template <typename OpTy>
class NoBroadcastPwOpIndexPropagation : public IndexPropagationOp<OpTy> {
 public:
  using IndexPropagationOp<OpTy>::IndexPropagationOp;

  LogicalResult propagateIndexMap(
      Operation *operation, AffineMap resultIndex,
      SmallVectorImpl<AffineMap> &operandIndices) const override {
    // All operands must have the same type.
    auto argRefType =
        operation->getOperand(0).getType().dyn_cast<RankedTensorType>();
    if (!argRefType) {
      return operation->emitError("expected operands to be of tensortype");
    }
    for (auto arg : operation->getOperands()) {
      auto argType = arg.getType().dyn_cast<RankedTensorType>();
      if (!argType || argType.getShape() != argRefType.getShape()) {
        return operation->emitError("expected operands to have same shape");
      }
      operandIndices.push_back(resultIndex);
    }
    return success();
  }
};

// ===-------------------------------------------------------------------- ===//
// ReshapeOp
// ===-------------------------------------------------------------------- ===//

/// Computes the index map representing the element of the operand accessed
/// given the index map of the result for a reshape operation.
LogicalResult getReshapeOperandMap(FuncOp funcOp, Builder &builder,
                                   AffineMap resultIndexMap,
                                   ArrayRef<int64_t> resultShapeRef,
                                   ArrayRef<int64_t> operandShapeRef,
                                   AffineMap &operandIndexMap);

/// Propagates the index map from result to operands of reshape type operations.
template <typename OpTy>
class ReshapeOpIndexPropagation final : public IndexPropagationOp<OpTy> {
 public:
  using IndexPropagationOp<OpTy>::IndexPropagationOp;

  LogicalResult propagateIndexMap(
      Operation *op, AffineMap resultIndex,
      SmallVectorImpl<AffineMap> &operandIndices) const override {
    Builder builder(op->getContext());
    auto resultType =
        op->getResult(0).getType().template dyn_cast<ShapedType>();
    auto operandType =
        op->getOperand(0).getType().template dyn_cast<ShapedType>();
    if (!resultType || !operandType) {
      return op->emitError("expected result and operand to be shaped types");
    }
    if (!resultType.hasStaticShape() || !operandType.hasStaticShape()) {
      return op->emitError(
          "unhandled non static shape of result/operands of reshape op");
    }
    if (resultType.getNumElements() != operandType.getNumElements()) {
      return op->emitError(
          "invalid reshape operation, mismatch in number of elements in "
          "operand and result");
    }
    // Check if the reshape is a adding or removing a dimension of size 1 (and
    // build the index for the operand accordingly).
    AffineMap operandIndexMap;
    if (failed(getReshapeOperandMap(op->getParentOfType<FuncOp>(), builder,
                                    resultIndex, resultType.getShape(),
                                    operandType.getShape(), operandIndexMap))) {
      return op->emitError("unhandled reshape index propagation");
    }
    operandIndices.push_back(operandIndexMap);
    return success();
  }
};

// ===-------------------------------------------------------------------- ===//
// ReverseOp
// ===-------------------------------------------------------------------- ===//

/// Propagates the index map from result to operands of reverse type operation.
/// See https://www.tensorflow.org/xla/operation_semantics#rev_reverse for more
/// details.
template <typename OpTy>
class ReverseOpIndexPropagation : public IndexPropagationOp<OpTy> {
 public:
  using IndexPropagationOp<OpTy>::IndexPropagationOp;
  virtual ~ReverseOpIndexPropagation() = default;

 protected:
  LogicalResult propagateIndexMapImpl(
      Operation *op, DenseSet<unsigned> dimensions, AffineMap resultIndex,
      SmallVectorImpl<AffineMap> &operandIndices) const {
    auto shaped_type = op->getOperand(0).getType().cast<ShapedType>();
    Builder builder(op->getContext());
    SmallVector<AffineExpr, 4> dimensionsExprs;
    for (unsigned index = 0; index < shaped_type.getRank(); ++index) {
      if (dimensions.count(index)) {
        auto reversed_index =
            shaped_type.getDimSize(index) - builder.getAffineDimExpr(index) - 1;
        dimensionsExprs.push_back(reversed_index);
      } else {
        dimensionsExprs.push_back(builder.getAffineDimExpr(index));
      }
    }
    auto dimensionsAffineMap = AffineMap::get(
        dimensionsExprs.size(), /*symbolCount=*/0, dimensionsExprs);

    // Compose the index map of the result with the index map for the
    // dimensions.
    auto operandMap = dimensionsAffineMap.compose(resultIndex);
    operandIndices.push_back(operandMap);
    return success();
  }
};

// ===-------------------------------------------------------------------- ===//
// SliceOp
// ===-------------------------------------------------------------------- ===//

/// Propagates the index map from result to operands of slice type operation.
/// See https://www.tensorflow.org/xla/operation_semantics#slice for more
/// details.
template <typename OpTy>
class SliceOpIndexPropagation : public IndexPropagationOp<OpTy> {
 public:
  using IndexPropagationOp<OpTy>::IndexPropagationOp;
  virtual ~SliceOpIndexPropagation() = default;

 protected:
  LogicalResult propagateIndexMapImpl(
      Operation *op, SmallVector<unsigned, 4> start_indices,
      SmallVector<unsigned, 4> strides, AffineMap resultIndex,
      SmallVectorImpl<AffineMap> &operandIndices) const {
    Builder builder(op->getContext());
    SmallVector<AffineExpr, 4> exprs;
    auto shaped_type = op->getOperand(0).getType().cast<ShapedType>();
    int rank = shaped_type.getRank();
    for (int i = 0; i < rank; ++i) {
      exprs.push_back(builder.getAffineDimExpr(i) * strides[i] +
                      start_indices[i]);
    }
    auto sliceAffineMap = AffineMap::get(rank, /*symbolCount=*/0, exprs);
    auto operandMap = sliceAffineMap.compose(resultIndex);
    operandIndices.push_back(operandMap);
    return success();
  }
};

/// Index map of the operand of a transpose op is obtained by composing the
/// affine map of the result with the affine map that represents the inverse of
/// the transpose permutation vector. The permutation vector must be supplied by
/// derived classes in the definition of the method `propagateIndexMap`.
template <typename OpTy>
class TransposeOpIndexPropagation : public IndexPropagationOp<OpTy> {
 public:
  using IndexPropagationOp<OpTy>::IndexPropagationOp;
  virtual ~TransposeOpIndexPropagation() = default;

 protected:
  LogicalResult propagateIndexMapImpl(
      Operation *op, ArrayRef<unsigned> permutation, AffineMap resultIndex,
      SmallVectorImpl<AffineMap> &operandIndices) const {
    Builder builder(op->getContext());
    SmallVector<AffineExpr, 4> permutationExprs;
    for (auto index : permutation) {
      permutationExprs.push_back(builder.getAffineDimExpr(index));
    }
    auto permutationAffineMap =
        AffineMap::get(permutationExprs.size(), 0, permutationExprs);
    // Compute the inverse of the permutation map.
    auto invPermutationMap = inversePermutation(permutationAffineMap);

    // Compose the index map of the result with the index map
    // for the inverse of the permutation.
    auto operandMap = invPermutationMap.compose(resultIndex);
    operandIndices.push_back(operandMap);
    return success();
  }
};

/// Maintains list of IndexPropagation objects that propagate the index
/// information from result to the operands of instructions.
template <typename... Ts>
class IndexPropagationList {
  using IndexPropagationListT =
      llvm::StringMap<std::unique_ptr<IndexPropagation>>;

 public:
  explicit IndexPropagationList() { insert(); }

  /// Performs the propagation.
  LogicalResult propagate(Region &region) const {
    if (region.getBlocks().size() != 1) {
      return emitError(
          region.getLoc(),
          "unimplemented handling multiple blocks within a region");
    }

    // Set the attribute for the number of launch dims.
    auto funcOp = region.getParentOfType<FuncOp>();
    SmallVector<int64_t, 3> launchSize;
    if (failed(getLaunchSize(funcOp, launchSize))) {
      return emitError(region.getLoc(),
                       "expected region of index propagation to be in dispatch "
                       "function to get launch size");
    }
    setNumLaunchDims(funcOp, launchSize.size());
    // TODO(ravishankarm) : Need to process blocks in reverse topological order.
    for (auto it = region.rbegin(), ie = region.rend(); it != ie; ++it) {
      for (auto jt = it->rbegin(), je = it->rend(); jt != je; ++jt) {
        auto &op = *jt;
        auto opName = op.getName().getStringRef();
        if (!indexPropagationList.count(opName)) {
          return op.emitError() << "unhandled index propagation for " << opName;
        }
        auto propagate = indexPropagationList.find(opName);
        if (failed(propagate->getValue()->propagateIndexMap(&op))) {
          return failure();
        }
      }
    }
    return success();
  }

 private:
  /// Builds the list by unpacking the parameter pack.
  void insert() {
    std::vector<std::unique_ptr<IndexPropagation>> objs;
    // TODO(ravishankarm) : This uses the fold logic from
    // mlir/IR/PatternMatch.h. There might be a simpler/more efficient way of
    // implementing this.
    using dummy = int[];
    (void)dummy{0, (objs.emplace_back(std::make_unique<Ts>()), 0)...};
    for (auto &elem : objs) {
      StringRef opName = elem->getOpName();
      indexPropagationList.try_emplace(opName, std::move(elem));
    }
  }

  /// List of methods for propagation indexed using opname.
  IndexPropagationListT indexPropagationList;
};

}  // namespace iree_compiler
}  // namespace mlir
#endif  // IREE_COMPILER_TRANSLATION_SPIRV_INDEXCOMPUTATION_INDEXCOMPUTATION_H

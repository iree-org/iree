// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LLVMGPURESOLVEVECTORMASKINGPASS
#include "iree/compiler/Codegen/LLVMGPU/Passes.h.inc"

namespace {

// Validates that maskOp has no passthru and has a vector mask type.
// Returns the mask value on success, failure otherwise.
static FailureOr<Value> validateMaskOp(vector::MaskOp maskOp,
                                       PatternRewriter &rewriter) {
  if (maskOp.getPassthru()) {
    return rewriter.notifyMatchFailure(maskOp, "passthru not supported");
  }

  Value mask = maskOp.getMask();
  if (!isa<VectorType>(mask.getType())) {
    return failure();
  }

  return mask;
}

// Extracts the iterator-dimension positions referenced by an indexing map.
// Returns failure if any result expression is not a simple AffineDimExpr.
static FailureOr<SmallVector<int64_t>>
getIteratorDimsForMap(AffineMap indexingMap) {
  SmallVector<int64_t> dims;
  for (int64_t i = 0; i < indexingMap.getNumResults(); ++i) {
    auto dimExpr = dyn_cast<AffineDimExpr>(indexingMap.getResult(i));
    if (!dimExpr) {
      return failure();
    }
    dims.push_back(dimExpr.getPosition());
  }
  return dims;
}

// Projects an iterator-space mask to an operand's space using the indexing map.
//
// For create_mask/constant_mask, produces a new mask op with remapped bounds
// (cleaner IR). For other masks, falls back to transpose+extract projection
// (assumes the mask is rectangular, i.e., uniform along dropped dimensions).
static FailureOr<Value> projectMaskToOperand(ImplicitLocOpBuilder &builder,
                                             Value iterMask,
                                             VectorType operandType,
                                             AffineMap indexingMap) {
  auto iterDims = getIteratorDimsForMap(indexingMap);
  if (failed(iterDims)) {
    return failure();
  }

  auto maskType = VectorType::get(operandType.getShape(), builder.getI1Type());

  // Specialized path for create_mask: remap dynamic bounds.
  if (auto createMask = iterMask.getDefiningOp<vector::CreateMaskOp>()) {
    SmallVector<Value> operandBounds;
    for (int64_t dim : *iterDims) {
      operandBounds.push_back(createMask.getOperand(dim));
    }
    return vector::CreateMaskOp::create(builder, maskType, operandBounds)
        .getResult();
  }

  // Specialized path for constant_mask: remap static dim sizes.
  if (auto constantMask = iterMask.getDefiningOp<vector::ConstantMaskOp>()) {
    ArrayRef<int64_t> allDimSizes = constantMask.getMaskDimSizes();
    SmallVector<int64_t> operandDimSizes;
    for (int64_t dim : *iterDims) {
      operandDimSizes.push_back(allDimSizes[dim]);
    }
    return vector::ConstantMaskOp::create(builder, maskType, operandDimSizes)
        .getResult();
  }

  // Generic fallback: transpose unused dims to front and extract at 0.
  int64_t numDims = indexingMap.getNumDims();
  llvm::SmallSetVector<int64_t, 4> usedDims;
  usedDims.insert(iterDims->begin(), iterDims->end());

  SmallVector<int64_t> transposePerm;
  for (int64_t i = 0; i < numDims; ++i) {
    if (!usedDims.contains(i)) {
      transposePerm.push_back(i);
    }
  }
  int64_t numSliced = transposePerm.size();
  transposePerm.append(usedDims.begin(), usedDims.end());
  Value transposed =
      vector::TransposeOp::create(builder, iterMask, transposePerm);
  SmallVector<int64_t> extractPos(numSliced, 0);
  return vector::ExtractOp::create(builder, transposed, extractPos).getResult();
}

// CRTP base for patterns that unwrap a vector.mask by replacing the masked
// operation with an unmasked equivalent that operates on identity-padded
// operands.
//
// ConcreteType must implement:
//   FailureOr<InnerOpType> createUnmaskedReplacement(
//       ImplicitLocOpBuilder &builder, InnerOpType innerOp,
//       Value mask, vector::CombiningKind kind) const;
//
template <typename ConcreteType, typename InnerOpType>
struct UnwrapMaskedOpPattern : OpRewritePattern<vector::MaskOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::MaskOp maskOp,
                                PatternRewriter &rewriter) const override {
    auto innerOp = dyn_cast<InnerOpType>(maskOp.getMaskableOp());
    if (!innerOp) {
      return failure();
    }

    auto mask = validateMaskOp(maskOp, rewriter);
    if (failed(mask)) {
      return failure();
    }

    ImplicitLocOpBuilder builder(maskOp.getLoc(), rewriter);
    vector::CombiningKind kind = innerOp.getKind();

    FailureOr<InnerOpType> maybeNewOp =
        static_cast<const ConcreteType *>(this)->createUnmaskedReplacement(
            builder, innerOp, *mask, kind);
    if (failed(maybeNewOp)) {
      return maybeNewOp;
    }

    InnerOpType newOp = *maybeNewOp;
    // Preserve discardable attributes, e.g., MMA kind on contract.
    newOp->setDiscardableAttrs(innerOp->getDiscardableAttrDictionary());
    rewriter.replaceOp(maskOp, newOp);
    return success();
  }
};

struct UnwrapMaskedContractPattern final
    : UnwrapMaskedOpPattern<UnwrapMaskedContractPattern,
                            vector::ContractionOp> {
  using UnwrapMaskedOpPattern::UnwrapMaskedOpPattern;

  FailureOr<vector::ContractionOp>
  createUnmaskedReplacement(ImplicitLocOpBuilder &builder,
                            vector::ContractionOp contractOp, Value mask,
                            vector::CombiningKind kind) const {
    auto lhsType = contractOp.getLhsType();
    auto rhsType = contractOp.getRhsType();

    SmallVector<AffineMap> indexingMaps = contractOp.getIndexingMapsArray();
    assert(indexingMaps.size() == 3);

    Value identityLhs =
        getCombiningIdentityValue(builder.getLoc(), builder, kind, lhsType);
    Value identityRhs =
        getCombiningIdentityValue(builder.getLoc(), builder, kind, rhsType);

    AffineMap lhsMap = indexingMaps[0];
    AffineMap rhsMap = indexingMaps[1];

    auto lhsMask = projectMaskToOperand(builder, mask, lhsType, lhsMap);
    if (failed(lhsMask)) {
      return failure();
    }
    auto rhsMask = projectMaskToOperand(builder, mask, rhsType, rhsMap);
    if (failed(rhsMask)) {
      return failure();
    }

    Value lhsMasked = arith::SelectOp::create(builder, *lhsMask,
                                              contractOp.getLhs(), identityLhs);
    Value rhsMasked = arith::SelectOp::create(builder, *rhsMask,
                                              contractOp.getRhs(), identityRhs);

    return vector::ContractionOp::create(
        builder, lhsMasked, rhsMasked, contractOp.getAcc(),
        contractOp.getIndexingMaps(), contractOp.getIteratorTypes(), kind);
  }
};

struct UnwrapMaskedMultiReductionPattern final
    : UnwrapMaskedOpPattern<UnwrapMaskedMultiReductionPattern,
                            vector::MultiDimReductionOp> {
  using UnwrapMaskedOpPattern::UnwrapMaskedOpPattern;

  FailureOr<vector::MultiDimReductionOp>
  createUnmaskedReplacement(ImplicitLocOpBuilder &builder,
                            vector::MultiDimReductionOp multiReduceOp,
                            Value mask, vector::CombiningKind kind) const {
    auto sourceType = multiReduceOp.getSourceVectorType();
    auto maskType = cast<VectorType>(mask.getType());
    if (maskType.getShape() != sourceType.getShape()) {
      return failure();
    }
    Value identitySource =
        getCombiningIdentityValue(builder.getLoc(), builder, kind, sourceType);
    Value sourceMasked = arith::SelectOp::create(
        builder, mask, multiReduceOp.getSource(), identitySource);
    return vector::MultiDimReductionOp::create(
        builder, kind, sourceMasked, multiReduceOp.getAcc(),
        multiReduceOp.getReductionDims());
  }
};

struct UnwrapMaskedReductionPattern final
    : UnwrapMaskedOpPattern<UnwrapMaskedReductionPattern, vector::ReductionOp> {
  using UnwrapMaskedOpPattern::UnwrapMaskedOpPattern;

  FailureOr<vector::ReductionOp>
  createUnmaskedReplacement(ImplicitLocOpBuilder &builder,
                            vector::ReductionOp reductionOp, Value mask,
                            vector::CombiningKind kind) const {
    auto sourceType = cast<VectorType>(reductionOp.getVector().getType());
    auto maskType = cast<VectorType>(mask.getType());
    if (maskType.getShape() != sourceType.getShape()) {
      return failure();
    }
    Value identitySource =
        getCombiningIdentityValue(builder.getLoc(), builder, kind, sourceType);
    Value sourceMasked = arith::SelectOp::create(
        builder, mask, reductionOp.getVector(), identitySource);
    if (reductionOp.getAcc()) {
      return vector::ReductionOp::create(builder, kind, sourceMasked,
                                         reductionOp.getAcc());
    }
    return vector::ReductionOp::create(builder, kind, sourceMasked);
  }
};

/// Decomposes create_mask/constant_mask operations into a combination of
/// vector.step and comparison with the mask bounds.
///
/// For each dimension, we create a vector.step containing all indices, then
/// broadcast the bound and compare (slt) to produce a per-dimension boolean
/// mask. For multi-dimensional masks, per-dimension masks are broadcast to
/// the full shape and combined with AND.
///
/// Example: create_mask %m, %n : vector<4x8xi1>
///   dim 0: step [0,1,2,3] < broadcast(%m) → <4xi1>, broadcast to <4x8xi1>
///   dim 1: step [0..7]    < broadcast(%n) → <8xi1>, broadcast to <4x8xi1>
///   result: dim0_mask AND dim1_mask
///
/// Assuming [%m, %n] = [3, 6] this would mean concretely:
/// For the first dimension, vector.step gives `[0, 1, 2, 3]`, broadcasting the
/// bound gives [3, 3, 3, 3], so the comparison results in `[1, 1, 1, 0]`. At
/// this point, the mask for the dimension 0 is still `<4xi1>`, but needs to
/// match the original vector shape. To that end, it is first broadcasted to
/// <8x4xi1>, as broadcast happens in the trailing dimension. This results in:
/// [[1, 1, 1, 0],
///  [1, 1, 1, 0],
///  [1, 1, 1, 0],
///  [1, 1, 1, 0],
///  [1, 1, 1, 0],
///  [1, 1, 1, 0],
///  [1, 1, 1, 0],
///  [1, 1, 1, 0]]
///
/// To get the original shape, it is then transposed, giving:
///  [[1, 1, 1, 1, 1, 1, 1, 1],
///   [1, 1, 1, 1, 1, 1, 1, 1],
///   [1, 1, 1, 1, 1, 1, 1, 1],
///   [0, 0, 0, 0, 0, 0, 0, 0]]
///
/// For the second dimension, the same process is repeated, resulting a <8xi1>
/// mask after comparison, with content [1, 1, 1, 1, 1, 1, 0, 0]. This is
/// broadcasted to the original shape and, as this is already the trailing
/// dimension of the original layout, doesn't need to be transposed.
/// This yields:
/// [[1, 1, 1, 1, 1, 1, 0, 0],
///  [1, 1, 1, 1, 1, 1, 0, 0],
///  [1, 1, 1, 1, 1, 1, 0, 0],
///  [1, 1, 1, 1, 1, 1, 0, 0]]
///
/// As a final step, the two masks are now AND-combined, yielding the expected
/// final result:
///  [[1, 1, 1, 1, 1, 1, 0, 0],
///   [1, 1, 1, 1, 1, 1, 0, 0],
///   [1, 1, 1, 1, 1, 1, 0, 0],
///   [0, 0, 0, 0, 0, 0, 0, 0]]
///
template <typename MaskOpTy>
struct DecomposeMaskOp final : OpRewritePattern<MaskOpTy> {
  using OpRewritePattern<MaskOpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(MaskOpTy maskOp,
                                PatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder builder(maskOp.getLoc(), rewriter);

    VectorType maskType = maskOp.getType();
    int64_t rank = maskType.getRank();

    SmallVector<Value> upperBounds = getUpperBounds(builder, maskOp);

    SmallVector<Value> dimMasks;
    for (int64_t dim = 0; dim < rank; ++dim) {
      int64_t dimSize = maskType.getDimSize(dim);

      // If bound is a constant equal to the dim size, all elements along this
      // dimension are valid and we can skip.
      APInt constBound;
      if (matchPattern(upperBounds[dim], m_ConstantInt(&constBound)) &&
          constBound.getSExtValue() == dimSize) {
        continue;
      }

      auto stepType = VectorType::get({dimSize}, builder.getIndexType());
      auto step = vector::StepOp::create(builder, stepType);
      auto bcastBound =
          vector::BroadcastOp::create(builder, stepType, upperBounds[dim]);
      auto cmp = arith::CmpIOp::create(builder, arith::CmpIPredicate::slt, step,
                                       bcastBound);

      Value fullMask = broadcastDimMaskToFullShape(builder, cmp, dim, maskType);
      dimMasks.push_back(fullMask);
    }

    // If all dimensions were skipped (all bounds equal dim sizes), emit
    // an all-true constant.
    if (dimMasks.empty()) {
      auto allTrue = arith::ConstantOp::create(
          builder, DenseElementsAttr::get(maskType, builder.getBoolAttr(true)));
      rewriter.replaceOp(maskOp, allTrue);
      return success();
    }

    // Combine per-dimension masks with AND.
    Value combined = dimMasks[0];
    for (size_t i = 1; i < dimMasks.size(); ++i) {
      combined = arith::AndIOp::create(builder, combined, dimMasks[i]);
    }
    rewriter.replaceOp(maskOp, combined);
    return success();
  }

private:
  /// Broadcast a rank-1 per-dimension mask to the full mask shape.
  /// For the trailing dimension, this is a simple rank-extension broadcast.
  /// For non-trailing dimensions, we broadcast with the target dim at trailing
  /// position and then transpose it back.
  static Value broadcastDimMaskToFullShape(ImplicitLocOpBuilder &builder,
                                           Value dimMask, int64_t dim,
                                           VectorType maskType) {
    int64_t rank = maskType.getRank();
    if (dim == rank - 1) {
      return vector::BroadcastOp::create(builder, maskType, dimMask);
    }

    // Non-trailing dim: broadcast to shape with dim at trailing position,
    // then transpose back.
    SmallVector<int64_t> bcastShape;
    SmallVector<int64_t> permToTrailing;
    for (int64_t i = 0; i < rank; ++i) {
      if (i != dim) {
        bcastShape.push_back(maskType.getDimSize(i));
        permToTrailing.push_back(i);
      }
    }
    bcastShape.push_back(maskType.getDimSize(dim));
    permToTrailing.push_back(dim);

    auto bcastType = VectorType::get(bcastShape, builder.getI1Type());
    auto bcast = vector::BroadcastOp::create(builder, bcastType, dimMask);

    SmallVector<int64_t> invPerm = invertPermutationVector(permToTrailing);
    return vector::TransposeOp::create(builder, bcast, invPerm);
  }

  static SmallVector<Value> getUpperBounds(ImplicitLocOpBuilder &builder,
                                           MaskOpTy maskOp) {
    if constexpr (std::is_same_v<MaskOpTy, vector::CreateMaskOp>) {
      return SmallVector<Value>(maskOp.getOperands());
    } else {
      SmallVector<Value> bounds;
      for (int64_t size : maskOp.getMaskDimSizes()) {
        bounds.push_back(arith::ConstantIndexOp::create(builder, size));
      }
      return bounds;
    }
  }
};

struct LLVMGPUResolveVectorMaskingPass final
    : impl::LLVMGPUResolveVectorMaskingPassBase<
          LLVMGPUResolveVectorMaskingPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<UnwrapMaskedContractPattern, UnwrapMaskedMultiReductionPattern,
                 UnwrapMaskedReductionPattern,
                 DecomposeMaskOp<vector::CreateMaskOp>,
                 DecomposeMaskOp<vector::ConstantMaskOp>>(context);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler

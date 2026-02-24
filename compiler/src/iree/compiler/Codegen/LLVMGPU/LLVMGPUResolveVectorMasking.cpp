// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

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

// Projects an iterator-space mask to an operand's space by transposing
// dimensions not present in the indexing map to the front and extracting at
// position 0. This assumes the mask is rectangular, i.e., uniform along the
// dropped dimensions.
static FailureOr<Value> projectMaskToOperand(ImplicitLocOpBuilder &builder,
                                             Value iterMask,
                                             AffineMap indexingMap) {
  int64_t numDims = indexingMap.getNumDims();
  llvm::SmallSetVector<int64_t, 4> usedDims;
  for (int64_t i = 0; i < indexingMap.getNumResults(); ++i) {
    auto dimExpr = dyn_cast<AffineDimExpr>(indexingMap.getResult(i));
    if (!dimExpr) {
      return failure();
    }
    usedDims.insert(dimExpr.getPosition());
  }

  // Sliced dims in ascending order, followed by the used dims in map order.
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
  Value result = vector::ExtractOp::create(builder, transposed, extractPos);
  return result;
}

// Creates a mask for an operand by projecting the iterator-space mask to the
// operand's space using the indexing map. Specialized for create_mask: produces
// a new create_mask with remapped bounds (cleaner IR than transpose+extract).
static FailureOr<Value> projectMaskForOperand(ImplicitLocOpBuilder &builder,
                                              vector::CreateMaskOp createMask,
                                              VectorType operandType,
                                              AffineMap indexingMap) {
  // The mask covers the full iterator domain. For each operand dimension,
  // extract the corresponding bound from the create_mask operation.
  SmallVector<Value> operandBounds;
  for (int64_t i = 0; i < indexingMap.getNumResults(); ++i) {
    auto dimExpr = dyn_cast<AffineDimExpr>(indexingMap.getResult(i));
    if (!dimExpr) {
      return failure();
    }
    operandBounds.push_back(createMask.getOperand(dimExpr.getPosition()));
  }

  // Create new mask for this operand
  auto maskType = VectorType::get(operandType.getShape(), builder.getI1Type());
  auto maskOp = vector::CreateMaskOp::create(builder, maskType, operandBounds);
  return maskOp.getResult();
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

    // Try create_mask path first (produces cleaner IR with new create_mask
    // ops). Fall back to general transpose+extract projection.
    Value lhsMaskVal, rhsMaskVal;
    auto createMask = mask.getDefiningOp<vector::CreateMaskOp>();
    if (createMask) {
      auto lhsMask =
          projectMaskForOperand(builder, createMask, lhsType, lhsMap);
      if (failed(lhsMask)) {
        return failure();
      }
      auto rhsMask =
          projectMaskForOperand(builder, createMask, rhsType, rhsMap);
      if (failed(rhsMask)) {
        return failure();
      }
      lhsMaskVal = *lhsMask;
      rhsMaskVal = *rhsMask;
    } else {
      auto lhsMask = projectMaskToOperand(builder, mask, lhsMap);
      if (failed(lhsMask)) {
        return failure();
      }
      auto rhsMask = projectMaskToOperand(builder, mask, rhsMap);
      if (failed(rhsMask)) {
        return failure();
      }
      lhsMaskVal = *lhsMask;
      rhsMaskVal = *rhsMask;
    }

    Value lhsMasked = arith::SelectOp::create(builder, lhsMaskVal,
                                              contractOp.getLhs(), identityLhs);
    Value rhsMasked = arith::SelectOp::create(builder, rhsMaskVal,
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

struct LLVMGPUResolveVectorMaskingPass final
    : impl::LLVMGPUResolveVectorMaskingPassBase<
          LLVMGPUResolveVectorMaskingPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<UnwrapMaskedContractPattern, UnwrapMaskedMultiReductionPattern,
                 UnwrapMaskedReductionPattern>(context);
    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};

} // namespace

} // namespace mlir::iree_compiler

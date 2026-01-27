// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "iree/compiler/Utils/ShapeUtils.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

namespace mlir::iree_compiler::IREE::TensorExt {

//===----------------------------------------------------------------------===//
// iree_tensor_ext.bitcast
//===----------------------------------------------------------------------===//

namespace {
struct ReplaceBitCastIfTensorOperandEmpty final : OpRewritePattern<BitCastOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(BitCastOp op,
                                PatternRewriter &rewriter) const override {
    auto emptyOp =
        dyn_cast_if_present<tensor::EmptyOp>(op.getSource().getDefiningOp());
    if (!emptyOp) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<tensor::EmptyOp>(op, op.getResult().getType(),
                                                 op.getResultDims());
    return success();
  }
};

struct BitCastOfTensorCastStaticInfo final : OpRewritePattern<BitCastOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(BitCastOp bitcastOp,
                                PatternRewriter &rewriter) const final {
    auto tensorCastOp = bitcastOp.getSource().getDefiningOp<tensor::CastOp>();
    if (!tensorCastOp) {
      return failure();
    }
    auto tensorCastSrcType =
        dyn_cast<RankedTensorType>(tensorCastOp.getOperand().getType());
    if (!tensorCastSrcType) {
      return failure();
    }

    // Ranked tensor casting can only change the amount of static information.
    // Only fold if the amount of static information is increasing.
    if (!tensor::canFoldIntoConsumerOp(tensorCastOp)) {
      return failure();
    }

    // All dims except last must match (last dim can differ due to element type
    // size change). This check also filters out rank mismatches.
    if (!compareMixedShapesEqualExceptLast(
            bitcastOp.getSource().getType(), bitcastOp.getSourceDims(),
            bitcastOp.getResult().getType(), bitcastOp.getResultDims())) {
      return failure();
    }

    RankedTensorType intermediateTensorType = bitcastOp.getSource().getType();
    RankedTensorType resTensorType = bitcastOp.getResult().getType();

    MLIRContext *ctx = bitcastOp.getContext();
    SmallVector<OpFoldResult> resultMixed = getMixedValues(
        resTensorType.getShape(), bitcastOp.getResultDims(), ctx);

    // Build new result shape by propagating static information.
    SmallVector<OpFoldResult> newResultMixed;
    int64_t rank = resTensorType.getRank();
    for (int64_t i = 0; i < rank - 1; ++i) {
      // Try cast source static info first.
      int64_t castSize = tensorCastSrcType.getShape()[i];
      if (!ShapedType::isDynamic(castSize)) {
        newResultMixed.push_back(rewriter.getIndexAttr(castSize));
        continue;
      }
      // Try result constant.
      OpFoldResult resOfr = resultMixed[i];
      if (std::optional<int64_t> resConst = getConstantIntValue(resOfr)) {
        newResultMixed.push_back(rewriter.getIndexAttr(*resConst));
        continue;
      }
      // Keep dynamic.
      newResultMixed.push_back(resOfr);
    }

    // Last dim: only use result info.
    OpFoldResult lastResOfr = resultMixed.back();
    if (std::optional<int64_t> resConst = getConstantIntValue(lastResOfr)) {
      newResultMixed.push_back(rewriter.getIndexAttr(*resConst));
    } else {
      newResultMixed.push_back(lastResOfr);
    }

    // Decompose into static shape and dynamic values.
    auto [newResultSizes, newDynDestDims] =
        decomposeMixedValues(newResultMixed);

    // Build new source dynamic dims from the cast source type.
    SmallVector<Value> newDynSrcDims;
    int64_t intermediateDynIdx = 0;
    for (int64_t i = 0; i < tensorCastSrcType.getRank(); ++i) {
      if (!intermediateTensorType.isDynamicDim(i)) {
        continue;
      }
      if (tensorCastSrcType.isDynamicDim(i)) {
        newDynSrcDims.push_back(bitcastOp.getSourceDims()[intermediateDynIdx]);
      }
      ++intermediateDynIdx;
    }

    auto newType =
        RankedTensorType::get(newResultSizes, resTensorType.getElementType(),
                              resTensorType.getEncoding());
    Value newBitcast = BitCastOp::create(rewriter, bitcastOp.getLoc(), newType,
                                         tensorCastOp.getOperand(),
                                         newDynSrcDims, newDynDestDims);
    // We create a new cast to continue propagating static information.
    rewriter.replaceOpWithNewOp<tensor::CastOp>(bitcastOp, resTensorType,
                                                newBitcast);

    return success();
  }
};

/// Replaces chains of two bitcast operations by a single bitcast operation.
/// bitcast(bitcast(x : A -> B) : B -> C) -> bitcast(x : A -> C).
struct ChainedBitCast final : OpRewritePattern<BitCastOp> {
  using OpRewritePattern<BitCastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BitCastOp bitcastOp,
                                PatternRewriter &rewriter) const override {
    auto producerBitcast = bitcastOp.getSource().getDefiningOp<BitCastOp>();
    if (!producerBitcast) {
      return failure();
    }

    auto resultType = cast<RankedTensorType>(bitcastOp.getType());
    rewriter.replaceOpWithNewOp<BitCastOp>(
        bitcastOp, resultType, producerBitcast.getSource(),
        producerBitcast.getSourceDims(), bitcastOp.getResultDims());
    return success();
  }
};

}; // namespace

OpFoldResult BitCastOp::fold(FoldAdaptor operands) {
  auto sourceType = cast<ShapedType>(getSource().getType());
  auto resultType = cast<ShapedType>(getResult().getType());
  if (sourceType.getElementType() != resultType.getElementType()) {
    // Element type mismatch, this is a bitcast.
    return {};
  }
  if (compareShapesEqual(sourceType, getSourceDims(), resultType,
                         getResultDims())) {
    // Shapes match and this is a no-op so just fold to the source.
    return getSource();
  }
  return {};
}

void BitCastOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.insert<ReplaceBitCastIfTensorOperandEmpty,
                 BitCastOfTensorCastStaticInfo, ChainedBitCast>(context);
}

//===----------------------------------------------------------------------===//
// iree_tensor_ext.dispatch.tensor.load
//===----------------------------------------------------------------------===//

namespace {

// Updates the |dimValues| of |tensorValue| with dimensions inferred from IR.
// The dimension values may be derived values that are redundant with captured
// dimensions and by redirecting to the captured values we can simplify things.
// Returns true if the dims were changed.
static bool updateTensorOpDims(RewriterBase &rewriter, Operation *op,
                               Value tensorValue,
                               MutableOperandRange mutableDimValues) {
  auto dynamicDimsOr = IREE::Util::findDynamicDims(tensorValue, op->getBlock(),
                                                   Block::iterator(op));
  if (!dynamicDimsOr.has_value()) {
    return false;
  }
  auto dynamicDims = dynamicDimsOr.value();
  bool anyChanged = false;
  OperandRange oldValueRange = mutableDimValues;
  auto oldValues = llvm::to_vector(oldValueRange);
  for (unsigned i = 0; i < dynamicDims.size(); ++i) {
    if (oldValues[i] != dynamicDims[i]) {
      rewriter.modifyOpInPlace(
          op, [&]() { mutableDimValues.slice(i, 1).assign(dynamicDims[i]); });
      anyChanged = true;
    }
  }
  return anyChanged;
}

struct ReuseDispatchTensorLoadShapeDims
    : public OpRewritePattern<DispatchTensorLoadOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(DispatchTensorLoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    return success(updateTensorOpDims(rewriter, loadOp, loadOp.getSource(),
                                      loadOp.getSourceDimsMutable()));
  }
};

// Inlining producers of an input to the dispatch region results in the
// `flow.dispatch.input.load` having a `tensor` type as input. This fails
// verification. Since inlining happens during canonicalization, add a pattern
// to convert
//
// flow.dispatch.input.load %v, offsets .., sizes .., strides..
//   : tensor<...> -> tensor<..>
//
// to
//
// subtensor %v[..] [..] [..]
struct ConvertDispatchInputLoadOfTensorToSubTensor
    : public OpRewritePattern<DispatchTensorLoadOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(DispatchTensorLoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    if (!isa<RankedTensorType>(loadOp.getSource().getType())) {
      return failure();
    }
    // If the offsets are empty rely on folding to take care of it.
    if (loadOp.offsets().empty() && loadOp.sizes().empty() &&
        loadOp.strides().empty()) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
        loadOp, loadOp.getSource(), loadOp.getMixedOffsets(),
        loadOp.getMixedSizes(), loadOp.getMixedStrides());
    return success();
  }
};

/// For `op` that implements the `OffsetsStridesAndSizesInterface`, canonicalize
/// the `offsets`, `sizes` and `strides` by replacing aby value operand that is
/// defined by a constant with the integer value directly. The type of the slice
/// (result type for `iree_tensor_ext.dispatch.tensor.load` and `value` type for
/// `iree_tensor_ext.dispatch.tensor.store`) is also passed in. The type of the
/// slice to use in the canonicalized op is returned.
template <typename OpTy>
static FailureOr<RankedTensorType>
canonicalizeSubViewParts(OpTy op, RankedTensorType sliceType,
                         SmallVector<OpFoldResult> &mixedOffsets,
                         SmallVector<OpFoldResult> &mixedSizes,
                         SmallVector<OpFoldResult> &mixedStrides) {
  // If there are no constant operands then we return early before the more
  // expensive work below.
  if (llvm::none_of(op.offsets(),
                    [](Value operand) {
                      return matchPattern(operand, matchConstantIndex());
                    }) &&
      llvm::none_of(op.sizes(),
                    [](Value operand) {
                      return matchPattern(operand, matchConstantIndex());
                    }) &&
      llvm::none_of(op.strides(), [](Value operand) {
        return matchPattern(operand, matchConstantIndex());
      })) {
    return failure();
  }

  // At least one of offsets/sizes/strides is a new constant.
  // Form the new list of operands and constant attributes from the existing.
  mixedOffsets.assign(op.getMixedOffsets());
  mixedSizes.assign(op.getMixedSizes());
  mixedStrides.assign(op.getMixedStrides());
  if (failed(foldDynamicIndexList(mixedOffsets)) &&
      failed(foldDynamicIndexList(mixedSizes)) &&
      failed(foldDynamicIndexList(mixedStrides))) {
    return failure();
  }

  // Drop out the same dimensions form before.
  llvm::SmallVector<int64_t> newShape;
  llvm::SmallBitVector droppedDims = op.getDroppedDims();
  for (auto size : llvm::enumerate(mixedSizes)) {
    if (droppedDims.test(size.index())) {
      continue;
    }
    std::optional<int64_t> staticSize = getConstantIntValue(size.value());
    newShape.push_back(staticSize ? staticSize.value() : ShapedType::kDynamic);
  }

  auto newSliceType =
      RankedTensorType::get(newShape, sliceType.getElementType());
  return newSliceType;
}

/// Pattern to rewrite a subview op with constant arguments.
struct DispatchTensorLoadOpWithOffsetSizesAndStridesConstantArgumentFolder final
    : public OpRewritePattern<DispatchTensorLoadOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(DispatchTensorLoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    SmallVector<OpFoldResult> mixedOffsets, mixedSizes, mixedStrides;
    RankedTensorType resultType = loadOp.getType();
    auto newResultType = canonicalizeSubViewParts(
        loadOp, resultType, mixedOffsets, mixedSizes, mixedStrides);
    if (failed(newResultType)) {
      return failure();
    }

    // We need to resolve the new inferred type with the specified type.
    Location loc = loadOp.getLoc();
    Value replacement = DispatchTensorLoadOp::create(
        rewriter, loc, newResultType.value(), loadOp.getSource(),
        loadOp.getSourceDims(), mixedOffsets, mixedSizes, mixedStrides);
    if (newResultType.value() != resultType) {
      replacement =
          tensor::CastOp::create(rewriter, loc, resultType, replacement);
    }
    rewriter.replaceOp(loadOp, replacement);
    return success();
  }
};

} // namespace

void DispatchTensorLoadOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<ReuseDispatchTensorLoadShapeDims>(context);
  results.insert<ConvertDispatchInputLoadOfTensorToSubTensor>(context);
  results.insert<
      DispatchTensorLoadOpWithOffsetSizesAndStridesConstantArgumentFolder>(
      context);
}

// Inlining producers of an input to the dispatch region results in the
// `flow.dispatch.input.load` having a `tensor` type as input. This fails
// verification. Fold such uses of the offsets, size and strides are emtpy.
// i.e, flow.dispatch.input.load %v -> %v
OpFoldResult DispatchTensorLoadOp::fold(FoldAdaptor operands) {
  if (getSource().getType() && isa<RankedTensorType>(getSource().getType()) &&
      getMixedOffsets().empty() && getMixedSizes().empty() &&
      getMixedStrides().empty()) {
    return getSource();
  }
  return {};
}

//===----------------------------------------------------------------------===//
// iree_tensor_ext.dispatch.tensor.store
//===----------------------------------------------------------------------===//

namespace {

struct ReuseDispatchTensorStoreShapeDims
    : public OpRewritePattern<DispatchTensorStoreOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(DispatchTensorStoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    return success(updateTensorOpDims(rewriter, storeOp, storeOp.getTarget(),
                                      storeOp.getTargetDimsMutable()));
  }
};

struct FoldCastOpIntoDispatchStoreOp
    : public OpRewritePattern<DispatchTensorStoreOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(DispatchTensorStoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    auto parentOp = storeOp.getValue().getDefiningOp<tensor::CastOp>();
    if (!parentOp || !tensor::canFoldIntoConsumerOp(parentOp)) {
      return failure();
    }

    // Only fold a cast when the (rank-reduced) type is consistent with the
    // static sizes.
    auto sourceTensorType =
        dyn_cast<RankedTensorType>(parentOp.getSource().getType());
    if (!sourceTensorType) {
      return failure();
    }
    auto inferredType = RankedTensorType::get(
        storeOp.getStaticSizes(), sourceTensorType.getElementType());
    if (isRankReducedType(inferredType, sourceTensorType) !=
        SliceVerificationResult::Success) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<DispatchTensorStoreOp>(
        storeOp, parentOp.getSource(), storeOp.getTarget(),
        storeOp.getTargetDims(), storeOp.getOffsets(), storeOp.getSizes(),
        storeOp.getStrides(), storeOp.getStaticOffsets(),
        storeOp.getStaticSizes(), storeOp.getStaticStrides());
    return success();
  }
};

struct DispatchTensorStoreOpWithOffsetSizesAndStridesConstantArgumentFolder
    final : public OpRewritePattern<DispatchTensorStoreOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(DispatchTensorStoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    SmallVector<OpFoldResult> mixedOffsets, mixedSizes, mixedStrides;
    RankedTensorType valueType = storeOp.getValueType();
    auto newValueType = canonicalizeSubViewParts(
        storeOp, valueType, mixedOffsets, mixedSizes, mixedStrides);
    if (failed(newValueType)) {
      return failure();
    }

    Value value = storeOp.getValue();
    Location loc = storeOp.getLoc();
    if (newValueType.value() != valueType) {
      value =
          tensor::CastOp::create(rewriter, loc, newValueType.value(), value);
    }
    rewriter.replaceOpWithNewOp<DispatchTensorStoreOp>(
        storeOp, value, storeOp.getTarget(), storeOp.getTargetDims(),
        mixedOffsets, mixedSizes, mixedStrides);
    return success();
  }
};

} // namespace

void DispatchTensorStoreOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<
      DispatchTensorStoreOpWithOffsetSizesAndStridesConstantArgumentFolder,
      FoldCastOpIntoDispatchStoreOp, ReuseDispatchTensorStoreShapeDims>(
      context);
}

//===----------------------------------------------------------------------===//
// iree_tensor_ext.dispatch.workload.ordinal
//===----------------------------------------------------------------------===//

namespace {

// Bubble up the ordinal ops so that all uses go through this operation.
struct BubbleUpOrdinalOp : public OpRewritePattern<DispatchWorkloadOrdinalOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(DispatchWorkloadOrdinalOp ordinalOp,
                                PatternRewriter &rewriter) const override {
    auto blockArg = dyn_cast<BlockArgument>(ordinalOp.getOperand());
    if (!blockArg) {
      return failure();
    }
    if (blockArg.hasOneUse()) {
      // Nothing to do.
      return failure();
    }
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(ordinalOp->getBlock());
    // Adjust the insertion point to keep the ordinals in order
    for (Operation &op : *ordinalOp->getBlock()) {
      if (auto insertionPoint = dyn_cast<DispatchWorkloadOrdinalOp>(&op)) {
        if (insertionPoint.getOrdinal().getZExtValue() <
            ordinalOp.getOrdinal().getZExtValue()) {
          rewriter.setInsertionPointAfter(insertionPoint);
          continue;
        }
      }
      break;
    }
    auto newOrdinalOp = DispatchWorkloadOrdinalOp::create(
        rewriter, ordinalOp.getLoc(), blockArg, ordinalOp.getOrdinalAttr());
    rewriter.replaceAllUsesExcept(blockArg, newOrdinalOp, newOrdinalOp);
    rewriter.replaceOp(ordinalOp, newOrdinalOp.getResult());
    return success();
  }
};

} // namespace

OpFoldResult DispatchWorkloadOrdinalOp::fold(FoldAdaptor operands) {
  // If the operand being annotated is a constant then just fold to it as
  // there's no longer any relation to the captured workload.
  if (operands.getOperand()) {
    return operands.getOperand();
  }

  // Fold away following sequence ordinal ops:
  //
  //   %1 = iree_tensor_ext.dispatch.workload.ordinal %0, 2
  //   %2 = iree_tensor_ext.dispatch.workload.ordinal %1, 2
  //
  // This can happen when the operands get deduped.
  if (auto producerOrdinalOp = dyn_cast_if_present<DispatchWorkloadOrdinalOp>(
          getOperand().getDefiningOp())) {
    if (producerOrdinalOp.getOrdinal() == getOrdinal()) {
      return producerOrdinalOp.getOperand();
    }
  }

  return {};
}

void DispatchWorkloadOrdinalOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<BubbleUpOrdinalOp>(context);
}

//===----------------------------------------------------------------------===//
// iree_tensor_ext.compute_barrier.start
//===----------------------------------------------------------------------===//

OpFoldResult ComputeBarrierStartOp::fold(FoldAdaptor adaptor) {
  // Fold duplicate barriers in a chain:
  // compute_barrier.start(compute_barrier.start(x)) -> compute_barrier.start(x)
  if (auto producer = getValue().getDefiningOp<ComputeBarrierStartOp>()) {
    return producer.getResult();
  }
  return {};
}

//===----------------------------------------------------------------------===//
// iree_tensor_ext.compute_barrier.end
//===----------------------------------------------------------------------===//

OpFoldResult ComputeBarrierEndOp::fold(FoldAdaptor adaptor) {
  // Fold duplicate barriers in a chain:
  // compute_barrier.end(compute_barrier.end(x)) -> compute_barrier.end(x)
  if (auto producer = getValue().getDefiningOp<ComputeBarrierEndOp>()) {
    return producer.getResult();
  }
  return {};
}

} // namespace mlir::iree_compiler::IREE::TensorExt

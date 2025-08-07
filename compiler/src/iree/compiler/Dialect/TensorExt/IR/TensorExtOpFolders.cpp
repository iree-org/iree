// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "iree/compiler/Utils/ShapeUtils.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace mlir::iree_compiler::IREE::TensorExt {

//===----------------------------------------------------------------------===//
// iree_tensor_ext.bitcast
//===----------------------------------------------------------------------===//

namespace {
struct ReplaceBitCastIfTensorOperandEmpty final : OpRewritePattern<BitCastOp> {
  using OpRewritePattern<BitCastOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(BitCastOp op,
                                PatternRewriter &rewriter) const override {
    auto emptyOp =
        dyn_cast_or_null<tensor::EmptyOp>(op.getSource().getDefiningOp());
    if (!emptyOp)
      return failure();
    rewriter.replaceOpWithNewOp<tensor::EmptyOp>(op, op.getResult().getType(),
                                                 op.getResultDims());
    return success();
  }
};

struct BitCastOfTensorCastStaticInfo final : OpRewritePattern<BitCastOp> {
  using OpRewritePattern<BitCastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BitCastOp bitcastOp,
                                PatternRewriter &rewriter) const final {
    auto tensorCastOp = bitcastOp.getSource().getDefiningOp<tensor::CastOp>();
    if (!tensorCastOp)
      return failure();
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

    // TODO: Support partial static info incorporation.
    if (bitcastOp.getSourceDims() != bitcastOp.getResultDims()) {
      return failure();
    }

    RankedTensorType intermediateTensorType = bitcastOp.getSource().getType();
    RankedTensorType resTensorType = bitcastOp.getResult().getType();
    ArrayRef<int64_t> resShape = resTensorType.getShape();

    SmallVector<Value> newDynamicDims;
    int64_t intermediateDynamicDim = 0;
    int64_t resDynamicDim = 0;
    SmallVector<int64_t> newResultSizes(resShape);
    // Drop the dynamic dims that become static after incorporating the cast.
    for (auto [castSize, sourceSize] : llvm::zip_equal(
             tensorCastSrcType.getShape(), intermediateTensorType.getShape())) {
      if (!ShapedType::isDynamic(sourceSize))
        continue;

      while (!ShapedType::isDynamic(resShape[resDynamicDim])) {
        ++resDynamicDim;
      }

      if (ShapedType::isDynamic(castSize)) {
        newDynamicDims.push_back(
            bitcastOp.getSourceDims()[intermediateDynamicDim]);
      } else {
        newResultSizes[resDynamicDim] = castSize;
      }
      ++intermediateDynamicDim;
    }

    auto newType =
        RankedTensorType::get(newResultSizes, resTensorType.getElementType(),
                              resTensorType.getEncoding());
    Value newBitcast = rewriter.create<BitCastOp>(
        bitcastOp.getLoc(), newType, tensorCastOp.getOperand(), newDynamicDims,
        newDynamicDims);
    // We create a new cast to continue propagating static information.
    rewriter.replaceOpWithNewOp<tensor::CastOp>(bitcastOp, resTensorType,
                                                newBitcast);

    return success();
  }
};

}; // namespace

OpFoldResult BitCastOp::fold(FoldAdaptor operands) {
  auto sourceType = llvm::cast<ShapedType>(getSource().getType());
  auto resultType = llvm::cast<ShapedType>(getResult().getType());
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
                 BitCastOfTensorCastStaticInfo>(context);
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
  if (!dynamicDimsOr.has_value())
    return false;
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
  using OpRewritePattern::OpRewritePattern;
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
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DispatchTensorLoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    if (!llvm::isa<RankedTensorType>(loadOp.getSource().getType())) {
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
    if (droppedDims.test(size.index()))
      continue;
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
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DispatchTensorLoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    SmallVector<OpFoldResult> mixedOffsets, mixedSizes, mixedStrides;
    RankedTensorType resultType = loadOp.getType();
    auto newResultType = canonicalizeSubViewParts(
        loadOp, resultType, mixedOffsets, mixedSizes, mixedStrides);
    if (failed(newResultType))
      return failure();

    // We need to resolve the new inferred type with the specified type.
    Location loc = loadOp.getLoc();
    Value replacement = rewriter.create<DispatchTensorLoadOp>(
        loc, newResultType.value(), loadOp.getSource(), loadOp.getSourceDims(),
        mixedOffsets, mixedSizes, mixedStrides);
    if (newResultType.value() != resultType) {
      replacement =
          rewriter.create<tensor::CastOp>(loc, resultType, replacement);
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
  if (getSource().getType() &&
      llvm::isa<RankedTensorType>(getSource().getType()) &&
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
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DispatchTensorStoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    return success(updateTensorOpDims(rewriter, storeOp, storeOp.getTarget(),
                                      storeOp.getTargetDimsMutable()));
  }
};

struct FoldCastOpIntoDispatchStoreOp
    : public OpRewritePattern<DispatchTensorStoreOp> {
  using OpRewritePattern::OpRewritePattern;
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
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DispatchTensorStoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    SmallVector<OpFoldResult> mixedOffsets, mixedSizes, mixedStrides;
    RankedTensorType valueType = storeOp.getValueType();
    auto newValueType = canonicalizeSubViewParts(
        storeOp, valueType, mixedOffsets, mixedSizes, mixedStrides);
    if (failed(newValueType))
      return failure();

    Value value = storeOp.getValue();
    Location loc = storeOp.getLoc();
    if (newValueType.value() != valueType) {
      value = rewriter.create<tensor::CastOp>(loc, newValueType.value(), value);
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
  using OpRewritePattern::OpRewritePattern;
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
    auto newOrdinalOp = rewriter.create<DispatchWorkloadOrdinalOp>(
        ordinalOp.getLoc(), blockArg, ordinalOp.getOrdinalAttr());
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
  if (auto producerOrdinalOp = dyn_cast_or_null<DispatchWorkloadOrdinalOp>(
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

} // namespace mlir::iree_compiler::IREE::TensorExt

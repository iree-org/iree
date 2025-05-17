// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <numeric>
#include <optional>

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "iree/compiler/Dialect/Util/IR/ClosureOpUtils.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringExtras.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::iree_compiler::IREE::Flow {

//===----------------------------------------------------------------------===//
// Folding utilities
//===----------------------------------------------------------------------===//

namespace {

// Erases an op if it has no uses.
// This is to support ops that are "pure" but can't be marked as such because
// the MLIR CSE pass would deduplicate them.
template <typename Op>
struct ElideUnusedOp : public OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;
  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter &rewriter) const override {
    if (!op.use_empty())
      return failure();
    rewriter.eraseOp(op);
    return success();
  }
};

// Returns true if |value| is definitely empty at runtime.
static bool isTensorZeroElements(Value value) {
  auto type = llvm::dyn_cast<ShapedType>(value.getType());
  if (!type)
    return false;
  // Any static dimension being zero is definitely empty.
  for (int64_t i = 0; i < type.getRank(); ++i) {
    int64_t dim = type.getDimSize(i);
    if (dim == 0)
      return true;
  }
  return false; // may still be dynamically empty
}

// Returns true if |value| is definitely empty at runtime.
// Returns false if the value is definitely not empty or may be empty at runtime
// (one or more dynamic dimensions).
static bool isTensorOperandZeroElements(Value value) {
  return isTensorZeroElements(value);
}

// Returns true if |value| is definitely empty at runtime.
// Returns false if the value is definitely not empty or may be empty at runtime
// (one or more dynamic dimensions).
static bool isTensorResultZeroElements(Value value) {
  return isTensorZeroElements(value);
}

template <typename Op, int OperandIdx, int ResultIdx = 0>
struct ReplaceOpIfTensorOperandZeroElements : public OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;
  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter &rewriter) const override {
    auto operand = op->getOperand(OperandIdx);
    if (!isTensorOperandZeroElements(operand))
      return failure();
    auto result = op->getResult(ResultIdx);
    auto dynamicDims = op.getResultDynamicDims(result.getResultNumber());
    rewriter.replaceOpWithNewOp<IREE::Flow::TensorEmptyOp>(op, result.getType(),
                                                           dynamicDims);
    return success();
  }
};

template <typename Op, int ResultIdx>
struct ReplaceOpIfTensorResultZeroElements : public OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;
  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter &rewriter) const override {
    auto result = op->getResult(ResultIdx);
    if (!isTensorResultZeroElements(result))
      return failure();
    auto dynamicDims = op.getResultDynamicDims(result.getResultNumber());
    rewriter.replaceOpWithNewOp<IREE::Flow::TensorEmptyOp>(op, result.getType(),
                                                           dynamicDims);
    return success();
  }
};

template <typename Op, int OperandIdx, int ResultIdx = 0>
struct ReplaceOpIfTensorOperandEmpty : public OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;
  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter &rewriter) const override {
    auto operand = op->getOperand(OperandIdx);
    auto emptyOp = dyn_cast_or_null<TensorEmptyOp>(operand.getDefiningOp());
    if (!emptyOp)
      return failure();
    auto result = op->getResult(ResultIdx);
    auto dynamicDims = op.getResultDynamicDims(result.getResultNumber());
    rewriter.replaceOpWithNewOp<IREE::Flow::TensorEmptyOp>(op, result.getType(),
                                                           dynamicDims);
    return success();
  }
};

// Returns a new set of dynamic dimensions for a shape carrying op when a type
// is being changed. This attempts to reuse the existing dimension values if
// they are available and will drop/insert new ones as required.
static SmallVector<Value> refreshDimsOnTypeChange(Operation *op, Type oldType,
                                                  Type newType,
                                                  ValueRange oldDims,
                                                  PatternRewriter &rewriter) {
  if (oldType == newType)
    return llvm::to_vector(oldDims);

  // Build an expanded list of all the dims - constants will be nullptr.
  // This lets us map back the new types without worrying about whether some
  // subset become static or dynamic.
  auto oldShapedType = llvm::cast<ShapedType>(oldType);
  SmallVector<Value> allOldDims(oldShapedType.getRank());
  for (unsigned i = 0; i < oldShapedType.getRank(); ++i) {
    if (oldShapedType.isDynamicDim(i)) {
      allOldDims[i] = oldDims.front();
      oldDims = oldDims.drop_front();
    }
  }

  auto newShapedType = llvm::cast<ShapedType>(newType);
  SmallVector<Value> newDims;
  for (unsigned i = 0; i < newShapedType.getRank(); ++i) {
    if (newShapedType.isDynamicDim(i)) {
      auto oldValue = allOldDims[i];
      if (oldValue) {
        // Old value valid; reuse.
        newDims.push_back(oldValue);
      } else {
        // Dimension has changed to be dynamic; insert a constant to use.
        // This sometimes happens during folding of casts and usually is cleaned
        // up pretty quickly.
        newDims.push_back(rewriter.createOrFold<arith::ConstantIndexOp>(
            op->getLoc(), oldShapedType.getDimSize(i)));
      }
    }
  }
  return newDims;
}

} // namespace

//===----------------------------------------------------------------------===//
// flow.dispatch.workgroups
//===----------------------------------------------------------------------===//

/// Helper method to take a list of values to be deduped and returns
/// - list of deduped values.
/// - mapping for a value from its position in the original list to
///   the deduped list.
static std::tuple<SmallVector<Value>, llvm::MapVector<int, int>>
dedupAndGetOldToNewPosMapping(ValueRange values) {
  llvm::MapVector<int, int> oldPosToNewPos;
  SmallVector<Value> uniquedList;
  int numUnique = 0;
  llvm::MapVector<Value, int> oldValueToNewPos;
  for (auto [index, val] : llvm::enumerate(values)) {
    if (oldValueToNewPos.count(val)) {
      oldPosToNewPos[index] = oldValueToNewPos[val];
      continue;
    }
    oldPosToNewPos[index] = numUnique;
    oldValueToNewPos[val] = numUnique;
    uniquedList.push_back(val);
    numUnique++;
  }
  return {uniquedList, oldPosToNewPos};
}

struct ReplaceDispatchResultIfZeroElements
    : public OpRewritePattern<DispatchWorkgroupsOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DispatchWorkgroupsOp op,
                                PatternRewriter &rewriter) const override {
    // NOTE: we only look at used results; if unused then closure optimization
    // will drop it.
    bool didReplaceAny = false;
    for (auto result : op.getResults()) {
      if (result.use_empty())
        continue;
      if (isTensorResultZeroElements(result)) {
        auto dynamicDims = op.getResultDynamicDims(result.getResultNumber());
        auto emptyOp = rewriter.create<IREE::Flow::TensorEmptyOp>(
            result.getLoc(), result.getType(), dynamicDims);
        rewriter.replaceAllUsesWith(result, emptyOp);
        didReplaceAny = true;
      }
    }
    return didReplaceAny ? success() : failure();
  }
};

/// Deduplicate redundant workload values of a dispatch.workgroups op. This
/// requires modifying the `count` region of the op to match the new workloads.
struct ElideRedundantWorkloadValues
    : public OpRewritePattern<DispatchWorkgroupsOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DispatchWorkgroupsOp op,
                                PatternRewriter &rewriter) const override {
    ValueRange workload = op.getWorkload();
    auto [newWorkload, oldWorkloadPosToNewWorkloadPos] =
        dedupAndGetOldToNewPosMapping(workload);
    if (newWorkload.size() == workload.size()) {
      // Nothing to do.
      return failure();
    }

    // Create a new flow.dispatch.workgroup op with new workloads.
    Location loc = op.getLoc();
    auto newWorkgroupsOp = rewriter.create<DispatchWorkgroupsOp>(
        loc, newWorkload, op.getResultTypes(), op.getResultDims(),
        op.getArguments(), op.getArgumentDims(),
        op.getTiedOperandsAsIntegerList(),
        getPrunedAttributeList(op, /*elidedAttrs=*/{}));

    // Move the body over.
    Region &body = op.getWorkgroupBody();
    if (!body.empty()) {
      Region &newBody = newWorkgroupsOp.getWorkgroupBody();
      rewriter.inlineRegionBefore(body, newBody, newBody.begin());
    }

    // Move the workgroup count region over.
    Region &count = op.getWorkgroupCount();
    if (!count.empty()) {
      Region &newCount = newWorkgroupsOp.getWorkgroupCount();
      rewriter.inlineRegionBefore(count, newCount, newCount.begin());

      // Create a new entry basic block with as many arguments as the workload
      // and then merge this block with the original entry block.
      auto newWorkloadTypes =
          llvm::map_to_vector(newWorkload, [](Value v) { return v.getType(); });
      auto newWorkloadLocs =
          llvm::map_to_vector(newWorkload, [](Value v) { return v.getLoc(); });
      Block *oldCountBlock = &newCount.front();
      Block *newCountBlock = rewriter.createBlock(
          &newCount.front(), newWorkloadTypes, newWorkloadLocs);
      auto newCountBlockArgs = newCountBlock->getArguments();
      SmallVector<Value> replacements;
      replacements.resize(oldCountBlock->getNumArguments());
      for (auto [index, val] : llvm::enumerate(oldCountBlock->getArguments())) {
        replacements[index] =
            newCountBlockArgs[oldWorkloadPosToNewWorkloadPos.lookup(index)];
      }
      rewriter.mergeBlocks(oldCountBlock, newCountBlock, replacements);
    }

    // Replace the old workgroups op with the new workgroups op.
    rewriter.replaceOp(op, newWorkgroupsOp.getResults());
    return success();
  }
};

/// Deduplicate operands of the `dispatch.workgroup_count_from_slice` op. This
/// requires updating the `iree_tensor_ext.dispatch.workload.ordinal` operation
/// in the body of the `dispatch.workgroups` op to match the new positions of
/// the operands in the `dispatch.workgroup_count_from_slice`.
struct ElideRedundantOperandsOfWorkgroupCountFromSliceOp
    : OpRewritePattern<DispatchWorkgroupsOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DispatchWorkgroupsOp op,
                                PatternRewriter &rewriter) const override {
    Region &count = op.getWorkgroupCount();
    if (count.empty()) {
      return failure();
    }

    assert(
        llvm::hasSingleElement(count) &&
        "expected dispatch.workgroup op count region to have a single block");

    // Check for `dispatch.workgroup_count_from_slice` operations in the count
    // region.
    Block &countBody = count.front();
    auto countFromSliceOps =
        countBody.getOps<IREE::TensorExt::DispatchWorkgroupCountFromSliceOp>();
    if (countFromSliceOps.empty()) {
      return failure();
    }
    assert(llvm::hasSingleElement(countFromSliceOps) &&
           "expected only one dispatch.workgroup_count_from_slice op in count "
           "region");
    auto countFromSliceOp = *countFromSliceOps.begin();

    // Deduplicate the operands and get a mapping from old position to new
    // position.
    auto [newOrdinals, oldOrdinalPosToNewOrdinalPos] =
        dedupAndGetOldToNewPosMapping(countFromSliceOp.getOperands());
    if (newOrdinals.size() == countFromSliceOp.getNumOperands()) {
      return failure();
    }

    // Replace the old `dispatch.workgroup_count_from_slice` with a new op
    // with deduped operands.
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(countFromSliceOp);
    rewriter
        .replaceOpWithNewOp<IREE::TensorExt::DispatchWorkgroupCountFromSliceOp>(
            countFromSliceOp, newOrdinals);

    // Adjust the iree_tensor_ext.dispatch.workload.ordinal ops in the body to
    // use the new ordinal numbers.
    Region &body = op.getWorkgroupBody();
    SmallVector<IREE::TensorExt::DispatchWorkloadOrdinalOp> ordinalOps;
    body.walk([&](IREE::TensorExt::DispatchWorkloadOrdinalOp ordinalOp) {
      ordinalOps.push_back(ordinalOp);
    });

    for (auto ordinalOp : ordinalOps) {
      int oldOrdinalPos = ordinalOp.getOrdinal().getSExtValue();
      rewriter.setInsertionPoint(ordinalOp);
      rewriter.replaceOpWithNewOp<IREE::TensorExt::DispatchWorkloadOrdinalOp>(
          ordinalOp, ordinalOp.getOperand(),
          rewriter.getIndexAttr(
              oldOrdinalPosToNewOrdinalPos.lookup(oldOrdinalPos)));
    }
    rewriter.modifyOpInPlace(op, []() {});
    return success();
  }
};

void DispatchWorkgroupsOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  // Disable constant inlining as we have done it during dispatch region
  // formation.
  IREE::Util::ClosureOptimizationOptions closureOptions;
  closureOptions.maxInlinedConstantBytes = 0;
  results.insert<IREE::Util::ClosureOptimizationPattern<DispatchWorkgroupsOp>>(
      context, closureOptions);
  results.insert<ElideRedundantWorkloadValues,
                 ElideRedundantOperandsOfWorkgroupCountFromSliceOp,
                 ReplaceDispatchResultIfZeroElements>(context);
}

//===----------------------------------------------------------------------===//
// flow.dispatch.tie_shape
//===----------------------------------------------------------------------===//

OpFoldResult DispatchTieShapeOp::fold(FoldAdaptor operands) {
  if (getDynamicDims().empty()) {
    return getOperand();
  }
  return {};
}

//===----------------------------------------------------------------------===//
// flow.dispatch
//===----------------------------------------------------------------------===//

namespace {

struct DeduplicateDispatchEntryRefs final
    : public OpRewritePattern<DispatchOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DispatchOp dispatchOp,
                                PatternRewriter &rewriter) const override {
    auto originalAttr = dispatchOp.getEntryPointsAttr();
    auto newAttr = deduplicateArrayElements(originalAttr);
    if (newAttr == originalAttr)
      return failure();
    rewriter.modifyOpInPlace(dispatchOp,
                             [&]() { dispatchOp.setEntryPointsAttr(newAttr); });
    return success();
  }
};

} // namespace

void DispatchOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                             MLIRContext *context) {
  results.insert<DeduplicateDispatchEntryRefs>(context);
}

//===----------------------------------------------------------------------===//
// Tensor ops
//===----------------------------------------------------------------------===//

/// Reduces the provided multidimensional index into a flattended 1D row-major
/// index. The |type| is expected to be statically shaped (as all constants
/// are).
static uint64_t getFlattenedIndex(ShapedType type, ArrayRef<uint64_t> index) {
  assert(type.hasStaticShape() && "for use on statically shaped types only");
  auto rank = type.getRank();
  auto shape = type.getShape();
  uint64_t valueIndex = 0;
  uint64_t dimMultiplier = 1;
  for (int i = rank - 1; i >= 0; --i) {
    valueIndex += index[i] * dimMultiplier;
    dimMultiplier *= shape[i];
  }
  return valueIndex;
}

static bool compareShapesEqual(ShapedType lhsType, ValueRange lhsDynamicDims,
                               ShapedType rhsType, ValueRange rhsDynamicDims) {
  if (lhsType.hasStaticShape() && rhsType.hasStaticShape()) {
    // Static shape equivalence means we can fast-path the check.
    return lhsType == rhsType;
  }
  if (lhsType.getRank() != rhsType.getRank()) {
    return false;
  }
  unsigned dynamicDimIndex = 0;
  unsigned numNonmatchingSSADims = 0;
  for (unsigned i = 0; i < lhsType.getRank(); ++i) {
    if (lhsType.isDynamicDim(i) != rhsType.isDynamicDim(i)) {
      // Static/dynamic dimension mismatch - definitely differ.
      return false;
    } else if (lhsType.isDynamicDim(i)) {
      unsigned j = dynamicDimIndex++;
      if (lhsDynamicDims[j] != rhsDynamicDims[j]) {
        numNonmatchingSSADims++;
      }
    } else {
      if (lhsType.getDimSize(i) != rhsType.getDimSize(i)) {
        // Static dimensions differ.
        return false;
      }
    }
  }
  return numNonmatchingSSADims <= 1;
}

//===----------------------------------------------------------------------===//
// flow.tensor.constant
//===----------------------------------------------------------------------===//

OpFoldResult TensorConstantOp::fold(FoldAdaptor operands) { return getValue(); }

//===----------------------------------------------------------------------===//
// flow.tensor.dynamic_constant
//===----------------------------------------------------------------------===//

OpFoldResult TensorDynamicConstantOp::fold(FoldAdaptor operands) {
  auto dynamicType = getType();
  if (dynamicType.getNumDynamicDims() == 0) {
    return getValue();
  }
  return {};
}

namespace {

struct ExpandDynamicShapeConstant
    : public OpRewritePattern<TensorDynamicConstantOp> {
  using OpRewritePattern<TensorDynamicConstantOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(TensorDynamicConstantOp op,
                                PatternRewriter &rewriter) const override {
    auto constantOp = rewriter.create<IREE::Flow::TensorConstantOp>(
        op.getLoc(), op.getValue());
    auto dynamicType = op.getType();
    auto staticType = cast<ShapedType>(op.getValue().getType());
    SmallVector<Value> dynamicDims;
    for (int64_t i = 0; i < dynamicType.getRank(); ++i) {
      if (dynamicType.isDynamicDim(i)) {
        auto dimValue = rewriter
                            .create<arith::ConstantIndexOp>(
                                op.getLoc(), staticType.getDimSize(i))
                            .getResult();
        dynamicDims.push_back(rewriter
                                  .create<IREE::Util::OptimizationBarrierOp>(
                                      op.getLoc(), dimValue)
                                  .getResult(0));
      }
    }
    rewriter.replaceOpWithNewOp<IREE::Flow::TensorReshapeOp>(
        op, dynamicType, constantOp.getResult(), dynamicDims);
    return success();
  }
};

} // namespace

void TensorDynamicConstantOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<ExpandDynamicShapeConstant>(context);
}

//===----------------------------------------------------------------------===//
// flow.tensor.tie_shape
//===----------------------------------------------------------------------===//

OpFoldResult TensorTieShapeOp::fold(FoldAdaptor operands) {
  if (getDynamicDims().empty()) {
    return getOperand();
  }
  return {};
}

void TensorTieShapeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                   MLIRContext *context) {
  results.insert<ReplaceOpIfTensorOperandZeroElements<TensorTieShapeOp, 0>>(
      context);
}

//===----------------------------------------------------------------------===//
// flow.tensor.reshape
//===----------------------------------------------------------------------===//

OpFoldResult TensorReshapeOp::fold(FoldAdaptor operands) {
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

namespace {

// Flatten a chain of reshapes or bitcasts (reshape/bitcast feeding into
// reshape or bitcast) such that a reshape only ever pulls from a non-reshape
// source. This prevents big useless chains and makes it easier to track the
// original storage for the tensor.
template <typename CastOpTy>
struct FlattenTensorCastLikeChain : public OpRewritePattern<CastOpTy> {
  using OpRewritePattern<CastOpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(CastOpTy reshapeOp,
                                PatternRewriter &rewriter) const override {
    // We want the same result value/shape but to source from the ancestor. We
    // need to pull any dynamic dims from that as we don't care about the
    // intermediate reshapes.
    Value source;
    ValueRange sourceDims;
    if (auto sourceOp = dyn_cast_or_null<TensorReshapeOp>(
            reshapeOp.getSource().getDefiningOp())) {
      source = sourceOp.getSource();
      sourceDims = sourceOp.getSourceDims();
    } else if (auto sourceOp = dyn_cast_or_null<TensorBitCastOp>(
                   reshapeOp.getSource().getDefiningOp())) {
      source = sourceOp.getSource();
      sourceDims = sourceOp.getSourceDims();
    }
    if (!source) {
      return failure();
    }

    auto sourceType = llvm::cast<ShapedType>(source.getType());
    auto resultType = llvm::cast<ShapedType>(reshapeOp.getResult().getType());

    // If the element types don't match, this is a bitcast, else we can use
    // reshape.
    if (sourceType.getElementType() != resultType.getElementType()) {
      rewriter.replaceOpWithNewOp<TensorBitCastOp>(
          reshapeOp, reshapeOp.getResult().getType(), source, sourceDims,
          reshapeOp.getResultDims());
    } else {
      rewriter.replaceOpWithNewOp<TensorReshapeOp>(
          reshapeOp, reshapeOp.getResult().getType(), source, sourceDims,
          reshapeOp.getResultDims());
    }
    return success();
  }
};

struct ResolveShapedRank : public OpRewritePattern<tensor::RankOp> {
  using OpRewritePattern<tensor::RankOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::RankOp op,
                                PatternRewriter &rewriter) const override {
    auto shapedType = llvm::cast<ShapedType>(op.getTensor().getType());
    rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(op,
                                                        shapedType.getRank());
    return success();
  }
};

struct ResolveShapedDim : public OpRewritePattern<tensor::DimOp> {
  using OpRewritePattern<tensor::DimOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::DimOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getConstantIndex().has_value()) {
      return rewriter.notifyMatchFailure(
          op, "non-constant index dim ops are unsupported");
    }
    auto idx = op.getConstantIndex().value();

    // Fold static dims from the type.
    auto shapedType = llvm::cast<ShapedType>(op.getSource().getType());
    if (!shapedType.isDynamicDim(idx)) {
      rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(
          op, shapedType.getDimSize(idx));
      return success();
    }

    // Find dims captured on shape-aware ops.
    auto dynamicDims = IREE::Util::findDynamicDims(
        op.getSource(), op->getBlock(), Block::iterator(op.getOperation()));
    if (dynamicDims.has_value()) {
      unsigned dimOffset = 0;
      for (unsigned i = 0; i < idx; ++i) {
        if (shapedType.isDynamicDim(i))
          ++dimOffset;
      }
      rewriter.replaceOp(op, dynamicDims.value()[dimOffset]);
      return success();
    }

    return rewriter.notifyMatchFailure(op, "no dynamic dims found/usable");
  }
};

} // namespace

void TensorReshapeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                  MLIRContext *context) {
  results.insert<ReplaceOpIfTensorOperandZeroElements<TensorReshapeOp, 0>>(
      context);
  results.insert<ReplaceOpIfTensorResultZeroElements<TensorReshapeOp, 0>>(
      context);
  results.insert<ReplaceOpIfTensorOperandEmpty<TensorReshapeOp, 0, 0>>(context);
  results.insert<FlattenTensorCastLikeChain<TensorReshapeOp>>(context);
  results.insert<ResolveShapedRank>(context);
  results.insert<ResolveShapedDim>(context);
}

//===----------------------------------------------------------------------===//
// flow.tensor.bitcast
//===----------------------------------------------------------------------===//

OpFoldResult TensorBitCastOp::fold(FoldAdaptor operands) {
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

void TensorBitCastOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                  MLIRContext *context) {
  results.insert<ReplaceOpIfTensorOperandZeroElements<TensorBitCastOp, 0>>(
      context);
  results.insert<ReplaceOpIfTensorResultZeroElements<TensorBitCastOp, 0>>(
      context);
  results.insert<ReplaceOpIfTensorOperandEmpty<TensorBitCastOp, 0, 0>>(context);
  results.insert<FlattenTensorCastLikeChain<TensorBitCastOp>>(context);
}

//===----------------------------------------------------------------------===//
// flow.tensor.load
//===----------------------------------------------------------------------===//

OpFoldResult TensorLoadOp::fold(FoldAdaptor operands) {
  if (auto source =
          llvm::dyn_cast_if_present<ElementsAttr>(operands.getSource())) {
    // Load directly from the constant source tensor.
    if (llvm::count(operands.getIndices(), nullptr) == 0) {
      return source.getValues<Attribute>()[llvm::map_to_vector(
          operands.getIndices(), [](Attribute value) {
            return llvm::cast<IntegerAttr>(value).getValue().getZExtValue();
          })];
    }
  }
  return {};
}

namespace {

// Replace `flow.tensor.splat`-`flow.tensor.load` op-pairs by the input
// primitive value for the splat op.
struct FoldSplatLoadIntoPrimitive : public OpRewritePattern<TensorLoadOp> {
  using OpRewritePattern<TensorLoadOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(TensorLoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    auto sourceOp =
        dyn_cast_or_null<TensorSplatOp>(loadOp.getSource().getDefiningOp());
    if (!sourceOp)
      return failure();
    rewriter.replaceOp(loadOp, sourceOp.getValue());
    return success();
  }
};

} // namespace

void TensorLoadOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                               MLIRContext *context) {
  results.insert<FoldSplatLoadIntoPrimitive>(context);
}

//===----------------------------------------------------------------------===//
// flow.tensor.store
//===----------------------------------------------------------------------===//

OpFoldResult TensorStoreOp::fold(FoldAdaptor operands) {
  auto value = operands.getValue();
  if (!value)
    return {};
  if (auto target =
          llvm::dyn_cast_if_present<ElementsAttr>(operands.getTarget())) {
    // Store into the constant target tensor.
    auto targetType = cast<ShapedType>(target.getType());
    if (targetType.getRank() == 0) {
      return DenseElementsAttr::get(targetType, {value});
    }
    if (llvm::count(operands.getIndices(), nullptr) == 0) {
      uint64_t offset = getFlattenedIndex(
          targetType,
          llvm::map_to_vector(operands.getIndices(), [](Attribute value) {
            return llvm::cast<IntegerAttr>(value).getValue().getZExtValue();
          }));
      SmallVector<Attribute, 16> newContents(target.getValues<Attribute>());
      newContents[offset] = value;
      return DenseElementsAttr::get(targetType, newContents);
    }
  }
  return {};
}

//===----------------------------------------------------------------------===//
// flow.tensor.alloca
//===----------------------------------------------------------------------===//

void TensorAllocaOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                 MLIRContext *context) {
  results.insert<ElideUnusedOp<TensorAllocaOp>>(context);
}

//===----------------------------------------------------------------------===//
// flow.tensor.empty
//===----------------------------------------------------------------------===//

void TensorEmptyOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                MLIRContext *context) {
  // TODO(benvanik): fold static shapes into dims.
}

//===----------------------------------------------------------------------===//
// flow.tensor.splat
//===----------------------------------------------------------------------===//

namespace {

struct FoldSplatReshapeIntoSplat : public OpRewritePattern<TensorReshapeOp> {
  using OpRewritePattern<TensorReshapeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(TensorReshapeOp reshapeOp,
                                PatternRewriter &rewriter) const override {
    auto splatOp = dyn_cast_if_present<TensorSplatOp>(
        reshapeOp.getSource().getDefiningOp());
    if (!splatOp)
      return failure();
    rewriter.replaceOpWithNewOp<TensorSplatOp>(
        reshapeOp, reshapeOp.getResult().getType(), splatOp.getValue(),
        reshapeOp.getResultDims());
    return success();
  }
};

} // namespace

void TensorSplatOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                MLIRContext *context) {
  // TODO(benvanik): canonicalize splat+slice to smaller splat.
  results.insert<ReplaceOpIfTensorResultZeroElements<TensorSplatOp, 0>>(
      context);
  results.insert<FoldSplatReshapeIntoSplat>(context);
}

//===----------------------------------------------------------------------===//
// flow.tensor.clone
//===----------------------------------------------------------------------===//

OpFoldResult TensorCloneOp::fold(FoldAdaptor operands) {
  if (auto operand = operands.getOperand()) {
    // Constants always fold.
    return operand;
  }

  // TODO(benvanik): elide clones when safe to do so. Right now clone is
  // load-bearing to work around our lack of cross-stream scheduling. Clones are
  // inserted to avoid mutating function arguments and any logic we perform here
  // (without *also* checking all the conditions that may insert a clone) will
  // just fight.
  //
  // Once the clones are not load-bearing we can remove them in all the normal
  // cases (one user, no intervening uses between clone and consumers of
  // operands, etc).

  return {};
}

void TensorCloneOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                MLIRContext *context) {
  results.insert<ReplaceOpIfTensorOperandZeroElements<TensorCloneOp, 0>>(
      context);
  results.insert<ReplaceOpIfTensorOperandEmpty<TensorCloneOp, 0, 0>>(context);
}

//===----------------------------------------------------------------------===//
// flow.tensor.barrier
//===----------------------------------------------------------------------===//

namespace {

// Moves (or clones) cast-like ops after a transfer operation and possibly
// updates the transfer op to match the new type.
//
// The cast-like behavior makes more sense near consumers: the producer does not
// need the information as it has already extracted it if required before
// converting into flow ops and if there are no consumers of the transfer that
// cast-like op would have been removed anyway.
//
// Example:
//  %reshape = flow.tensor.reshape %source : tensor<1x2xf32> -> tensor<2x1xf32>
//  %target = flow.tensor.transfer %reshape : tensor<2x1xf32> to "foo"
// ->
//  %target = flow.tensor.transfer %source : tensor<1x2xf32> to "foo"
//  %reshape = flow.tensor.reshape %target : tensor<1x2xf32> -> tensor<2x1xf32>
template <typename TransferOpT, typename CastOpT>
struct SinkCastLikeOpAcrossTransfer final : OpRewritePattern<TransferOpT> {
  using OpRewritePattern<TransferOpT>::OpRewritePattern;
  LogicalResult matchAndRewrite(TransferOpT transferOp,
                                PatternRewriter &rewriter) const override {
    auto sourceOp =
        dyn_cast_if_present<CastOpT>(transferOp.getOperand().getDefiningOp());
    if (!sourceOp) {
      return rewriter.notifyMatchFailure(
          transferOp, "source op not available or does not match");
    }
    Value originValue = sourceOp.getSource();
    ValueRange originDims = sourceOp.getSourceDims();
    auto newOp =
        rewriter.create<TransferOpT>(transferOp.getLoc(), originValue,
                                     originDims, transferOp.getTargetAttr());
    IRMapping mapper;
    mapper.map(originValue, newOp.getResult());
    rewriter.setInsertionPointAfter(newOp);
    auto clonedOp =
        cast<CastOpT>(rewriter.clone(*sourceOp.getOperation(), mapper));
    rewriter.replaceAllUsesExcept(transferOp.getResult(), clonedOp.getResult(),
                                  newOp);
    rewriter.eraseOp(transferOp);
    return success();
  }
};

} // namespace

void TensorBarrierOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                  MLIRContext *context) {
  results.add<SinkCastLikeOpAcrossTransfer<IREE::Flow::TensorBarrierOp,
                                           IREE::Flow::TensorBitCastOp>>(
      context);
  results.add<SinkCastLikeOpAcrossTransfer<IREE::Flow::TensorBarrierOp,
                                           IREE::Flow::TensorReshapeOp>>(
      context);
}

//===----------------------------------------------------------------------===//
// flow.tensor.transfer
//===----------------------------------------------------------------------===//

namespace {

// Attempts to identify trivial cases where we locally recognize that a tensor
// is transferred to the same context it's already on. This does not look across
// control flow edges or globals and is mostly for simplifying IR that may come
// in with a transfer on every single tensor.
struct ElideRedundantTransfer : public OpRewritePattern<TensorTransferOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(TensorTransferOp op,
                                PatternRewriter &rewriter) const override {
    auto baseValue =
        IREE::Util::TiedOpInterface::findTiedBaseValue(op.getOperand());
    if (auto transferOp = dyn_cast_if_present<IREE::Flow::TensorTransferOp>(
            baseValue.getDefiningOp())) {
      if (transferOp.getTarget() == op.getTarget()) {
        rewriter.replaceOp(op, op.getOperand());
        return success();
      }
    }
    return failure();
  }
};

// Attempts to identify trivial case of chained transfer ops (A -> B -> C) and
// rewrite it as (A -> C). Writes it as A -> B and A -> C relying on dead code
// elimination to remove the unused A -> B transfer.
struct ElideIntermediateTransfer final : OpRewritePattern<TensorTransferOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(TensorTransferOp targetTransferOp,
                                PatternRewriter &rewriter) const override {
    auto sourceTransferOp = dyn_cast_if_present<IREE::Flow::TensorTransferOp>(
        targetTransferOp.getOperand().getDefiningOp());
    if (!sourceTransferOp) {
      return failure();
    }
    if (sourceTransferOp.getTarget() == targetTransferOp.getTarget()) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<IREE::Flow::TensorTransferOp>(
        targetTransferOp, targetTransferOp->getResultTypes(),
        sourceTransferOp.getOperand(), targetTransferOp.getOperandDims(),
        targetTransferOp.getTarget());
    return success();
  }
};

} // namespace

void TensorTransferOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                   MLIRContext *context) {
  results.add<ElideRedundantTransfer>(context);
  results.add<ElideIntermediateTransfer>(context);
  results.add<SinkCastLikeOpAcrossTransfer<IREE::Flow::TensorTransferOp,
                                           IREE::Flow::TensorBitCastOp>>(
      context);
  results.add<SinkCastLikeOpAcrossTransfer<IREE::Flow::TensorTransferOp,
                                           IREE::Flow::TensorReshapeOp>>(
      context);
}

//===----------------------------------------------------------------------===//
// flow.tensor.slice
//===----------------------------------------------------------------------===//

// Slices tensor from start to (start + length) exclusively at dim.
static ElementsAttr tensorSlice(ElementsAttr tensor, uint64_t dim,
                                uint64_t start, uint64_t length) {
  auto tensorType = cast<ShapedType>(tensor.getType());
  auto shape = llvm::to_vector(tensorType.getShape());
  if (length == shape[dim]) {
    // No need to slice.
    return tensor;
  }
  auto outputShape = shape;
  outputShape[dim] = length;
  auto outputType =
      RankedTensorType::get(outputShape, getElementTypeOrSelf(tensor));
  llvm::SmallVector<Attribute> newContents;
  newContents.reserve(outputType.getNumElements());
  auto valuesBegin = tensor.getValues<Attribute>().begin();
  int64_t step =
      std::accumulate(shape.rbegin(), shape.rbegin() + shape.size() - dim,
                      /*init=*/1, /*op=*/std::multiplies<int64_t>());
  int64_t num = length * step / shape[dim];
  for (int64_t offset = step / shape[dim] * start,
               numElements = tensorType.getNumElements();
       offset < numElements; offset += step) {
    newContents.append(valuesBegin + offset, valuesBegin + offset + num);
  }
  return DenseElementsAttr::get(outputType, newContents);
}

OpFoldResult TensorSliceOp::fold(FoldAdaptor operands) {
  if (llvm::count(operands.getOperands(), nullptr) == 0) {
    // Ignore DenseResources for now and do not perfom folding on them.
    if (isa<DenseResourceElementsAttr>(operands.getSource())) {
      return {};
    }
    // Fully constant arguments so we can perform the slice here.
    auto tensor = llvm::cast<ElementsAttr>(operands.getSource());
    int64_t rank = llvm::cast<ShapedType>(getSource().getType()).getRank();
    auto start =
        llvm::map_to_vector(operands.getStartIndices(), [](Attribute value) {
          return llvm::cast<IntegerAttr>(value).getValue().getZExtValue();
        });
    auto length =
        llvm::map_to_vector(operands.getLengths(), [](Attribute value) {
          return llvm::cast<IntegerAttr>(value).getValue().getZExtValue();
        });
    for (int64_t dim = 0; dim < rank; ++dim) {
      tensor = tensorSlice(tensor, dim, start[dim], length[dim]);
    }
    return tensor;
  }
  return {};
}

void TensorSliceOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                MLIRContext *context) {
  // TODO(benvanik): canonicalize multiple slices (traverse upward through ssa).
  results.insert<ReplaceOpIfTensorOperandZeroElements<TensorSliceOp, 0>>(
      context);
  results.insert<ReplaceOpIfTensorResultZeroElements<TensorSliceOp, 0>>(
      context);
  results.insert<ReplaceOpIfTensorOperandEmpty<TensorSliceOp, 0, 0>>(context);
}

//===----------------------------------------------------------------------===//
// flow.tensor.update
//===----------------------------------------------------------------------===//

static ElementsAttr tensorUpdate(ElementsAttr update, ElementsAttr target,
                                 ArrayRef<Attribute> startIndicesAttrs) {
  auto updateType = llvm::cast<ShapedType>(update.getType());
  auto targetType = llvm::cast<ShapedType>(target.getType());
  // If either target or update has zero element, then no update happens.
  if (updateType.getNumElements() == 0 || targetType.getNumElements() == 0) {
    return target;
  }

  int64_t rank = targetType.getRank();
  // If target is scalar, update is also scalar and is the new content.
  if (rank == 0) {
    return update;
  }

  auto startIndex = llvm::map_to_vector(startIndicesAttrs, [](Attribute value) {
    return llvm::cast<IntegerAttr>(value).getValue().getZExtValue();
  });
  auto targetValues = llvm::to_vector(target.getValues<Attribute>());
  // target indices start from startIndicesAttrs and update indices start from
  // all zeros.
  llvm::SmallVector<uint64_t> targetIndex(startIndex);
  llvm::SmallVector<uint64_t> updateIndex(rank, 0);
  int64_t numElements = updateType.getNumElements();
  while (numElements--) {
    targetValues[getFlattenedIndex(targetType, targetIndex)] =
        update.getValues<Attribute>()[updateIndex];
    // Increment the index at last dim.
    ++updateIndex.back();
    ++targetIndex.back();
    // If the index in dim j exceeds dim size, reset dim j and
    // increment dim (j-1).
    for (int64_t j = rank - 1;
         j >= 0 && updateIndex[j] >= updateType.getDimSize(j); --j) {
      updateIndex[j] = 0;
      targetIndex[j] = startIndex[j];
      if (j - 1 >= 0) {
        ++updateIndex[j - 1];
        ++targetIndex[j - 1];
      }
    }
  }
  return DenseElementsAttr::get(targetType, targetValues);
}

OpFoldResult TensorUpdateOp::fold(FoldAdaptor operands) {
  bool allIndicesConstant =
      llvm::count(operands.getStartIndices(), nullptr) == 0;
  if (operands.getUpdate() && operands.getTarget() && allIndicesConstant) {
    // Fully constant arguments so we can perform the update here.
    return tensorUpdate(llvm::cast<ElementsAttr>(operands.getUpdate()),
                        llvm::cast<ElementsAttr>(operands.getTarget()),
                        operands.getStartIndices());
  } else {
    // Replace the entire tensor when the sizes match.
    auto updateType = llvm::cast<ShapedType>(getUpdate().getType());
    auto targetType = llvm::cast<ShapedType>(getTarget().getType());
    if (updateType.hasStaticShape() && targetType.hasStaticShape() &&
        updateType == targetType) {
      return getUpdate();
    }
  }
  return {};
}

namespace {

// When the target tensor is a result of a tensor.cast operation, the op needs
// to be updated to use the source of the cast as the target tensor.
struct FoldTensorUpdateOpWithCasts : public OpRewritePattern<TensorUpdateOp> {
  using OpRewritePattern<TensorUpdateOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(TensorUpdateOp updateOp,
                                PatternRewriter &rewriter) const override {
    auto targetCastOp = updateOp.getTarget().getDefiningOp<tensor::CastOp>();
    auto updateCastOp = updateOp.getUpdate().getDefiningOp<tensor::CastOp>();
    if (!targetCastOp && !updateCastOp)
      return failure();
    Value target = (targetCastOp ? cast<Value>(targetCastOp.getSource())
                                 : cast<Value>(updateOp.getTarget()));
    Value update = (updateCastOp ? cast<Value>(updateCastOp.getSource())
                                 : cast<Value>(updateOp.getUpdate()));
    auto newOp = rewriter.create<TensorUpdateOp>(
        updateOp.getLoc(), target.getType(), target,
        refreshDimsOnTypeChange(updateOp, updateOp.getTarget().getType(),
                                target.getType(), updateOp.getTargetDims(),
                                rewriter),
        updateOp.getStartIndices(), update,
        refreshDimsOnTypeChange(updateOp, updateOp.getUpdate().getType(),
                                update.getType(), updateOp.getUpdateDims(),
                                rewriter));
    rewriter.replaceOpWithNewOp<tensor::CastOp>(
        updateOp, updateOp.getResult().getType(), newOp.getResult());
    return success();
  }
};

struct ReplaceOpIfTensorUpdateOperandZeroElements
    : public OpRewritePattern<TensorUpdateOp> {
  using OpRewritePattern<TensorUpdateOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(TensorUpdateOp op,
                                PatternRewriter &rewriter) const override {
    auto operand = op.getUpdate();
    if (!isTensorOperandZeroElements(operand))
      return failure();
    rewriter.replaceOp(op, op.getTarget());
    return success();
  }
};

} // namespace

void TensorUpdateOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                 MLIRContext *context) {
  results.insert<FoldTensorUpdateOpWithCasts>(context);
  // target:
  results.insert<ReplaceOpIfTensorOperandZeroElements<TensorUpdateOp, 0>>(
      context);
  // update:
  results.insert<ReplaceOpIfTensorUpdateOperandZeroElements>(context);
}

//===----------------------------------------------------------------------===//
// flow.channel.split
//===----------------------------------------------------------------------===//

void ChannelSplitOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                 MLIRContext *context) {
  results.insert<ElideUnusedOp<ChannelSplitOp>>(context);
}

} // namespace mlir::iree_compiler::IREE::Flow

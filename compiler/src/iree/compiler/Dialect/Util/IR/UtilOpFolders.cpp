// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <numeric>

#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::iree_compiler::IREE::Util {

//===----------------------------------------------------------------------===//
// util.cast
//===----------------------------------------------------------------------===//

OpFoldResult CastOp::fold(FoldAdaptor operands) {
  if (auto castOp = dyn_cast_or_null<CastOp>(getOperand().getDefiningOp())) {
    if (castOp.getOperand().getType() == getResult().getType()) {
      return castOp.getOperand();
    }
  }
  return {};
}

namespace {

/// Folds cast ops into the result of other ops.
/// Only safe to apply to ops that don't care about their types.
struct FoldCastIntoNullOp : public OpRewritePattern<CastOp> {
  using OpRewritePattern<CastOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(CastOp castOp,
                                PatternRewriter &rewriter) const override {
    auto nullOp = dyn_cast_or_null<NullOp>(castOp.getOperand().getDefiningOp());
    if (!nullOp)
      return failure();
    rewriter.replaceOpWithNewOp<NullOp>(castOp, castOp.getResult().getType());
    return success();
  }
};

} // namespace

void CastOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.add<FoldCastIntoNullOp>(context);
}

//===----------------------------------------------------------------------===//
// util.cmp.eq
//===----------------------------------------------------------------------===//

OpFoldResult CmpEQOp::fold(FoldAdaptor operands) {
  auto makeBool = [&](bool value) {
    return IntegerAttr::get(IntegerType::get(getContext(), 1), value ? 1 : 0);
  };
  if (getLhs() == getRhs()) {
    // SSA values are exactly the same.
    return makeBool(true);
  } else if (operands.getLhs() && operands.getRhs() &&
             operands.getLhs() == operands.getRhs()) {
    // Folded attributes are equal but may come from separate ops.
    return makeBool(true);
  }
  // TODO(benvanik): we could add some interfaces for comparing, but this is
  // likely good enough for now.
  return {};
}

//===----------------------------------------------------------------------===//
// util.cmp.ne
//===----------------------------------------------------------------------===//

OpFoldResult CmpNEOp::fold(FoldAdaptor operands) {
  auto makeBool = [&](bool value) {
    return IntegerAttr::get(IntegerType::get(getContext(), 1), value ? 1 : 0);
  };
  if (getLhs() == getRhs()) {
    // SSA values are exactly the same.
    return makeBool(false);
  } else if (operands.getLhs() && operands.getRhs() &&
             operands.getLhs() == operands.getRhs()) {
    // Folded attributes are equal but may come from separate ops.
    return makeBool(false);
  }
  // TODO(benvanik): we could add some interfaces for comparing, but this is
  // likely good enough for now.
  return {};
}

//===----------------------------------------------------------------------===//
// util.range.min/max
//===----------------------------------------------------------------------===//

static int64_t xmin(int64_t a, int64_t b) { return std::min(a, b); }
static int64_t xmax(int64_t a, int64_t b) { return std::max(a, b); }

template <int64_t initialValue, int64_t expr(int64_t, int64_t)>
static OpFoldResult foldRangeOp(Type type, ValueRange operands,
                                ArrayRef<Attribute> attrOperands) {
  // One operand is a pass-through.
  if (operands.size() == 1) {
    return operands.front();
  }

  // If all operands are constant then fold into a constant.
  int64_t value = initialValue;
  for (auto operand : attrOperands) {
    auto intValue = llvm::dyn_cast_if_present<IntegerAttr>(operand);
    if (!intValue)
      return {};
    value = expr(value, intValue.getValue().getSExtValue());
  }
  return IntegerAttr::get(type, value);
}

OpFoldResult RangeMinOp::fold(FoldAdaptor operands) {
  return foldRangeOp<INT64_MAX, xmin>(getType(), this->getOperands(),
                                      operands.getOperands());
}

OpFoldResult RangeMaxOp::fold(FoldAdaptor operands) {
  return foldRangeOp<INT64_MIN, xmax>(getType(), this->getOperands(),
                                      operands.getOperands());
}

namespace {

// Replaces util.range.min/max ops with the builtin min/max ops when possible.
//
// Example:
//  %min = util.range.min %0, %1 : index
// ->
//  %min = arith.minui %0, %1 : index
template <typename RangeOpT, typename StdOpT>
struct ExpandSimpleRangeOp : public OpRewritePattern<RangeOpT> {
  using OpRewritePattern<RangeOpT>::OpRewritePattern;
  LogicalResult matchAndRewrite(RangeOpT op,
                                PatternRewriter &rewriter) const override {
    if (op.getOperands().size() == 1) {
      rewriter.replaceOp(op, {op.getOperands().front()});
      return success();
    } else if (op.getOperands().size() == 2) {
      rewriter.replaceOpWithNewOp<StdOpT>(op, op.getOperands().front(),
                                          op.getOperands().back());
      return success();
    }
    return failure();
  }
};

// Simplifies min/max ops by folding constants and deduplicating values.
//
// Example:
//  %min = util.range.min %0, %c1, %c2, %0, %1
// ->
//  %min = util.range.min %c1, %0, %1
template <typename OpT, int64_t initialValue, int64_t expr(int64_t, int64_t)>
struct SimplifyUniformRangeOp : public OpRewritePattern<OpT> {
  using OpRewritePattern<OpT>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpT op,
                                PatternRewriter &rewriter) const override {
    SetVector<Value> operands;
    int64_t constantValue = initialValue;
    for (auto operand : op.getOperands()) {
      APInt constantInt;
      if (matchPattern(operand, m_ConstantInt(&constantInt))) {
        // Constant value.
        constantValue = expr(constantValue, constantInt.getSExtValue());
      } else {
        // Dynamic value.
        operands.insert(operand);
      }
    }
    if (operands.size() + (constantValue != initialValue ? 1 : 0) ==
        op.getOperands().size()) {
      // No change in operand count.
      return failure();
    }
    if (constantValue != initialValue) {
      operands.insert(rewriter.create<arith::ConstantOp>(
          op.getLoc(), op.getResult().getType(),
          rewriter.getIntegerAttr(op.getResult().getType(), constantValue)));
    }
    rewriter.replaceOpWithNewOp<OpT>(op, op.getResult().getType(),
                                     operands.takeVector());
    return success();
  }
};

} // namespace

void RangeMinOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                             MLIRContext *context) {
  results.insert<ExpandSimpleRangeOp<RangeMinOp, arith::MinUIOp>>(context);
  results.insert<SimplifyUniformRangeOp<RangeMinOp, INT64_MAX, xmin>>(context);
}

void RangeMaxOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                             MLIRContext *context) {
  results.insert<ExpandSimpleRangeOp<RangeMaxOp, arith::MaxUIOp>>(context);
  results.insert<SimplifyUniformRangeOp<RangeMaxOp, INT64_MIN, xmax>>(context);
}

//===----------------------------------------------------------------------===//
// util.range.extents
//===----------------------------------------------------------------------===//

static Value makeRangeEnd(Location loc, Value offset, Value length, Value one,
                          OpBuilder &builder) {
  return builder.create<arith::SubIOp>(
      loc, builder.create<arith::AddIOp>(loc, offset, length), one);
}
static Value makeRangeEnd(Location loc, Value offset, Value length,
                          OpBuilder &builder) {
  return makeRangeEnd(
      loc, offset, length,
      builder.create<arith::ConstantOp>(
          loc, offset.getType(), builder.getIntegerAttr(offset.getType(), 1)),
      builder);
}

namespace {

struct FoldConstantRanges : public OpRewritePattern<RangeExtentsOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(RangeExtentsOp op,
                                PatternRewriter &rewriter) const override {
    // Build a constant range for all we find and preserve the dynamic pairs.
    SmallVector<Value> offsets;
    SmallVector<Value> lengths;
    offsets.reserve(op.getOffsets().size());
    lengths.reserve(op.getLengths().size());
    int64_t constantMin = INT64_MAX;
    int64_t constantMax = INT64_MIN;
    for (auto [offset, length] :
         llvm::zip_equal(op.getOffsets(), op.getLengths())) {
      APInt rangeOffset, rangeLength;
      if (matchPattern(offset, m_ConstantInt(&rangeOffset)) &&
          matchPattern(length, m_ConstantInt(&rangeLength))) {
        // Both offset and length are constant so we can fold.
        constantMin = std::min(constantMin, rangeOffset.getSExtValue());
        constantMax = std::max(constantMax,
                               (rangeOffset + rangeLength - 1).getSExtValue());
      } else {
        // Dynamic value that we'll preserve.
        offsets.push_back(offset);
        lengths.push_back(length);
      }
    }
    if (offsets.size() == op.getOffsets().size())
      return failure();

    // Preserve dynamic ranges.
    Value min;
    Value max;
    if (!offsets.empty()) {
      auto newOp = rewriter.create<RangeExtentsOp>(
          op.getLoc(), op.getMin().getType(), op.getMax().getType(), offsets,
          lengths);
      min = newOp.getMin();
      max = newOp.getMax();
    }

    // Min/max with constant ranges. This allows for normal folding to happen
    // downstream of the op.
    auto constantMinOp = rewriter.create<arith::ConstantOp>(
        op.getLoc(), op.getMin().getType(),
        rewriter.getIntegerAttr(op.getMin().getType(), constantMin));
    auto constantMaxOp = rewriter.create<arith::ConstantOp>(
        op.getLoc(), op.getMax().getType(),
        rewriter.getIntegerAttr(op.getMax().getType(),
                                constantMax - constantMin + 1));
    min = min ? rewriter.create<arith::MinUIOp>(op.getLoc(), min, constantMinOp)
                    .getResult()
              : constantMinOp.getResult();
    max = max ? rewriter.create<arith::MaxUIOp>(op.getLoc(), max, constantMaxOp)
                    .getResult()
              : constantMaxOp.getResult();

    rewriter.replaceOp(op, {min, max});
    return success();
  }
};

struct ExpandSimpleRangeExtentsOp : public OpRewritePattern<RangeExtentsOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(RangeExtentsOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value minValue, maxValue;
    if (op.getOffsets().size() == 1) {
      // Single range folds to the min/max of that one range.
      minValue = op.getOffsets().front();
      maxValue = makeRangeEnd(loc, op.getOffsets().front(),
                              op.getLengths().front(), rewriter);
    } else if (op.getOffsets().size() == 2) {
      // Two ranges turn into min/max.
      minValue = rewriter.create<arith::MinUIOp>(loc, op.getOffsets().front(),
                                                 op.getOffsets().back());
      auto one = rewriter.create<arith::ConstantOp>(
          loc, op.getMin().getType(),
          rewriter.getIntegerAttr(op.getMin().getType(), 1));
      auto endLhs = makeRangeEnd(loc, op.getOffsets().front(),
                                 op.getLengths().front(), one, rewriter);
      auto endRhs = makeRangeEnd(loc, op.getOffsets().back(),
                                 op.getLengths().back(), one, rewriter);
      maxValue = rewriter.create<arith::MaxUIOp>(loc, endLhs, endRhs);
    }
    if (!minValue || !maxValue)
      return failure();
    rewriter.replaceOp(op, {minValue, maxValue});
    return success();
  }
};

struct DeduplicateRangeExtentsOp : public OpRewritePattern<RangeExtentsOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(RangeExtentsOp op,
                                PatternRewriter &rewriter) const override {
    // First filter out any pure duplicates. Note SetVector so order is
    // preserved.
    using Range = std::tuple<Value, Value>;
    SetVector<Range> ranges;
    for (auto range : llvm::zip_equal(op.getOffsets(), op.getLengths())) {
      ranges.insert(range);
    }
    if (ranges.size() == op.getOffsets().size())
      return failure();

    // Recreate with the deduplicated ranges.
    SmallVector<Value> offsets;
    SmallVector<Value> lengths;
    offsets.reserve(ranges.size());
    lengths.reserve(ranges.size());
    for (auto [offset, length] : ranges) {
      offsets.push_back(offset);
      lengths.push_back(length);
    }
    rewriter.replaceOpWithNewOp<RangeExtentsOp>(
        op, op.getMin().getType(), op.getMax().getType(), offsets, lengths);
    return success();
  }
};

} // namespace

void RangeExtentsOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                 MLIRContext *context) {
  // TODO(benvanik): extract ranges with common offsets or lengths and move them
  // to min/max ops where they have a better chance of folding.
  results.insert<FoldConstantRanges>(context);
  results.insert<ExpandSimpleRangeExtentsOp>(context);
  results.insert<DeduplicateRangeExtentsOp>(context);
}

//===----------------------------------------------------------------------===//
// util.align
//===----------------------------------------------------------------------===//

// TODO(#5405): add canonicalizers that reach further in the IR or a dedicated
// pass for full potential-value-set analysis.

// Returns true if |value| is definitely aligned to at least |alignment|.
// Recursively checks up the source of the value to see if we can trivially
// prove the alignment either directly matches (when dynamic) or is >= the
// specified |alignment|. This does not walk across blocks or calls but catches
// a large majority of the cases we generate ourselves from packing/allocation.
static bool isAlignedTo(Value value, Value alignment) {
  APInt staticValue;
  bool hasStaticValue = matchPattern(value, m_ConstantInt(&staticValue));
  APInt staticAlignment;
  bool hasStaticAlignment =
      matchPattern(alignment, m_ConstantInt(&staticAlignment));
  if (hasStaticValue && hasStaticAlignment) {
    // If this value is itself a multiple of the alignment then we can fold.
    if (staticValue.urem(staticAlignment).isZero()) {
      return true; // value % alignment == 0
    }
  }

  // If the value is produced by an align op we can check that.
  if (auto sourceAlignOp = value.getDefiningOp<IREE::Util::AlignOp>()) {
    // Check for same exact alignment - even if dynamic.
    if (sourceAlignOp.getAlignment() == alignment)
      return true;

    // If the alignments are constant we can compare them inline.
    APInt sourceAlignment;
    if (hasStaticAlignment && matchPattern(sourceAlignOp.getAlignment(),
                                           m_ConstantInt(&sourceAlignment))) {
      if (sourceAlignment.uge(staticAlignment) &&
          std::gcd(sourceAlignment.getZExtValue(),
                   staticAlignment.getZExtValue()) ==
              staticAlignment.getZExtValue()) {
        return true; // source alignment is >= our alignment
      }
    }

    // Recurse and check the alignment on the input to the align; if it was
    // aligned earlier we can rely on that as align will never shrink a value.
    return isAlignedTo(sourceAlignOp.getValue(), alignment);
  }

  // Affine apply ops producing the value to be aligned usually include
  // alignment already.
  if (auto affineOp = value.getDefiningOp<affine::AffineApplyOp>()) {
    if (hasStaticAlignment) {
      return (affineOp.getAffineMap().getLargestKnownDivisorOfMapExprs() %
              staticAlignment.getZExtValue()) == 0;
    }
  }

  // If we are sourced from add/mul we peephole check to see if what is being
  // added is also aligned. This should be part of a larger pass doing IPO but
  // as the common case is that we align+add+align this is worth having in a
  // folder. This single folder can avoid ever even materializing thousands of
  // ops.
  if (auto sourceAddOp = value.getDefiningOp<arith::AddIOp>()) {
    // Two aligned values added together are still aligned.
    if (isAlignedTo(sourceAddOp.getLhs(), alignment) &&
        isAlignedTo(sourceAddOp.getRhs(), alignment)) {
      return true;
    }
  } else if (auto sourceSubOp = value.getDefiningOp<arith::SubIOp>()) {
    // An aligned value subtracted from an aligned value is still aligned.
    if (isAlignedTo(sourceSubOp.getLhs(), alignment) &&
        isAlignedTo(sourceSubOp.getRhs(), alignment)) {
      return true;
    }
  } else if (auto sourceMulOp = value.getDefiningOp<arith::MulIOp>()) {
    // Two aligned values multiplied together are still aligned.
    if (isAlignedTo(sourceMulOp.getLhs(), alignment) ||
        isAlignedTo(sourceMulOp.getRhs(), alignment)) {
      return true;
    }
  }

  return false;
}

OpFoldResult AlignOp::fold(FoldAdaptor operands) {
  // If aligning an already-aligned value then fold if this is provably a
  // no-op. We can check this for equality even with dynamic alignments.
  if (isAlignedTo(getValue(), getAlignment()))
    return getValue();

  // If values are static we can perform the alignment here.
  APInt staticValue;
  APInt staticAlignment;
  if (matchPattern(getValue(), m_ConstantInt(&staticValue)) &&
      matchPattern(getAlignment(), m_ConstantInt(&staticAlignment))) {
    return IntegerAttr::get(getResult().getType(),
                            align(staticValue.getZExtValue(), staticAlignment));
  }

  return {};
}

//===----------------------------------------------------------------------===//
// util.sizeof
//===----------------------------------------------------------------------===//

OpFoldResult SizeOfOp::fold(FoldAdaptor operands) {
  Type t = getSizedType();
  if (llvm::isa<IntegerType>(t) || llvm::isa<FloatType>(t)) {
    return IntegerAttr::get(IndexType::get(getContext()),
                            getRoundedElementByteWidth(t));
  }
  return {};
}

//===----------------------------------------------------------------------===//
// util.switch
//===----------------------------------------------------------------------===//

OpFoldResult SwitchOp::fold(FoldAdaptor operands) {
  APInt indexValue;
  if (matchPattern(getIndex(), m_ConstantInt(&indexValue))) {
    // Index is constant and we can resolve immediately.
    int64_t index = indexValue.getSExtValue();
    if (index < 0 || index >= getValues().size()) {
      return getDefaultValue();
    }
    return getValues()[index];
  }

  bool allValuesMatch = true;
  for (auto value : getValues()) {
    if (value != getDefaultValue()) {
      allValuesMatch = false;
      break;
    }
  }
  if (allValuesMatch) {
    // All values (and the default) are the same so just return it regardless of
    // the provided index.
    return getDefaultValue();
  }

  return {};
}

//===----------------------------------------------------------------------===//
// Compiler hints
//===----------------------------------------------------------------------===//

namespace {

struct ExpandUnfoldableConstantOp
    : public OpRewritePattern<UnfoldableConstantOp> {
  using OpRewritePattern<IREE::Util::UnfoldableConstantOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(UnfoldableConstantOp op,
                                PatternRewriter &rewriter) const override {
    auto stdConst = rewriter.create<arith::ConstantOp>(
        op.getLoc(), cast<TypedAttr>(op.getValue()));
    rewriter.replaceOpWithNewOp<OptimizationBarrierOp>(op,
                                                       stdConst.getResult());
    return success();
  }
};

} // namespace

void UnfoldableConstantOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<ExpandUnfoldableConstantOp>(context);
}

//===----------------------------------------------------------------------===//
// Globals
//===----------------------------------------------------------------------===//

namespace {

// Deletes empty vm.initializer ops.
struct DropEmptyInitializerOp : public OpRewritePattern<InitializerOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(InitializerOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getBody().getBlocks().size() != 1)
      return failure();
    auto &block = op.getBody().front();
    if (block.empty() || isa<InitializerReturnOp>(block.front())) {
      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }
};

// Inlines constant stores from initializers into the global initializer.
// This is not strictly required but can help our initialization code perform
// more efficient initialization of large numbers of primitive values.
struct InlineConstantGlobalInitializer
    : public OpRewritePattern<InitializerOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(InitializerOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Operation *> deadOps;
    op.walk([&](GlobalStoreOpInterface storeOp) {
      Attribute valueAttr;
      if (!matchPattern(storeOp.getStoredGlobalValue(),
                        m_Constant(&valueAttr))) {
        return;
      }
      auto globalOp =
          SymbolTable::lookupNearestSymbolFrom<IREE::Util::GlobalOpInterface>(
              storeOp->getParentOp(), storeOp.getGlobalAttr());
      rewriter.updateRootInPlace(
          globalOp, [&]() { globalOp.setGlobalInitialValue(valueAttr); });

      deadOps.push_back(storeOp);
    });
    if (deadOps.empty())
      return failure();
    for (auto deadOp : deadOps)
      rewriter.eraseOp(deadOp);
    return success();
  }
};

} // namespace

void InitializerOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                MLIRContext *context) {
  results.insert<DropEmptyInitializerOp, InlineConstantGlobalInitializer>(
      context);
}

void GlobalOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {}

namespace {

/// Turns util.global.address -> util.global.load.indirect into a direct load.
class PropagateGlobalLoadAddress
    : public OpRewritePattern<GlobalLoadIndirectOp> {
  using OpRewritePattern::OpRewritePattern;

public:
  LogicalResult matchAndRewrite(GlobalLoadIndirectOp op,
                                PatternRewriter &rewriter) const override {
    if (auto addressOp = dyn_cast_or_null<GlobalAddressOpInterface>(
            op.getGlobal().getDefiningOp())) {
      rewriter.replaceOpWithNewOp<GlobalLoadOp>(op, op.getResult().getType(),
                                                addressOp.getGlobalAttr());
      return success();
    }
    return failure();
  }
};

} // namespace

void GlobalLoadIndirectOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<PropagateGlobalLoadAddress>(context);
}

namespace {

/// Erases util.global.store ops that are no-ops.
/// This can happen if there was a global load, some DCE'd usage, and a
/// store back to the same global: we want to be able to elide the entire load
/// and store.
struct EraseUnusedGlobalStoreOp : public OpRewritePattern<GlobalStoreOp> {
  using OpRewritePattern<GlobalStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GlobalStoreOp op,
                                PatternRewriter &rewriter) const override {
    if (auto loadOp = dyn_cast_or_null<GlobalLoadOpInterface>(
            op.getValue().getDefiningOp())) {
      if (loadOp.getGlobalName() == op.getGlobal()) {
        rewriter.eraseOp(op);
        return success();
      }
    }
    return failure();
  }
};

} // namespace

void GlobalStoreOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                MLIRContext *context) {
  results.insert<EraseUnusedGlobalStoreOp>(context);
}

namespace {

/// Turns util.global.address -> util.global.store.indirect into a direct store.
class PropagateGlobalStoreAddress
    : public OpRewritePattern<GlobalStoreIndirectOp> {
  using OpRewritePattern::OpRewritePattern;

public:
  LogicalResult matchAndRewrite(GlobalStoreIndirectOp op,
                                PatternRewriter &rewriter) const override {
    if (auto addressOp = dyn_cast_or_null<GlobalAddressOpInterface>(
            op.getGlobal().getDefiningOp())) {
      rewriter.replaceOpWithNewOp<GlobalStoreOp>(op, op.getValue(),
                                                 addressOp.getGlobalAttr());
      return success();
    }
    return failure();
  }
};

} // namespace

void GlobalStoreIndirectOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<PropagateGlobalStoreAddress>(context);
}

//===----------------------------------------------------------------------===//
// util.buffer.alloc
//===----------------------------------------------------------------------===//

void BufferAllocOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                MLIRContext *context) {
  // TODO(benvanik): elide if only users are writes and dealloc.
}

//===----------------------------------------------------------------------===//
// util.buffer.subspan
//===----------------------------------------------------------------------===//

OpFoldResult BufferSubspanOp::fold(FoldAdaptor operands) {
  if (getSourceSize() == getResultSize()) {
    // Entire range is covered; return it all.
    return getSource();
  }
  return {};
}

namespace {

// Folds subspan -> subspan to point at the original source buffer with an
// updated range.
struct FoldBufferSubspanOps : public OpRewritePattern<BufferSubspanOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(BufferSubspanOp op,
                                PatternRewriter &rewriter) const override {
    auto parentOp = BufferSubspanOp::findSubspanOp(op.getSource());
    if (!parentOp)
      return failure();
    auto fusedLoc = rewriter.getFusedLoc({parentOp.getLoc(), op.getLoc()});
    auto newOffset = rewriter.createOrFold<arith::AddIOp>(
        fusedLoc, parentOp.getSourceOffset(), op.getSourceOffset());
    auto newOp = rewriter.create<BufferSubspanOp>(
        fusedLoc, parentOp.getSource(), parentOp.getSourceSize(), newOffset,
        op.getResultSize());
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

// Folds subspan ranges into consumer ranges.
//
// Example:
//  %0 = util.buffer.subspan %src[%subspan_offset] ... -> {%subspan_length}
//  %1 = util.buffer.subspan %dst[%subspan_offset] ... -> {%subspan_length}
//  util.buffer.copy %0[%offset], %1[%offset], %length
// ->
//  %new_offset = arith.addi %offset, %subspan_offset
//  util.buffer.copy %src[%new_offset], %dst[%new_offset], %subspan_length
struct FoldBufferSubspanOpsIntoConsumers
    : public OpRewritePattern<BufferSubspanOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(BufferSubspanOp op,
                                PatternRewriter &rewriter) const override {
    bool didUpdateAny = false;
    for (auto &use : llvm::make_early_inc_range(op.getResult().getUses())) {
      auto subrangeOp =
          dyn_cast<IREE::Util::SubrangeOperandOpInterface>(use.getOwner());
      if (!subrangeOp)
        continue;
      didUpdateAny = true;
      rewriter.setInsertionPoint(subrangeOp);
      auto oldRange = subrangeOp.getSubrangeOperand(use.getOperandNumber());
      auto fusedLoc =
          rewriter.getFusedLoc({op.getLoc(), use.getOwner()->getLoc()});
      auto newOffset = rewriter.createOrFold<arith::AddIOp>(
          fusedLoc, op.getSourceOffset(), oldRange.offset);
      auto newRange = SubrangeOperand{op.getSource(), op.getSourceSize(),
                                      newOffset, oldRange.length};
      rewriter.updateRootInPlace(subrangeOp, [&]() {
        subrangeOp.setSubrangeOperand(use.getOperandNumber(), newRange);
      });
    }
    return success(didUpdateAny);
  }
};

// Turns selects of subspans of a buffer into selects of the offset.
// This only works if the subspan sizes match.
//
// Example:
//  %subspan0 = util.buffer.subspan %src[%offset0]
//  %subspan1 = util.buffer.subspan %src[%offset1]
//  %subspan = select %cond, %subspan0, %subspan1 : !util.buffer
// ->
//  %offset = select %cond, %offset0, %offset1 : index
//  %subspan = util.buffer.subspan %src[%offset]
struct SinkSubspanAcrossSelectOps
    : public OpRewritePattern<mlir::arith::SelectOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(mlir::arith::SelectOp op,
                                PatternRewriter &rewriter) const override {
    if (!llvm::isa<IREE::Util::BufferType>(op.getType()))
      return failure();
    auto trueSubspan = dyn_cast_or_null<IREE::Util::BufferSubspanOp>(
        op.getTrueValue().getDefiningOp());
    auto falseSubspan = dyn_cast_or_null<IREE::Util::BufferSubspanOp>(
        op.getFalseValue().getDefiningOp());
    if (!trueSubspan || !falseSubspan)
      return failure();
    if (trueSubspan.getSource() != falseSubspan.getSource() ||
        trueSubspan.getResultSize() != falseSubspan.getResultSize()) {
      return failure();
    }
    auto offsetSelectOp = rewriter.create<mlir::arith::SelectOp>(
        op.getLoc(), op.getCondition(), trueSubspan.getSourceOffset(),
        falseSubspan.getSourceOffset());
    rewriter.replaceOpWithNewOp<IREE::Util::BufferSubspanOp>(
        op, op.getResult().getType(), trueSubspan.getSource(),
        trueSubspan.getSourceSize(), offsetSelectOp.getResult(),
        trueSubspan.getResultSize());
    return success();
  }
};

} // namespace

void BufferSubspanOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                  MLIRContext *context) {
  results.insert<FoldBufferSubspanOps>(context);
  results.insert<FoldBufferSubspanOpsIntoConsumers>(context);
  results.insert<SinkSubspanAcrossSelectOps>(context);
}

//===----------------------------------------------------------------------===//
// util.buffer.size
//===----------------------------------------------------------------------===//

OpFoldResult BufferSizeOp::fold(FoldAdaptor operands) {
  // Try to find the size in the use-def chain.
  // If it's out of the local scope we'll need IPO to help out.
  // During A->B->C dialect conversion, the type may not be legal so be
  // defensive.
  auto operand = getOperand();
  if (auto sizeAwareType =
          dyn_cast<IREE::Util::SizeAwareTypeInterface>(operand.getType())) {
    Operation *op = this->getOperation();
    if (auto sizeValue = sizeAwareType.findSizeValue(operand, op->getBlock(),
                                                     Block::iterator(op))) {
      return sizeValue;
    }
  }

  // If the source is a constant then we can calculate that immediately.
  if (auto constantOp = dyn_cast_or_null<IREE::Util::BufferConstantOp>(
          operand.getDefiningOp())) {
    if (auto storageAttr = dyn_cast_if_present<IREE::Util::SizedStorageAttr>(
            constantOp.getValue())) {
      return IntegerAttr::get(IndexType::get(storageAttr.getContext()),
                              storageAttr.getStorageSize());
    }
  }

  return {};
}

namespace {

// Propagates buffer sizes through select ops by selecting on the sizes of the
// select operands.
//
// Example:
//  %a = util.buffer... : !util.buffer{%a_sz}
//  %b = util.buffer... : !util.buffer{%b_sz}
//  %c = select %cond, %a, %b : !util.buffer
//  %c_sz = util.buffer.size %c : !util.buffer
// ->
//  %c = select %cond, %a, %b : !util.buffer
//  %c_sz = select %cond, %a_sz, %b_sz : index
struct SelectBufferSizeOp : public OpRewritePattern<BufferSizeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(BufferSizeOp op,
                                PatternRewriter &rewriter) const override {
    auto selectOp = op.getOperand().getDefiningOp<mlir::arith::SelectOp>();
    if (!selectOp)
      return failure();
    auto trueSize = rewriter.createOrFold<IREE::Util::BufferSizeOp>(
        op.getLoc(), selectOp.getTrueValue());
    auto falseSize = rewriter.createOrFold<IREE::Util::BufferSizeOp>(
        op.getLoc(), selectOp.getFalseValue());
    rewriter.replaceOpWithNewOp<mlir::arith::SelectOp>(
        op, selectOp.getCondition(), trueSize, falseSize);
    return success();
  }
};

} // namespace

void BufferSizeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                               MLIRContext *context) {
  results.insert<SelectBufferSizeOp>(context);
}

//===----------------------------------------------------------------------===//
// util.buffer.storage
//===----------------------------------------------------------------------===//

namespace {

// Folds subspan ranges into storage ranges.
//
// Example:
//  %0 = util.buffer.subspan %src[%subspan_offset] ... -> {%subspan_length}
//  %storage, %offset = util.buffer.storage %0
// ->
//  %storage, %raw_offset = util.buffer.storage %src
//  %offset = arith.addi %raw_offset, %subspan_offset
struct FoldSubspansIntoStorageOp : public OpRewritePattern<BufferStorageOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(BufferStorageOp op,
                                PatternRewriter &rewriter) const override {
    auto subspanOp = BufferSubspanOp::findSubspanOp(op.getOperand());
    if (!subspanOp)
      return failure();
    auto fusedLoc = rewriter.getFusedLoc({subspanOp.getLoc(), op.getLoc()});
    rewriter.setInsertionPointAfter(op);
    auto newOffset = rewriter.createOrFold<arith::AddIOp>(
        fusedLoc, subspanOp.getSourceOffset(), op.getOffset());
    rewriter.updateRootInPlace(op, [&]() {
      op.getOperandMutable().assign(subspanOp.getSource());
      op.getOperandSizeMutable().assign(subspanOp.getSourceSize());
      SmallPtrSet<Operation *, 2> exceptions;
      exceptions.insert(op);
      if (auto newOffsetOp = newOffset.getDefiningOp()) {
        exceptions.insert(newOffsetOp);
      }
      op.getOffset().replaceAllUsesExcept(newOffset, exceptions);
    });
    return success();
  }
};

} // namespace

void BufferStorageOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                  MLIRContext *context) {
  results.insert<FoldSubspansIntoStorageOp>(context);
}

//===----------------------------------------------------------------------===//
// util.buffer.load
//===----------------------------------------------------------------------===//

OpFoldResult BufferLoadOp::fold(FoldAdaptor operands) {
  // TODO(benvanik): if source is a constant then perform the load.
  return {};
}

} // namespace mlir::iree_compiler::IREE::Util

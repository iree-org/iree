// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <numeric>

#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Util/IR/ClosureOpUtils.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringExtras.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Stream {

//===----------------------------------------------------------------------===//
// Utilities shared across patterns
//===----------------------------------------------------------------------===//

// Returns the stream.yield op in |block| if it is the only op.
//
// Example:
//  stream.async.concurrent ... {
//    stream.yield
//  }
static std::optional<IREE::Stream::YieldOp> getYieldIfOnlyOp(Block &block) {
  if (block.empty()) return std::nullopt;
  if (&block.front() != &block.back()) return std::nullopt;
  auto yieldOp = dyn_cast<IREE::Stream::YieldOp>(block.back());
  if (yieldOp) return yieldOp;
  return std::nullopt;
}

// Various patterns try to sink ops, and in case of uses in multiple blocks
// they might be sunk to the end of a block. When multiple such ops are being
// sunk, they can "fight" over who is at the end of the block, resulting in
// infinite pattern recursion. To avoid this, we need to collectively know
// across patterns which ops are liable to be sunk that way.
static bool isSinkCandidate(Operation *op) {
  return isa<AsyncSplatOp, AsyncAllocaOp, TimepointAwaitOp>(op);
}

// Determine if sinking |toBeSunkOp| before |targetOp| won't result in an
// unstable oscillation across patterns. Oscillations can occur if there
// are multiple ops inserted before a single op as insertion order based on
// canonicalization is undefined.
//
// Example:
//   %0 = op.a
//   %1 = op.b
//   %2 = op.c %0, %1
// If %0 and %1 are sunk to %2 the ordering will depend on which sink pattern
// runs first and each of the patterns will fight trying to sink lower than the
// other. As long as sinking only happens when this function returns `true`,
// then the sinking across patterns will reach a fixed-point.
static bool canStablySinkTo(Operation *toBeSunkOp, Operation *targetOp) {
  // Stably sinking implies that other sinking won't "fight" with this
  // sinking. This is obviously not possible in an open pattern ecosystem,
  // but for the purpose of this function, we assume that all sinking patterns
  // that we are concerned with are the other patterns in the `stream` dialect.
  //
  // In typical usage, this function will result in various patterns sinking
  // their relevant ops before `targetOp`. This results in a sequence of
  // sinkable ops before `targetOp`. This is fine, until we start to sink
  // them again, which can result in "fighting". We detect that scenario
  // by seeing if all the ops between `toBeSunkOp` and `targetOp` might be sunk
  // again.
  //
  // To prove that this function results in sinking that reaches a fixed-point,
  // we can design a potential function `f(the_module) -> int`, and show that it
  // decreases strictly monotonically with each sinking operation (and cannot go
  // below 0). In particular, we choose the following function: `f(the_module) =
  // sum(g(op) for op in the_module)`, where `g(op) -> int` gives the distance
  // between op's current location and the latest it could appear in the program
  // (infinite, if that location is in another block).
  assert(isSinkCandidate(toBeSunkOp) && "asking to sink a non-sinkable op");

  // If `targetOp` is a terminator, then it might be chosen as a sink location
  // purely for control flow reasons, and not due to use-def chains. This means
  // that if `targetOp` is not a terminator, then we can prune the set of
  // sinkable ops that might fight with `toBeSunkOp` more aggressively by using
  // use-def chains.
  bool allowUseDefPruning = !targetOp->hasTrait<mlir::OpTrait::IsTerminator>();

  // If the sinking operation would be a no-op, then we need to prevent
  // the sinking operation, to avoid infinite pattern applications.
  if (Block::iterator(targetOp) == std::next(Block::iterator(toBeSunkOp)))
    return false;

  // If the sinking is to a different block, then it okay, since for any later
  // sinkings, this reduces the problem to stable sinking within a single
  // block (handled below).
  if (toBeSunkOp->getBlock() != targetOp->getBlock()) return true;

  SmallPtrSet<Operation *, 4> producerOps;
  if (allowUseDefPruning) {
    for (auto operand : targetOp->getOperands()) {
      if (operand.getDefiningOp()) {
        producerOps.insert(operand.getDefiningOp());
      }
    }
  }

  // If any of the ops between `toBeSunkOp` and `targetOp` are known to not
  // fight with this op, then it is stable to sink.
  for (Operation &op : llvm::make_range(Block::iterator(toBeSunkOp),
                                        Block::iterator(targetOp))) {
    // If the intervening op that is not even a sink candidate itself,
    // then it cannot fight.
    if (!isSinkCandidate(&op)) return true;
    // If the op is pruned by use-def chains, then it won't fight.
    if (allowUseDefPruning && !producerOps.contains(&op)) return true;
  }
  return false;
}

// Sinks |op| down to |targetOp|, ensuring that we don't oscillate.
// Returns success if the op was sunk and failure if sinking was not needed.
static LogicalResult sinkOp(Operation *op, Operation *targetOp) {
  if (!canStablySinkTo(op, targetOp)) return failure();
  op->moveBefore(targetOp);
  return success();
}

// Sets |rewriter| to point immediately before the parent execution region.
// Example:
//   %0 =
//   <-- insertion point set to here -->
//   stream.async.execute ... {
//     %1 = op
//   }
static void setInsertionPointToParentExecutionScope(Operation *op,
                                                    PatternRewriter &rewriter) {
  if (auto parentOp = op->getParentOfType<AsyncExecuteOp>()) {
    rewriter.setInsertionPoint(parentOp);
  } else if (auto parentOp = op->getParentOfType<CmdExecuteOp>()) {
    rewriter.setInsertionPoint(parentOp);
  } else {
    assert(false && "must be nested within an execution region");
  }
}

namespace {

// Erases an op if it has no uses.
// This is to support ops that are "pure" but can't be marked as such because
// the MLIR CSE pass would deduplicate them.
template <typename Op>
struct ElideUnusedOp : public OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;
  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter &rewriter) const override {
    if (!op.use_empty()) return failure();
    rewriter.eraseOp(op);
    return success();
  }
};

// Materialize copy-on-write (🐄) ops where required for |rootValue|.
// Only valid in tensor/async ops - don't use with stream.cmd.*.
static bool materializeCOW(Location loc, Value rootValue, OpBuilder &builder) {
  auto valueType = rootValue.getType().dyn_cast<IREE::Stream::ResourceType>();
  if (!valueType) return false;

  // If our rootValue is a constant then we need to ensure that we aren't
  // tied to a constant operand. If we are we need to clone to a
  // non-constant value.
  bool forceClone = valueType.getLifetime() == IREE::Stream::Lifetime::Constant;

  // Identify if we need to insert a copy-on-write clone.
  // We do this per use as a single consuming op may use the result of this
  // multiple times - some tied and some not - and if it has it tied several
  // times each will need its own clone.
  struct TiedUse {
    Operation *user;
    unsigned operandIndex;
    Value value;
  };
  SmallVector<TiedUse> tiedUses;
  unsigned untiedUses = 0;
  for (auto &use : rootValue.getUses()) {
    if (isa<IREE::Stream::TimepointAwaitOp>(use.getOwner())) continue;
    auto tiedOp = dyn_cast<IREE::Util::TiedOpInterface>(use.getOwner());
    bool isTied = tiedOp && tiedOp.isOperandTied(use.getOperandNumber());
    if (isTied) {
      tiedUses.push_back({use.getOwner(), use.getOperandNumber(), rootValue});
    } else {
      ++untiedUses;
    }
  }
  if (tiedUses.empty()) {
    // All uses are as normal capturing SSA values.
    return false;
  } else if (tiedUses.size() == 1 && untiedUses == 0 && !forceClone) {
    // Only one use and it's tied - we've already reserved our results for it.
    return false;
  }

  // Mixed/multiple tied uses. Clone for each tied use but leave the untied
  // ones referencing us.
  IREE::Stream::AffinityAttr sourceAffinity;
  if (auto affinityOp = dyn_cast_or_null<IREE::Stream::AffinityOpInterface>(
          rootValue.getDefiningOp())) {
    sourceAffinity = affinityOp.getAffinity();
  }
  for (auto &tiedUse : tiedUses) {
    auto cloneLoc =
        FusedLoc::get(builder.getContext(), {loc, tiedUse.user->getLoc()});

    builder.setInsertionPoint(tiedUse.user);

    auto sizeAwareType =
        tiedUse.value.getType()
            .template cast<IREE::Util::SizeAwareTypeInterface>();
    auto targetSize =
        sizeAwareType.queryValueSize(cloneLoc, tiedUse.value, builder);

    IREE::Stream::AffinityAttr targetAffinity;
    if (auto affinityOp =
            dyn_cast<IREE::Stream::AffinityOpInterface>(tiedUse.user)) {
      targetAffinity = affinityOp.getAffinity();
    }

    auto cloneOp = builder.create<IREE::Stream::AsyncCloneOp>(
        cloneLoc, tiedUse.value.getType(), tiedUse.value, targetSize,
        targetSize, targetAffinity ? targetAffinity : sourceAffinity);
    tiedUse.user->setOperand(tiedUse.operandIndex, cloneOp.getResult());
  }

  return true;
}

// Materialize copy-on-write (🐄) ops where required.
// This models what a runtime normally does with copy-on-write but uses the
// information we have in the SSA use-def chain to identify ties that write and
// covering reads.
template <typename Op>
struct MaterializeCOW : public OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;
  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter &rewriter) const override {
    bool didChange = false;

    // Handle results of this op (primary use case).
    for (auto result : op->getResults()) {
      didChange = materializeCOW(op.getLoc(), result, rewriter) || didChange;
    }

    return didChange ? success() : failure();
  }
};

// Ties the results of execution region to their operands when the region
// operations are tied throughout the entire body.
//
// Example:
//  %ret:2 = stream.async.execute with(%src as %arg0) -> !stream.resource<*> {
//    %2 = stream.async.dispatch ... (%arg0) -> %arg0
//    stream.yield %2
//  }
// ->
//  %ret:2 = stream.async.execute with(%src as %arg0) -> %src {
//    %2 = stream.async.dispatch ... (%arg0) -> %arg0
//    stream.yield %2
//  }
template <typename Op>
struct TieRegionResults : public OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;
  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter &rewriter) const override {
    assert(op.getRegion().getBlocks().size() == 1 &&
           "only one stream block supported");
    bool didModify = false;
    for (auto yieldOp : op.template getOps<IREE::Stream::YieldOp>()) {
      for (auto result : llvm::enumerate(yieldOp.getResourceOperands())) {
        if (op.getTiedResultOperandIndex(result.index()).has_value()) {
          continue;  // Already tied.
        }
        auto baseValue =
            IREE::Util::TiedOpInterface::findTiedBaseValue(result.value());
        if (auto blockArg = baseValue.template dyn_cast<BlockArgument>()) {
          unsigned operandIndex = blockArg.getArgNumber();
          rewriter.updateRootInPlace(op, [&]() {
            op.setTiedResultOperandIndex(result.index(), operandIndex);
          });
          didModify = true;
        }
      }
    }
    return didModify ? success() : failure();
  }
};

// Adds await dependencies on |newTimepoints| to the op with an optional
// |existingTimepoint| by possibly producing a new timepoint to await.
// This may just pass through the provided timepoint or create a join based on
// the existing await behavior of the op and the new values.
static Value joinAwaitTimepoints(Location loc, Value existingTimepoint,
                                 ArrayRef<Value> newTimepoints,
                                 OpBuilder &builder) {
  if (newTimepoints.empty()) {
    // No new timepoints - preserve existing.
    return existingTimepoint;
  } else if (newTimepoints.size() == 1 && !existingTimepoint) {
    // Adding a single new timepoint.
    return newTimepoints.front();
  }

  // Materialize a join of the new timepoints + the existing (if present).
  SmallVector<Value> joinTimepoints;
  if (existingTimepoint) {
    joinTimepoints.push_back(existingTimepoint);
  }
  llvm::append_range(joinTimepoints, newTimepoints);
  return builder.create<IREE::Stream::TimepointJoinOp>(
      loc, builder.getType<IREE::Stream::TimepointType>(), joinTimepoints);
}

// Elides waits that are known to be immediately resolved.
//
// Example:
//  %0 = stream.timepoint.immediate
//  %1 = stream.resource.alloca await(%0) ...
// ->
//  %1 = stream.resource.alloca ...
template <typename Op>
struct ElideImmediateTimepointWait : public OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;
  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter &rewriter) const override {
    bool isImmediate =
        op.getAwaitTimepoint() && isa_and_nonnull<TimepointImmediateOp>(
                                      op.getAwaitTimepoint().getDefiningOp());
    if (!isImmediate) return failure();
    rewriter.updateRootInPlace(
        op, [&]() { op.getAwaitTimepointMutable().clear(); });
    return success();
  }
};

// Chains operand resources produced by an await to dependent execution regions.
// This elides host waits and allows for device-side wait resolution.
//
// Example:
//  %0 = stream.cmd.execute with(%resource)
//  %1 = stream.timepoint.await %0 => %resource
//  %2 = stream.cmd.execute with(%resource)
// ->
//  %0 = stream.cmd.execute with(%resource)
//  %2 = stream.cmd.execute await(%0) => with(%resource)
template <typename Op>
struct ChainDependentAwaits : public OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;
  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> newTimepoints;
    SmallVector<std::pair<unsigned, Value>> replacements;
    for (auto operand : llvm::enumerate(op.getResourceOperands())) {
      if (auto awaitOp =
              operand.value().template getDefiningOp<TimepointAwaitOp>()) {
        newTimepoints.push_back(awaitOp.getAwaitTimepoint());
        replacements.push_back(std::make_pair(
            operand.index(), awaitOp.getTiedResultOperand(operand.value())));
      }
    }
    if (replacements.empty()) return failure();
    rewriter.updateRootInPlace(op, [&]() {
      auto newTimepoint = joinAwaitTimepoints(
          op.getLoc(), op.getAwaitTimepoint(), newTimepoints, rewriter);
      op.getAwaitTimepointMutable().assign(newTimepoint);
      for (auto replacement : replacements) {
        op.getResourceOperandsMutable()
            .slice(replacement.first, 1)
            .assign(replacement.second);
      }
    });
    return success();
  }
};

}  // namespace

//===----------------------------------------------------------------------===//
// stream.resource.alloc
//===----------------------------------------------------------------------===//

void ResourceAllocOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                  MLIRContext *context) {
  // TODO(benvanik): sink to first user.
}

//===----------------------------------------------------------------------===//
// stream.resource.alloca
//===----------------------------------------------------------------------===//

void ResourceAllocaOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                   MLIRContext *context) {
  // TODO(benvanik): sink to first user.
  // TODO(benvanik): elide if only user is dealloc.
  results.insert<ElideImmediateTimepointWait<ResourceAllocaOp>>(context);
}

//===----------------------------------------------------------------------===//
// stream.resource.dealloca
//===----------------------------------------------------------------------===//

void ResourceDeallocaOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                     MLIRContext *context) {
  // TODO(benvanik): move up to producer of timepoint.
  results.insert<ElideImmediateTimepointWait<ResourceDeallocaOp>>(context);
}

//===----------------------------------------------------------------------===//
// stream.resource.size
//===----------------------------------------------------------------------===//

OpFoldResult ResourceSizeOp::fold(FoldAdaptor operands) {
  auto sizeAwareType =
      getOperand().getType().cast<IREE::Util::SizeAwareTypeInterface>();
  Operation *op = this->getOperation();
  return sizeAwareType.findSizeValue(getOperand(), op->getBlock(),
                                     Block::iterator(op));
}

namespace {

// Propagates resource sizes through select ops by selecting on the sizes of the
// select operands.
//
// Example:
//  %a = stream... : !stream.resource<*>{%a_sz}
//  %b = stream... : !stream.resource<*>{%b_sz}
//  %c = select %cond, %a, %b : !stream.resource<*>
//  %c_sz = stream.resource.size %c : !stream.resource<*>
// ->
//  %c = select %cond, %a, %b : !stream.resource<*>
//  %c_sz = select %cond, %a_sz, %b_sz : index
struct SelectResourceSizeOp : public OpRewritePattern<ResourceSizeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ResourceSizeOp op,
                                PatternRewriter &rewriter) const override {
    auto selectOp = op.getOperand().getDefiningOp<mlir::arith::SelectOp>();
    if (!selectOp) return failure();
    auto trueSize = rewriter.createOrFold<IREE::Stream::ResourceSizeOp>(
        op.getLoc(), selectOp.getTrueValue(), op.getAffinityAttr());
    auto falseSize = rewriter.createOrFold<IREE::Stream::ResourceSizeOp>(
        op.getLoc(), selectOp.getFalseValue(), op.getAffinityAttr());
    rewriter.replaceOpWithNewOp<mlir::arith::SelectOp>(
        op, selectOp.getCondition(), trueSize, falseSize);
    return success();
  }
};

}  // namespace

void ResourceSizeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                 MLIRContext *context) {
  results.insert<SelectResourceSizeOp>(context);
}

//===----------------------------------------------------------------------===//
// stream.resource.map
//===----------------------------------------------------------------------===//

void ResourceMapOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                MLIRContext *context) {
  // TODO(benvanik): fold subviews up into maps to limit range.
  results.insert<ElideUnusedOp<ResourceMapOp>>(context);
}

//===----------------------------------------------------------------------===//
// stream.resource.try_map
//===----------------------------------------------------------------------===//

void ResourceTryMapOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                   MLIRContext *context) {
  // TODO(benvanik): fold subviews up into maps to limit range.
  // TODO(benvanik): if mapping for staging then turn into a map?
  results.insert<ElideUnusedOp<ResourceTryMapOp>>(context);
}

//===----------------------------------------------------------------------===//
// stream.resource.load
//===----------------------------------------------------------------------===//

namespace {

// Folds subview offsets into loads.
//
// Example:
//  %0 = stream.resource.subview %src[%subview_offset] ... -> {%subview_length}
//  %1 = stream.resource.load %0[%offset]
// ->
//  %new_offset = arith.addi %offset, %subview_offset
//  %1 = stream.resource.load %src[%new_offset]
struct FoldSubviewIntoLoadOp : public OpRewritePattern<ResourceLoadOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ResourceLoadOp op,
                                PatternRewriter &rewriter) const override {
    auto subviewOp = ResourceSubviewOp::findSubviewOp(op.getSource());
    if (!subviewOp) return failure();
    auto fusedLoc = rewriter.getFusedLoc({subviewOp.getLoc(), op.getLoc()});
    auto newOffset = rewriter.createOrFold<arith::AddIOp>(
        fusedLoc, subviewOp.getSourceOffset(), op.getSourceOffset());
    rewriter.updateRootInPlace(op, [&]() {
      op.getSourceMutable().assign(subviewOp.getSource());
      op.getSourceSizeMutable().assign(subviewOp.getSourceSize());
      op.getSourceOffsetMutable().assign(newOffset);
    });
    return success();
  }
};

}  // namespace

void ResourceLoadOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                 MLIRContext *context) {
  // TODO(benvanik): if staging resource comes from splat (through transfers)
  //                 then pull splat value.
  // TODO(benvanik): combine multiple loads from the same target if contiguous.
  // TODO(benvanik): value->transfer->load -> value->slice->transfer->load?
  results.insert<FoldSubviewIntoLoadOp>(context);
  results.insert<ElideUnusedOp<ResourceLoadOp>>(context);
}

//===----------------------------------------------------------------------===//
// stream.resource.store
//===----------------------------------------------------------------------===//

namespace {

// Folds subview offsets into stores.
//
// Example:
//  %0 = stream.resource.subview %dst[%subview_offset] ... -> {%subview_length}
//  stream.resource.store %c123_i32, %0[%offset]
// ->
//  %new_offset = arith.addi %offset, %subview_offset
//  stream.resource.store %c123_i32, %dst[%new_offset]
struct FoldSubviewIntoStoreOp : public OpRewritePattern<ResourceStoreOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ResourceStoreOp op,
                                PatternRewriter &rewriter) const override {
    auto subviewOp = ResourceSubviewOp::findSubviewOp(op.getTarget());
    if (!subviewOp) return failure();
    auto fusedLoc = rewriter.getFusedLoc({subviewOp.getLoc(), op.getLoc()});
    auto newOffset = rewriter.createOrFold<arith::AddIOp>(
        fusedLoc, subviewOp.getSourceOffset(), op.getTargetOffset());
    rewriter.updateRootInPlace(op, [&]() {
      op.getTargetMutable().assign(subviewOp.getSource());
      op.getTargetSizeMutable().assign(subviewOp.getSourceSize());
      op.getTargetOffsetMutable().assign(newOffset);
    });
    return success();
  }
};

}  // namespace

void ResourceStoreOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                  MLIRContext *context) {
  // TODO(benvanik): combine multiple stores to the same target if contiguous.
  // TODO(benvanik): if value is a constant splat then turn into fill?
  results.insert<FoldSubviewIntoStoreOp>(context);
}

//===----------------------------------------------------------------------===//
// stream.resource.pack
//===----------------------------------------------------------------------===//

LogicalResult ResourcePackOp::fold(FoldAdaptor operands,
                                   SmallVectorImpl<OpFoldResult> &results) {
  Builder builder(getContext());

  // If there are no slices then the entire pack results in a zero-length slab.
  if (getPackedOffsets().empty()) {
    results.push_back(builder.getZeroAttr(builder.getIndexType()));
    return success();
  }

  // If there's a single slice then we just use that as there is no packing to
  // perform.
  if (getPackedOffsets().size() == 1) {
    // Total length is the slice size and offset is always either 0 or the
    // provided optional base offset.
    results.push_back(getDynamicSliceSizes()[0]);
    if (getOffset()) {
      results.push_back(getOffset());
    } else {
      results.push_back(builder.getZeroAttr(builder.getIndexType()));
    }
    return success();
  }

  return failure();
}

namespace {

// Propagates base offsets on a pack op to its results.
// This allows for better folding of the results after packing has completed.
// The offset value is just a convenience for when splitting pack ops and has
// no impact on the actual packing operation.
struct PropagateResourcePackBaseOffset
    : public OpRewritePattern<ResourcePackOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ResourcePackOp op,
                                PatternRewriter &rewriter) const override {
    // Offset is optional.
    auto baseOffset = op.getOffset();
    if (!baseOffset) return failure();

    // We always strip the offset here.
    rewriter.updateRootInPlace(op, [&]() { op.getOffsetMutable().clear(); });

    // Zero offsets don't do anything and can just be removed so we can avoid
    // inserting a bunch of additional IR.
    if (auto constantOp = baseOffset.getDefiningOp<arith::ConstantIndexOp>()) {
      if (constantOp.getValue() == 0) {
        return success();
      }
    }

    // Propagate the offset to all returned slice offsets.
    rewriter.setInsertionPointAfter(op);
    for (auto sliceOffset : op.getPackedOffsets()) {
      auto addOp =
          rewriter.create<arith::AddIOp>(op.getLoc(), baseOffset, sliceOffset);
      rewriter.replaceAllUsesExcept(sliceOffset, addOp.getResult(), addOp);
    }

    return success();
  }
};

// Sorts and compacts the slice intervals into a dense ascending order set.
// This is not required by the packing algorithm but yields more
// consistent-looking IR and makes the range overlaps easier to see for us
// meatbags.
//
// Example:
//  %0:3 = stream.resource.pack slices({
//    [1, 2] = %size,
//    [0, 4] = %size,
//  }) : index
// ->
//  %0:3 = stream.resource.pack slices({
//    [0, 4] = %size,
//    [1, 2] = %size,
//  }) : index
struct CanonicalizeResourcePackIntervals
    : public OpRewritePattern<ResourcePackOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ResourcePackOp op,
                                PatternRewriter &rewriter) const override {
    // Get the slices in a possibly unsorted order and sort.
    auto slices = op.getSlices();
    std::stable_sort(slices.begin(), slices.end());

    // See if the sorted order is different than how they are stored in the op.
    bool orderChanged = false;
    for (auto [slice, packedOffset] :
         llvm::zip_equal(slices, op.getPackedOffsets())) {
      if (slice.packedOffset != packedOffset) {
        orderChanged = true;
        break;
      }
    }
    if (!orderChanged) return failure();

    // TODO(benvanik): compact the slice ranges.

    // Rebuild the op with the sorted values.
    SmallVector<int64_t> lifetimeIntervals(slices.size() * 2);
    SmallVector<Value> dynamicSliceSizes(slices.size());
    for (size_t i = 0; i < slices.size(); ++i) {
      const auto &slice = slices[i];
      lifetimeIntervals[2 * i + 0] = slice.lifetimeStart;
      lifetimeIntervals[2 * i + 1] = slice.lifetimeEnd;
      dynamicSliceSizes[i] = slice.dynamicSize;
    }
    SmallVector<Type> packedOffsetTypes(slices.size(), rewriter.getIndexType());
    auto newOp = rewriter.create<ResourcePackOp>(
        op.getLoc(), op.getTotalLength().getType(), packedOffsetTypes,
        op.getOffset(), rewriter.getIndexArrayAttr(lifetimeIntervals),
        dynamicSliceSizes, op.getAffinityAttr());

    // Remap existing values to the new values.
    rewriter.replaceAllUsesWith(op.getTotalLength(), newOp.getTotalLength());
    for (size_t i = 0; i < newOp.getPackedOffsets().size(); ++i) {
      rewriter.replaceAllUsesWith(slices[i].packedOffset,
                                  newOp.getPackedOffsets()[i]);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

}  // namespace

void ResourcePackOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                 MLIRContext *context) {
  results.insert<PropagateResourcePackBaseOffset>(context);
  results.insert<CanonicalizeResourcePackIntervals>(context);
}

//===----------------------------------------------------------------------===//
// stream.resource.pack
//===----------------------------------------------------------------------===//

OpFoldResult ResourceSubviewOp::fold(FoldAdaptor operands) {
  if (getSourceSize() == getResultSize()) {
    // Entire range is covered; return it all.
    return getSource();
  }
  return {};
}

namespace {

// Folds subview -> subview to point at the original source resource with an
// updated range.
struct FoldResourceSubviewOps : public OpRewritePattern<ResourceSubviewOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ResourceSubviewOp op,
                                PatternRewriter &rewriter) const override {
    auto parentOp = ResourceSubviewOp::findSubviewOp(op.getSource());
    if (!parentOp) return failure();
    auto fusedLoc = rewriter.getFusedLoc({parentOp.getLoc(), op.getLoc()});
    auto newOffset = rewriter.createOrFold<arith::AddIOp>(
        fusedLoc, parentOp.getSourceOffset(), op.getSourceOffset());
    auto newOp = rewriter.create<ResourceSubviewOp>(
        fusedLoc, parentOp.getSource(), parentOp.getSourceSize(), newOffset,
        op.getResultSize());
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

// Turns selects of subviews of a resource into selects of the offset.
// This only works if the subview sizes match.
//
// Example:
//  %subview0 = stream.resource.subview %src[%offset0]
//  %subview1 = stream.resource.subview %src[%offset1]
//  %subview = select %cond, %subview0, %subview1 : !stream.resource<transient>
// ->
//  %offset = select %cond, %offset0, %offset1 : index
//  %subview = stream.resource.subview %src[%offset]
struct SinkSubviewAcrossSelectOps
    : public OpRewritePattern<mlir::arith::SelectOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(mlir::arith::SelectOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getType().isa<IREE::Stream::ResourceType>()) return failure();
    auto trueSubview = dyn_cast_or_null<IREE::Stream::ResourceSubviewOp>(
        op.getTrueValue().getDefiningOp());
    auto falseSubview = dyn_cast_or_null<IREE::Stream::ResourceSubviewOp>(
        op.getFalseValue().getDefiningOp());
    if (!trueSubview || !falseSubview) return failure();
    if (trueSubview.getSource() != falseSubview.getSource() ||
        trueSubview.getResultSize() != falseSubview.getResultSize()) {
      return failure();
    }
    auto offsetSelectOp = rewriter.create<mlir::arith::SelectOp>(
        op.getLoc(), op.getCondition(), trueSubview.getSourceOffset(),
        falseSubview.getSourceOffset());
    rewriter.replaceOpWithNewOp<IREE::Stream::ResourceSubviewOp>(
        op, op.getResult().getType(), trueSubview.getSource(),
        trueSubview.getSourceSize(), offsetSelectOp.getResult(),
        trueSubview.getResultSize());
    return success();
  }
};

}  // namespace

void ResourceSubviewOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                    MLIRContext *context) {
  results.insert<FoldResourceSubviewOps>(context);
  results.insert<SinkSubviewAcrossSelectOps>(context);
}

//===----------------------------------------------------------------------===//
// stream.tensor.import
//===----------------------------------------------------------------------===//

OpFoldResult TensorImportOp::fold(FoldAdaptor operands) {
  // If operand comes from an export with the same affinity and size then fold.
  // Different affinities may indicate exporting from one device or queue and
  // importing to a different device or queue.
  // We assume that differing encodings and shapes are compatible.
  auto exportOp = getSource().getDefiningOp<TensorExportOp>();
  if (exportOp && getAffinity() == exportOp.getAffinity() &&
      getResultSize() == exportOp.getSourceSize()) {
    return exportOp.getSource();
  }
  return {};
}

void TensorImportOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                 MLIRContext *context) {
  // TODO(benvanik): check operand and dedupe imports.
}

//===----------------------------------------------------------------------===//
// stream.tensor.export
//===----------------------------------------------------------------------===//

OpFoldResult TensorExportOp::fold(FoldAdaptor operands) {
  // If operand comes from import with the same properties then fold.
  // These checks are conservative, since encoding changes may be meaningful.
  auto importOp = getSource().getDefiningOp<TensorImportOp>();
  if (importOp && getSourceEncoding() == importOp.getResultEncoding() &&
      getSourceEncodingDims() == importOp.getResultEncodingDims() &&
      getSourceSize() == importOp.getResultSize() &&
      getAffinity() == importOp.getAffinity()) {
    return importOp.getSource();
  }
  return {};
}

void TensorExportOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                 MLIRContext *context) {
  // TODO(benvanik): check operand and dedupe exports.
}

//===----------------------------------------------------------------------===//
// stream.tensor.sizeof
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// stream.tensor.empty
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// stream.tensor.constant
//===----------------------------------------------------------------------===//

namespace {

struct TensorConstantToEmpty : public OpRewritePattern<TensorConstantOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(TensorConstantOp constantOp,
                                PatternRewriter &rewriter) const override {
    auto shapedType = constantOp.getResultEncoding().dyn_cast<ShapedType>();
    if (!shapedType) return failure();

    // See if any dim (including dynamic ones) is known zero.
    // It's still possible for empty tensors to slip through if their dynamic
    // dimensions are unknowable (external to the program, data dependencies,
    // etc) or can't be analyzed (complex global state, control flow, etc).
    auto dynamicDims = constantOp.getResultEncodingDims();
    bool anyZeroDims = false;
    for (int64_t i = 0, j = 0; i < shapedType.getRank(); ++i) {
      if (shapedType.isDynamicDim(i)) {
        auto dim = dynamicDims[j++];
        if (matchPattern(dim, m_Zero())) {
          anyZeroDims = true;
          break;
        }
      } else if (shapedType.getDimSize(i) == 0) {
        anyZeroDims = true;
        break;
      }
    }
    if (!anyZeroDims) return failure();

    // Definitely empty if here.
    auto resultSize = rewriter.createOrFold<IREE::Stream::TensorSizeOfOp>(
        constantOp.getLoc(), rewriter.getIndexType(),
        TypeAttr::get(constantOp.getResultEncoding()),
        constantOp.getResultEncodingDims(), constantOp.getAffinityAttr());
    rewriter.replaceOpWithNewOp<TensorEmptyOp>(
        constantOp, constantOp.getResult().getType(),
        constantOp.getResultEncoding(), constantOp.getResultEncodingDims(),
        resultSize, constantOp.getAffinityAttr());
    return success();
  }
};

struct TensorConstantToSplat : public OpRewritePattern<TensorConstantOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(TensorConstantOp constantOp,
                                PatternRewriter &rewriter) const override {
    auto splatAttr = constantOp.getValue().dyn_cast<SplatElementsAttr>();
    if (!splatAttr || !splatAttr.isSplat()) {
      return rewriter.notifyMatchFailure(
          constantOp,
          "only constant splat attrs can be converted to splat ops");
    }

    auto splatElementAttr = splatAttr.getSplatValue<TypedAttr>();
    auto splatValue = rewriter.create<arith::ConstantOp>(
        constantOp.getLoc(), splatElementAttr.getType(), splatElementAttr);
    auto resultType = IREE::Stream::ResourceType::get(constantOp.getContext());
    auto resultSize = rewriter.createOrFold<IREE::Stream::TensorSizeOfOp>(
        constantOp.getLoc(), rewriter.getIndexType(),
        TypeAttr::get(constantOp.getResultEncoding()),
        constantOp.getResultEncodingDims(), constantOp.getAffinityAttr());
    auto splatOp = rewriter.create<TensorSplatOp>(
        constantOp.getLoc(), resultType, splatValue,
        constantOp.getResultEncoding(), constantOp.getResultEncodingDims(),
        resultSize, constantOp.getAffinityAttr());
    rewriter.replaceOpWithNewOp<AsyncTransferOp>(
        constantOp, constantOp.getResult().getType(), splatOp.getResult(),
        resultSize, resultSize,
        /*source_affinity=*/constantOp.getAffinityAttr(),
        /*result_affinity=*/nullptr);
    return success();
  }
};

}  // namespace

void TensorConstantOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                   MLIRContext *context) {
  results.insert<TensorConstantToEmpty>(context);
  results.insert<TensorConstantToSplat>(context);
  // TODO(benvanik): if value is _mostly_ a splat, turn into splat + updates.
}

//===----------------------------------------------------------------------===//
// stream.tensor.splat
//===----------------------------------------------------------------------===//

namespace {

// Returns an integer with a bit width as small as possible to represent the
// input |pattern|, aligned to 8-bits.
//
// Examples:
//            0 : i64 ->    0 : i8
//            1 : i32 ->    1 : i8
//          123 : i32 ->  123 : i8
//         1234 : i32 -> 1234 : i16
//   0xCDCDCDCD : i32 -> 0xCD : i8
static APInt computeRequiredPatternBits(APInt pattern) {
  // Special case for well-known constant values.
  if (pattern.isZero()) return APInt(8, 0u);
  if (pattern.isAllOnes()) return APInt(8, 0xFF);

  // Extend up to a power of two bit width. This makes the value easier to work
  // with as we'll be dealing with one of 4 sizes (1/2/4/8b).
  uint64_t bitWidth = llvm::PowerOf2Ceil(pattern.getBitWidth());
  if (bitWidth != pattern.getBitWidth()) {
    // Extending as we operate - that's not good: users should have taken care
    // of this earier.
    return pattern;
  }

  uint64_t byteWidth = bitWidth / 8;
  uint64_t value = pattern.getZExtValue();
  switch (byteWidth) {
    case 1:
      // Can't go smaller than 1 byte.
      return pattern;
    case 2: {
      uint64_t b0 = value & 0xFF;
      uint64_t b1 = (value >> 8) & 0xFF;
      if (b0 == b1) {
        // 0xAAAA : i16 => 0xAA : i8
        return APInt(8, value & 0xFF);
      }
      return pattern;
    }
    case 4: {
      uint64_t b0 = value & 0xFF;
      uint64_t b1 = (value >> 8) & 0xFF;
      uint64_t b2 = (value >> 16) & 0xFF;
      uint64_t b3 = (value >> 24) & 0xFF;
      if (b0 == b1 && b0 == b2 && b0 == b3) {
        // 0xAAAAAAAA : i32 => 0xAA : i8
        return APInt(8, b0);
      } else if (b0 == b2 && b1 == b3) {
        // 0xAABBAABB : i32 => 0xAABB : i16
        return APInt(16, b0 | (b1 << 8));
      }
      return pattern;
    }
    case 8: {
      uint64_t b0 = value & 0xFF;
      uint64_t b1 = (value >> 8) & 0xFF;
      uint64_t b2 = (value >> 16) & 0xFF;
      uint64_t b3 = (value >> 24) & 0xFF;
      uint64_t b4 = (value >> 32) & 0xFF;
      uint64_t b5 = (value >> 40) & 0xFF;
      uint64_t b6 = (value >> 48) & 0xFF;
      uint64_t b7 = (value >> 56) & 0xFF;
      if (b0 == b1 && b0 == b2 && b0 == b3 && b0 == b4 && b0 == b5 &&
          b0 == b6 && b0 == b7) {
        // 0xAAAAAAAAAAAAAAAA : i64 => 0xAA : i8
        return APInt(8, b0);
      } else if ((b0 == b2 && b0 == b4 && b0 == b6) &&
                 (b1 == b3 && b1 == b5 && b1 == b7)) {
        // 0xAABBAABBAABBAABB : i64 => 0xAABB : i16
        return APInt(16, b0 | (b1 << 8));
      } else if (b0 == b4 && b1 == b5 && b2 == b6 && b3 == b7) {
        // 0xAABBCCDDAABBCCDD : i64 => 0xAABBCCDD : i32
        return APInt(32, b0 | (b1 << 8) | (b2 << 16) | (b3 << 32));
      }
      return pattern;
    }
    default:
      // Unhandled bit width.
      return pattern;
  }
}

// Narrows the bit width of a splat/fill pattern when known safe to do so.
// Target HAL implementations don't support 64-bit and a real 64-bit splat needs
// to be emulated - if we can avoid that here that's a big win. Some HAL
// implementations (such as Metal) only support 8-bit fills and anything larger
// needs to be implemented as well.
static Attribute tryNarrowPatternBits(Attribute patternAttr) {
  // Get the old pattern bitcast to an APInt. Splats are bitwise operations
  // and we don't care what the value originally was.
  APInt oldPattern;
  if (auto floatAttr = patternAttr.dyn_cast<FloatAttr>()) {
    oldPattern = floatAttr.getValue().bitcastToAPInt();
  } else if (auto intAttr = patternAttr.dyn_cast<IntegerAttr>()) {
    oldPattern = intAttr.getValue();
  } else {
    // Can't handle today.
    return patternAttr;
  }

  // Try narrowing the pattern.
  auto newPattern = computeRequiredPatternBits(oldPattern);
  if (newPattern.getBitWidth() == oldPattern.getBitWidth()) return patternAttr;

  // Wrap the result in an attribute - note that it is always an integer.
  return IntegerAttr::get(
      IntegerType::get(patternAttr.getContext(), newPattern.getBitWidth()),
      newPattern);
}

// Tries to narrow constant splat patterns to a smaller bit width.
struct NarrowSplatPattern : public OpRewritePattern<TensorSplatOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(TensorSplatOp splatOp,
                                PatternRewriter &rewriter) const override {
    // Try narrowing the pattern.
    Attribute oldPatternAttr;
    if (!matchPattern(splatOp.getValue(), m_Constant(&oldPatternAttr))) {
      return failure();
    }
    auto newPatternAttr = tryNarrowPatternBits(oldPatternAttr);
    if (newPatternAttr == oldPatternAttr) return failure();

    // Replace the pattern on the op with the new one.
    auto narrowValue =
        rewriter.create<arith::ConstantOp>(splatOp.getLoc(), newPatternAttr);
    rewriter.updateRootInPlace(
        splatOp, [&]() { splatOp.getValueMutable().assign(narrowValue); });
    return success();
  }
};

}  // namespace

void TensorSplatOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                MLIRContext *context) {
  results.insert<ElideUnusedOp<TensorSplatOp>>(context);
  results.insert<NarrowSplatPattern>(context);
}

//===----------------------------------------------------------------------===//
// stream.tensor.clone
//===----------------------------------------------------------------------===//

OpFoldResult TensorCloneOp::fold(FoldAdaptor operands) {
  auto users = getResult().getUsers();
  if (!users.empty() && std::next(users.begin()) == users.end()) {
    // If the second user is the end it means there's one user.
    return getSource();
  }
  return {};
}

namespace {

// Elides clones that don't do anything meaningful (like setting up a tie).
struct ElideUnneededTensorClones : public OpRewritePattern<TensorCloneOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(TensorCloneOp cloneOp,
                                PatternRewriter &rewriter) const override {
    if (!IREE::Util::TiedOpInterface::hasAnyTiedUses(cloneOp.getResult())) {
      rewriter.replaceOp(cloneOp, cloneOp.getSource());
      return success();
    }
    return failure();
  }
};

}  // namespace

void TensorCloneOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                MLIRContext *context) {
  // TODO(benvanik): splat -> clone duplicates splat.
  // TODO(benvanik): some way to reduce deep clone->clone->clone chains.
  // TODO(benvanik): clone + slice => slice.
  // TODO(benvanik): if both operand and result are used once then elide.
  //                 (if not tied block/fn arguments)
  results.insert<ElideUnneededTensorClones>(context);
}

//===----------------------------------------------------------------------===//
// stream.tensor.slice
//===----------------------------------------------------------------------===//

OpFoldResult TensorSliceOp::fold(FoldAdaptor operands) {
  // TODO(benvanik): fold if source_size == result_size and affinity/lifetime.
  return {};
}

void TensorSliceOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                MLIRContext *context) {
  // TODO(benvanik): turn into a transfer if target_size == update_size and
  //                 affinity/lifetime differ.
  // TODO(benvanik): splat->slice -> splat.
  // TODO(benvanik): clone->slice -> slice.
}

//===----------------------------------------------------------------------===//
// stream.tensor.fill
//===----------------------------------------------------------------------===//

namespace {

// Tries to narrow constant fill patterns to a smaller bit width.
struct NarrowFillPattern : public OpRewritePattern<TensorFillOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(TensorFillOp fillOp,
                                PatternRewriter &rewriter) const override {
    // Try narrowing the pattern.
    Attribute oldPatternAttr;
    if (!matchPattern(fillOp.getValue(), m_Constant(&oldPatternAttr))) {
      return failure();
    }
    auto newPatternAttr = tryNarrowPatternBits(oldPatternAttr);
    if (newPatternAttr == oldPatternAttr) return failure();

    // Replace the pattern on the op with the new one.
    auto narrowValue =
        rewriter.create<arith::ConstantOp>(fillOp.getLoc(), newPatternAttr);
    rewriter.updateRootInPlace(
        fillOp, [&]() { fillOp.getValueMutable().assign(narrowValue); });
    return success();
  }
};

}  // namespace

void TensorFillOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                               MLIRContext *context) {
  // TODO(benvanik): if target_size == sizeof(value) turn into splat.
  results.insert<NarrowFillPattern>(context);
}

//===----------------------------------------------------------------------===//
// stream.tensor.update
//===----------------------------------------------------------------------===//

OpFoldResult TensorUpdateOp::fold(FoldAdaptor operands) {
  // TODO(benvanik): fold if target_size == update_size and affinity/lifetime.
  return {};
}

void TensorUpdateOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                 MLIRContext *context) {
  // TODO(benvanik): turn into a transfer if target_size == update_size and
  //                 affinity/lifetime differ.
  // TODO(benvanik): turn into fill if source is a splat.
}

//===----------------------------------------------------------------------===//
// stream.tensor.load
//===----------------------------------------------------------------------===//

void TensorLoadOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                               MLIRContext *context) {
  // TODO(benvanik): splat + load -> splat value.
  // TODO(benvanik): clone + ex load -> slice (ranged) + load.
  // TODO(benvanik): slice + ex load -> slice (ranged) + load.
  // TODO(benvanik): value->transfer->load -> value->slice->transfer->load?
  // TODO(benvanik): combine multiple loads from the same target if contiguous.
}

//===----------------------------------------------------------------------===//
// stream.tensor.store
//===----------------------------------------------------------------------===//

void TensorStoreOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                MLIRContext *context) {
  // TODO(benvanik): if value is a constant splat then turn into fill.
  // TODO(benvanik): combine multiple stores to the same target if contiguous.
}

//===----------------------------------------------------------------------===//
// stream.async.alloca
//===----------------------------------------------------------------------===//

namespace {

// Sinks transient alloca-like ops down to their consumers to avoid cases where
// we allocate and then keep that live/copy-on-write it when not required.
template <typename Op>
struct SinkAllocaLikeOpToConsumers : public OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;
  LogicalResult matchAndRewrite(Op producerOp,
                                PatternRewriter &rewriter) const override {
    auto users = llvm::to_vector<4>(producerOp->getUsers());
    if (users.size() == 0) return failure();

    // If we have a single user then we can sink right to it.
    if (users.size() == 1) {
      return sinkOp(producerOp, users.front());
    }

    // If we only have users in the same block then we can safely move to the
    // first (as no change to cross-block SSA dominance can happen).
    if (!producerOp.getResult().isUsedOutsideOfBlock(producerOp->getBlock())) {
      Operation *targetOp = nullptr;
      for (auto user : users) {
        if (!targetOp || user->isBeforeInBlock(targetOp)) {
          targetOp = user;
        }
      }
      assert(targetOp);
      return sinkOp(producerOp, targetOp);
    }

    // Redundant computation here, but only in cases where we have multiple
    // users that may live outside the block the op is in.
    DominanceInfo domInfo(producerOp->getParentOp());

    // Find the common dominator block across all uses. This may be the
    // entry block itself.
    Block *commonDominator = users.front()->getBlock();
    for (auto user : users) {
      commonDominator =
          domInfo.findNearestCommonDominator(commonDominator, user->getBlock());
    }

    // Find the first use within the dominator block (if any) so that we
    // can sink down to it.
    Operation *firstUserInDominator = commonDominator->getTerminator();
    for (auto user : users) {
      if (user->getBlock() == commonDominator) {
        if (user->isBeforeInBlock(firstUserInDominator)) {
          firstUserInDominator = user;
        }
      }
    }

    // Sink to the common dominator - which may not even use the op but will
    // at least prevent us from doing extra work.
    return sinkOp(producerOp, firstUserInDominator);
  }
};

}  // namespace

void AsyncAllocaOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                MLIRContext *context) {
  // TODO(benvanik): alloca (staging) -> non-staging change to target.
  // TODO(benvanik): alloca (non-staging) -> staging change to target.
  results.insert<SinkAllocaLikeOpToConsumers<AsyncAllocaOp>>(context);
}

//===----------------------------------------------------------------------===//
// stream.async.constant
//===----------------------------------------------------------------------===//

namespace {

// Converts constants with splat values into splats.
struct ConvertSplatConstantsIntoSplats
    : public OpRewritePattern<AsyncConstantOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AsyncConstantOp constantOp,
                                PatternRewriter &rewriter) const override {
    auto value = constantOp.getValue();
    if (!value.isSplat()) return failure();

    auto splatElementAttr =
        value.dyn_cast<SplatElementsAttr>().getSplatValue<TypedAttr>();
    auto splatValue = rewriter.create<arith::ConstantOp>(
        constantOp.getLoc(), splatElementAttr.getType(), splatElementAttr);
    rewriter.replaceOpWithNewOp<IREE::Stream::AsyncSplatOp>(
        constantOp, constantOp.getResult().getType(), splatValue,
        constantOp.getResultSize(), constantOp.getAffinityAttr());
    return success();
  }
};

}  // namespace

void AsyncConstantOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                  MLIRContext *context) {
  results.insert<ConvertSplatConstantsIntoSplats>(context);
  // TODO(benvanik): if value is _mostly_ a splat, turn into splat + updates.
}

//===----------------------------------------------------------------------===//
// stream.async.splat
//===----------------------------------------------------------------------===//

void AsyncSplatOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                               MLIRContext *context) {
  // TODO(#6972): find splat+update-from and turn into fill.
  // TODO(#6972): find splat+copy-from and turn into fill.
  // TODO(#6972): find splat+update-into and turn into alloca+fill+update.
  // TODO(#6972): find splat+copy-into and turn into alloca+fill+copy.
  // TODO(#6972): clone instead of sinking to common dominator.
  results.insert<SinkAllocaLikeOpToConsumers<AsyncSplatOp>>(context);
  results.insert<ElideUnusedOp<AsyncSplatOp>>(context);
}

//===----------------------------------------------------------------------===//
// stream.async.clone
//===----------------------------------------------------------------------===//

OpFoldResult AsyncCloneOp::fold(FoldAdaptor operands) {
  // TODO(benvanik): trivial elides when there are no tied users/one user.
  return {};
}

namespace {

// Clones ops that prefer to be cloned directly.
// This prevents us from splatting out a value and then cloning that (keeping
// the memory live/etc) instead of just splatting it again on-demand.
//
// Example:
//  %0 = stream.async.splat %c123_i32
//  %1 = stream.async.clone %0
// ->
//  %1 = stream.async.splat %c123_i32
struct PropagateClonableOps : public OpRewritePattern<AsyncCloneOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AsyncCloneOp cloneOp,
                                PatternRewriter &rewriter) const override {
    if (cloneOp.use_empty()) return failure();
    auto sourceOp = cloneOp.getSource()
                        .getDefiningOp<IREE::Stream::StreamableOpInterface>();
    if (!sourceOp || !sourceOp.preferCloneToConsumers()) return failure();
    for (auto &use :
         llvm::make_early_inc_range(cloneOp.getResult().getUses())) {
      rewriter.setInsertionPoint(use.getOwner());
      auto clonedOp = rewriter.clone(*sourceOp);
      use.set(clonedOp->getResult(0));
    }
    if (cloneOp.use_empty()) {
      rewriter.eraseOp(cloneOp);
    }
    return success();
  }
};

}  // namespace

void AsyncCloneOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                               MLIRContext *context) {
  // TODO(benvanik): some way to reduce deep clone->clone->clone chains.
  results.insert<PropagateClonableOps>(context);
  results.insert<ElideUnusedOp<AsyncCloneOp>>(context);
}

//===----------------------------------------------------------------------===//
// stream.async.slice
//===----------------------------------------------------------------------===//

OpFoldResult AsyncSliceOp::fold(FoldAdaptor operands) {
  if (getSourceSize() == getResultSize()) {
    // Slicing entire source - just reroute to source.
    // Note that this breaks copy-on-write semantics but will be fixed up during
    // canonicalization if needed.
    return getSource();
  }
  return {};
}

namespace {

// Clones a splat op through a slice as a splat+slice is just a smaller splat.
//
// Example:
//  %0 = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%sz0}
//  %1 = stream.async.slice %0[%c0 to %c128] ... {%c128}
// ->
//  %1 = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%c128}
struct PropagateSplatsThroughSlices : public OpRewritePattern<AsyncSliceOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AsyncSliceOp sliceOp,
                                PatternRewriter &rewriter) const override {
    auto splatOp =
        sliceOp.getSource().getDefiningOp<IREE::Stream::AsyncSplatOp>();
    if (!splatOp) return failure();
    rewriter.replaceOpWithNewOp<IREE::Stream::AsyncSplatOp>(
        sliceOp, sliceOp.getResult().getType(), splatOp.getValue(),
        sliceOp.getResultSize(), sliceOp.getAffinityAttr());
    return success();
  }
};

}  // namespace

void AsyncSliceOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                               MLIRContext *context) {
  // TODO(benvanik): turn into a transfer if target_size == update_size and
  //                 affinity/lifetime differ.
  results.insert<PropagateSplatsThroughSlices>(context);
  results.insert<ElideUnusedOp<AsyncSliceOp>>(context);
}

//===----------------------------------------------------------------------===//
// stream.async.fill
//===----------------------------------------------------------------------===//

namespace {

// Turns fills that cover an entire target resource into splats.
// This acts as a discard as it indicates we don't care about the previous
// resource contents.
//
// Example:
//  %0 = stream.async.fill %cst, %dst[%c0 to %dstsz for %dstsz] ... {%dstsz}
// ->
//  %0 = stream.async.splat %cst : f32 -> !stream.resource<*>{%dstsz}
struct FlattenFullFillToSplat : public OpRewritePattern<AsyncFillOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AsyncFillOp fillOp,
                                PatternRewriter &rewriter) const override {
    if (fillOp.getTargetLength() == fillOp.getTargetSize()) {
      rewriter.replaceOpWithNewOp<IREE::Stream::AsyncSplatOp>(
          fillOp, fillOp.getResult().getType(), fillOp.getValue(),
          fillOp.getTargetSize(), fillOp.getAffinityAttr());
      return success();
    }
    return failure();
  }
};

}  // namespace

void AsyncFillOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.insert<FlattenFullFillToSplat>(context);
  results.insert<ElideUnusedOp<AsyncFillOp>>(context);
}

//===----------------------------------------------------------------------===//
// stream.async.update
//===----------------------------------------------------------------------===//

OpFoldResult AsyncUpdateOp::fold(FoldAdaptor operands) {
  if (getUpdateSize() == getTargetSize()) {
    // If updating the entire target then just replace with the update.
    // Note that this breaks copy-on-write semantics but will be fixed up during
    // canonicalization if needed.
    return getUpdate();
  }
  return {};
}

namespace {

// Turns a splat+update-from into a fill.
//
// Example:
//  %0 = stream.async.splat %c123_i32 ... {%c128}
//  %1 = stream.async.update %0, %dst[%c0 to %c128]
// ->
//  %1 = stream.async.fill %c123_i32, %dst[%c0 to %c128 for %c128]
struct CombineSplatUpdateFromToFill : public OpRewritePattern<AsyncUpdateOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AsyncUpdateOp updateOp,
                                PatternRewriter &rewriter) const override {
    auto splatOp =
        updateOp.getUpdate().getDefiningOp<IREE::Stream::AsyncSplatOp>();
    if (!splatOp) return failure();
    rewriter.replaceOpWithNewOp<IREE::Stream::AsyncFillOp>(
        updateOp, updateOp.getResult().getType(), updateOp.getTarget(),
        updateOp.getTargetSize(), updateOp.getTargetOffset(),
        updateOp.getTargetEnd(), updateOp.getUpdateSize(), splatOp.getValue(),
        updateOp.getAffinityAttr());
    return success();
  }
};

// Turns slice+update-from into a copy.
// This is equivalent behavior at runtime but better to schedule as a single
// operation.
//
// This could pessimize memory consumption if the slice is far from the consumer
// update: it's better to slice away a small part of a resource to retain than
// keeping the whole one around.
//
// Example:
//  %0 = stream.async.slice %src[%c0 to %c128]
//  %1 = stream.async.update %0, %dst[%c0 to %c128]
// ->
//  %1 stream.async.copy %src[%c0 to %c128], %dst[%c0 to %c128], %c128
//
// TODO(benvanik): evaluate if we want to do this in all cases - we may only
// want if it there are users of the source after this op such that we wouldn't
// be the op keeping the entire unsliced source resource live.
struct CombineSliceUpdateFromToCopy : public OpRewritePattern<AsyncUpdateOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AsyncUpdateOp updateOp,
                                PatternRewriter &rewriter) const override {
    auto sliceOp =
        updateOp.getUpdate().getDefiningOp<IREE::Stream::AsyncSliceOp>();
    if (!sliceOp || sliceOp->getBlock() != updateOp->getBlock()) {
      // Source is not a slice or a slice from out-of-block. We don't want to
      // grow memory usage by sinking the slice here (we may slice into the
      // body of a for loop, for example).
      return failure();
    }
    rewriter.replaceOpWithNewOp<IREE::Stream::AsyncCopyOp>(
        updateOp, updateOp.getResult().getType(), updateOp.getTarget(),
        updateOp.getTargetSize(), updateOp.getTargetOffset(),
        updateOp.getTargetEnd(), sliceOp.getSource(), sliceOp.getSourceSize(),
        sliceOp.getSourceOffset(), sliceOp.getSourceEnd(),
        sliceOp.getResultSize(), updateOp.getAffinityAttr());
    return success();
  }
};

}  // namespace

void AsyncUpdateOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                MLIRContext *context) {
  // TODO(benvanik): turn into a transfer if target_size == update_size and
  //                 affinity/lifetime differ.
  // TODO(#6972): updates into splats could become alloca + fill exclusive
  //              region + update into undefined contents (used in padding).
  results.insert<CombineSplatUpdateFromToFill>(context);
  results.insert<CombineSliceUpdateFromToCopy>(context);
  results.insert<ElideUnusedOp<AsyncUpdateOp>>(context);
}

//===----------------------------------------------------------------------===//
// stream.async.copy
//===----------------------------------------------------------------------===//

namespace {

// Turns a copy from an entire resource into an update. Updates can be more
// efficient during allocation as we know the producer can write directly into
// the target.
//
// Example:
//  %2 = stream.async.copy %0[%c0 to %sz0], %1[%c0 to %sz1], %sz0
// ->
//  %2 = stream.async.update %0, %1[%c0 to %sz1]
struct AsyncCopyFullSourceToUpdate : public OpRewritePattern<AsyncCopyOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AsyncCopyOp copyOp,
                                PatternRewriter &rewriter) const override {
    if (copyOp.getSourceEnd() == copyOp.getSourceSize() &&
        copyOp.getLength() == copyOp.getSourceSize()) {
      rewriter.replaceOpWithNewOp<IREE::Stream::AsyncUpdateOp>(
          copyOp, copyOp.getResult().getType(), copyOp.getTarget(),
          copyOp.getTargetSize(), copyOp.getTargetOffset(),
          copyOp.getTargetEnd(), copyOp.getSource(), copyOp.getSourceSize(),
          copyOp.getAffinityAttr());
      return success();
    }
    return failure();
  }
};

}  // namespace

void AsyncCopyOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.insert<AsyncCopyFullSourceToUpdate>(context);
  results.insert<ElideUnusedOp<AsyncCopyOp>>(context);
}

//===----------------------------------------------------------------------===//
// stream.async.collective
//===----------------------------------------------------------------------===//

void AsyncCollectiveOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                    MLIRContext *context) {
  results.insert<ElideUnusedOp<AsyncCollectiveOp>>(context);
}

//===----------------------------------------------------------------------===//
// stream.async.transfer
//===----------------------------------------------------------------------===//

OpFoldResult AsyncTransferOp::fold(FoldAdaptor operands) {
  if (auto sourceTransferOp = getSource().getDefiningOp<AsyncTransferOp>()) {
    if (sourceTransferOp.getSource().getType() == getResult().getType() &&
        sourceTransferOp.getSourceAffinity() == getResultAffinity()) {
      return sourceTransferOp.getSource();
    }
  }
  return {};
}

namespace {

// Elides transfer operations that are a no-op (from/to the same affinity and
// same resource type).
struct RedundantTransferElision : public OpRewritePattern<AsyncTransferOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AsyncTransferOp transferOp,
                                PatternRewriter &rewriter) const override {
    if (transferOp.getSourceAffinityAttr() ==
            transferOp.getResultAffinityAttr() &&
        transferOp.getSource().getType() == transferOp.getResult().getType()) {
      // Transfer performs no work, elide.
      rewriter.replaceOp(transferOp, transferOp.getSource());
      return success();
    }
    return failure();
  }
};

}  // namespace

void AsyncTransferOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                  MLIRContext *context) {
  // TODO(benvanik): staging propagation (fill of staging -> fill on device).
  results.insert<RedundantTransferElision>(context);
  results.insert<ElideUnusedOp<AsyncTransferOp>>(context);
}

//===----------------------------------------------------------------------===//
// stream.async.load
//===----------------------------------------------------------------------===//

namespace {

// Folds subsequent bitcasts into the load op. The bit width will be the same
// and it avoids additional conversion.
struct FoldAsyncLoadBitcast : public OpRewritePattern<AsyncLoadOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AsyncLoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    auto loadedValue = loadOp.getResult();
    if (!loadedValue.hasOneUse()) return failure();
    auto bitcastOp =
        dyn_cast<arith::BitcastOp>(*loadedValue.getUsers().begin());
    if (!bitcastOp) return failure();
    rewriter.updateRootInPlace(
        loadOp, [&]() { loadedValue.setType(bitcastOp.getType()); });
    rewriter.replaceOp(bitcastOp, loadedValue);
    return success();
  }
};

}  // namespace

void AsyncLoadOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  // TODO(benvanik): splat + load -> splat value.
  // TODO(benvanik): clone + ex load -> slice (ranged) + load.
  // TODO(benvanik): slice + ex load -> slice (ranged) + load.
  // TODO(benvanik): value->transfer->load -> value->slice->transfer->load?
  // TODO(benvanik): combine multiple loads from the same target if contiguous.
  results.insert<FoldAsyncLoadBitcast>(context);
}

//===----------------------------------------------------------------------===//
// stream.async.store
//===----------------------------------------------------------------------===//

namespace {

// Folds preceding bitcasts into the store op. The bit width will be the same
// and it avoids additional conversion.
struct FoldAsyncStoreBitcast : public OpRewritePattern<AsyncStoreOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AsyncStoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    auto storedValue = storeOp.getValue();
    if (auto bitcastOp =
            dyn_cast_or_null<arith::BitcastOp>(storedValue.getDefiningOp())) {
      rewriter.updateRootInPlace(storeOp, [&]() {
        storeOp.getValueMutable().assign(bitcastOp.getOperand());
      });
      return success();
    }
    return failure();
  }
};

}  // namespace

void AsyncStoreOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                               MLIRContext *context) {
  // TODO(benvanik): if value is a constant splat then turn into fill.
  // TODO(benvanik): combine multiple stores to the same target if contiguous.
  results.insert<FoldAsyncStoreBitcast>(context);
}

//===----------------------------------------------------------------------===//
// stream.async.dispatch
//===----------------------------------------------------------------------===//

void AsyncDispatchOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                  MLIRContext *context) {
  // TODO(benvanik): nothing? maybe tied type/lifetime updates?
  results.insert<ElideUnusedOp<AsyncDispatchOp>>(context);
}

//===----------------------------------------------------------------------===//
// stream.async.call
//===----------------------------------------------------------------------===//

void AsyncCallOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  // TODO(benvanik): elide calls to targets that have nosideeffects.
  // results.insert<ElideUnusedOp<AsyncCallOp>>(context);
}

//===----------------------------------------------------------------------===//
// stream.async.execute
//===----------------------------------------------------------------------===//

namespace {

// If any operands are sourced from subviews clone those subviews into the
// region and rewrite the operands to point at the original resource. This
// allows us to progressively fold the subviews into the ops consuming them.
struct CloneCapturedAsyncExecuteSubviewOps
    : public OpRewritePattern<AsyncExecuteOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AsyncExecuteOp op,
                                PatternRewriter &rewriter) const override {
    struct SubviewCapture {
      unsigned operandIdx;
      IREE::Stream::ResourceSubviewOp subviewOp;
    };
    SmallVector<SubviewCapture> captures;
    for (auto operand : llvm::enumerate(op.getResourceOperands())) {
      auto subviewOp = ResourceSubviewOp::findSubviewOp(operand.value());
      if (!subviewOp) continue;
      captures.push_back(
          SubviewCapture{static_cast<unsigned>(operand.index()), subviewOp});
    }
    if (captures.empty()) return failure();
    rewriter.startRootUpdate(op);

    auto &entryBlock = op.getBody().front();
    rewriter.setInsertionPointToStart(&entryBlock);
    for (auto &capture : captures) {
      // Replace operand with the source subview resource.
      op.getResourceOperandsMutable()
          .slice(capture.operandIdx, 1)
          .assign(capture.subviewOp.getSource());
      op.getResourceOperandSizesMutable()
          .slice(capture.operandIdx, 1)
          .assign(capture.subviewOp.getSourceSize());

      // Clone the subview into the region and wire it up to take the same
      // range as the original.
      auto arg = entryBlock.getArgument(capture.operandIdx);
      auto newOp = rewriter.create<ResourceSubviewOp>(
          capture.subviewOp.getLoc(), arg, capture.subviewOp.getSourceSize(),
          capture.subviewOp.getSourceOffset(),
          capture.subviewOp.getResultSize());
      rewriter.replaceAllUsesExcept(arg, newOp.getResult(), newOp);
    }

    rewriter.finalizeRootUpdate(op);
    return success();
  }
};

// Elides stream.async.execute ops when they have no meaningful work.
// The returned timepoint is replaced with an immediately resolved timepoint.
//
// Example:
//  %result, %timepoint = stream.async.execute with(%capture as %arg0) {
//    stream.yield %arg0
//  }
// ->
//  %result = %capture
//  %timepoint = stream.timepoint.immediate
struct ElideNoOpAsyncExecuteOp : public OpRewritePattern<AsyncExecuteOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AsyncExecuteOp op,
                                PatternRewriter &rewriter) const override {
    auto &entryBlock = op.getBody().front();
    auto yieldOp = getYieldIfOnlyOp(entryBlock);
    if (!yieldOp.has_value()) {
      // Has non-yield ops.
      return failure();
    }
    SmallVector<Value> newResults;
    for (auto operand : yieldOp->getResourceOperands()) {
      auto arg = operand.cast<BlockArgument>();
      auto capture = op.getResourceOperands()[arg.getArgNumber()];
      assert(arg.getType() == capture.getType() &&
             "expect 1:1 types on captures to results");
      newResults.push_back(capture);
    }
    auto immediateTimepoint = rewriter.create<TimepointImmediateOp>(
        op.getLoc(), op.getResultTimepoint().getType());
    newResults.push_back(immediateTimepoint);
    rewriter.replaceOp(op, newResults);
    return success();
  }
};

}  // namespace

void AsyncExecuteOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                 MLIRContext *context) {
  results.insert<ElideImmediateTimepointWait<AsyncExecuteOp>>(context);
  results.insert<ChainDependentAwaits<AsyncExecuteOp>>(context);
  results.insert<CloneCapturedAsyncExecuteSubviewOps>(context);
  results.insert<ElideNoOpAsyncExecuteOp>(context);
  results.insert<IREE::Util::ClosureOptimizationPattern<AsyncExecuteOp>>(
      context);
  results.insert<TieRegionResults<AsyncExecuteOp>>(context);
  results.insert<ElideUnusedOp<AsyncExecuteOp>>(context);
}

//===----------------------------------------------------------------------===//
// stream.async.concurrent
//===----------------------------------------------------------------------===//

void AsyncConcurrentOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                    MLIRContext *context) {
  results.insert<IREE::Util::ClosureOptimizationPattern<AsyncConcurrentOp>>(
      context);
  results.insert<TieRegionResults<AsyncConcurrentOp>>(context);
  results.insert<ElideUnusedOp<AsyncConcurrentOp>>(context);
}

//===----------------------------------------------------------------------===//
// stream.cmd.flush
//===----------------------------------------------------------------------===//

namespace {

// Folds subview ranges into flush ranges.
//
// Example:
//  %0 = stream.resource.subview %dst[%subview_offset] ... -> {%subview_length}
//  stream.cmd.flush %0[%offset for %length]
// ->
//  %new_offset = arith.addi %offset, %subview_offset
//  stream.cmd.flush %dst[%new_offset for %subview_length]
struct FoldSubviewsIntoCmdFlushOp : public OpRewritePattern<CmdFlushOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(CmdFlushOp op,
                                PatternRewriter &rewriter) const override {
    auto subviewOp = ResourceSubviewOp::findSubviewOp(op.getTarget());
    if (!subviewOp) return failure();
    setInsertionPointToParentExecutionScope(op, rewriter);
    auto fusedLoc = rewriter.getFusedLoc({subviewOp.getLoc(), op.getLoc()});
    auto newOffset = rewriter.createOrFold<arith::AddIOp>(
        fusedLoc, subviewOp.getSourceOffset(), op.getTargetOffset());
    rewriter.updateRootInPlace(op, [&]() {
      op.getTargetMutable().assign(subviewOp.getSource());
      op.getTargetSizeMutable().assign(subviewOp.getSourceSize());
      op.getTargetOffsetMutable().assign(newOffset);
    });
    return success();
  }
};

}  // namespace

void CmdFlushOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                             MLIRContext *context) {
  results.insert<FoldSubviewsIntoCmdFlushOp>(context);
}

//===----------------------------------------------------------------------===//
// stream.cmd.invalidate
//===----------------------------------------------------------------------===//

namespace {

// Folds subview ranges into invalidate ranges.
//
// Example:
//  %0 = stream.resource.subview %dst[%subview_offset] ... -> {%subview_length}
//  stream.cmd.invalidate %0[%offset for %length]
// ->
//  %new_offset = arith.addi %offset, %subview_offset
//  stream.cmd.invalidate %dst[%new_offset for %subview_length]
struct FoldSubviewsIntoCmdInvalidateOp
    : public OpRewritePattern<CmdInvalidateOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(CmdInvalidateOp op,
                                PatternRewriter &rewriter) const override {
    auto subviewOp = ResourceSubviewOp::findSubviewOp(op.getTarget());
    if (!subviewOp) return failure();
    setInsertionPointToParentExecutionScope(op, rewriter);
    auto fusedLoc = rewriter.getFusedLoc({subviewOp.getLoc(), op.getLoc()});
    auto newOffset = rewriter.createOrFold<arith::AddIOp>(
        fusedLoc, subviewOp.getSourceOffset(), op.getTargetOffset());
    rewriter.updateRootInPlace(op, [&]() {
      op.getTargetMutable().assign(subviewOp.getSource());
      op.getTargetSizeMutable().assign(subviewOp.getSourceSize());
      op.getTargetOffsetMutable().assign(newOffset);
    });
    return success();
  }
};

}  // namespace

void CmdInvalidateOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                  MLIRContext *context) {
  results.insert<FoldSubviewsIntoCmdInvalidateOp>(context);
}

//===----------------------------------------------------------------------===//
// stream.cmd.discard
//===----------------------------------------------------------------------===//

namespace {

// Folds subview ranges into discard ranges.
//
// Example:
//  %0 = stream.resource.subview %dst[%subview_offset] ... -> {%subview_length}
//  stream.cmd.discard %0[%offset for %length]
// ->
//  %new_offset = arith.addi %offset, %subview_offset
//  stream.cmd.discard %dst[%new_offset for %subview_length]
struct FoldSubviewsIntoCmdDiscardOp : public OpRewritePattern<CmdDiscardOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(CmdDiscardOp op,
                                PatternRewriter &rewriter) const override {
    auto subviewOp = ResourceSubviewOp::findSubviewOp(op.getTarget());
    if (!subviewOp) return failure();
    setInsertionPointToParentExecutionScope(op, rewriter);
    auto fusedLoc = rewriter.getFusedLoc({subviewOp.getLoc(), op.getLoc()});
    auto newOffset = rewriter.createOrFold<arith::AddIOp>(
        fusedLoc, subviewOp.getSourceOffset(), op.getTargetOffset());
    rewriter.updateRootInPlace(op, [&]() {
      op.getTargetMutable().assign(subviewOp.getSource());
      op.getTargetSizeMutable().assign(subviewOp.getSourceSize());
      op.getTargetOffsetMutable().assign(newOffset);
    });
    return success();
  }
};

}  // namespace

void CmdDiscardOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                               MLIRContext *context) {
  results.insert<FoldSubviewsIntoCmdDiscardOp>(context);
}

//===----------------------------------------------------------------------===//
// stream.cmd.fill
//===----------------------------------------------------------------------===//

namespace {

// Folds subview ranges into fill ranges.
//
// Example:
//  %0 = stream.resource.subview %dst[%subview_offset] ... -> {%subview_length}
//  stream.cmd.fill %cst, %0[%offset for %length]
// ->
//  %new_offset = arith.addi %offset, %subview_offset
//  stream.cmd.fill %cst, %dst[%new_offset for %subview_length]
struct FoldSubviewsIntoCmdFillOp : public OpRewritePattern<CmdFillOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(CmdFillOp op,
                                PatternRewriter &rewriter) const override {
    auto subviewOp = ResourceSubviewOp::findSubviewOp(op.getTarget());
    if (!subviewOp) return failure();
    setInsertionPointToParentExecutionScope(op, rewriter);
    auto fusedLoc = rewriter.getFusedLoc({subviewOp.getLoc(), op.getLoc()});
    auto newOffset = rewriter.createOrFold<arith::AddIOp>(
        fusedLoc, subviewOp.getSourceOffset(), op.getTargetOffset());
    rewriter.updateRootInPlace(op, [&]() {
      op.getTargetMutable().assign(subviewOp.getSource());
      op.getTargetSizeMutable().assign(subviewOp.getSourceSize());
      op.getTargetOffsetMutable().assign(newOffset);
    });
    return success();
  }
};

}  // namespace

void CmdFillOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.insert<FoldSubviewsIntoCmdFillOp>(context);
}

//===----------------------------------------------------------------------===//
// stream.cmd.copy
//===----------------------------------------------------------------------===//

namespace {

// Folds subview ranges into copy ranges.
//
// Example:
//  %0 = stream.resource.subview %src[%subview_offset] ... -> {%subview_length}
//  %1 = stream.resource.subview %dst[%subview_offset] ... -> {%subview_length}
//  stream.cmd.copy %0[%offset], %1[%offset], %length
// ->
//  %new_offset = arith.addi %offset, %subview_offset
//  stream.cmd.copy %src[%new_offset], %dst[%new_offset], %subview_length
struct FoldSubviewsIntoCmdCopyOp : public OpRewritePattern<CmdCopyOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(CmdCopyOp op,
                                PatternRewriter &rewriter) const override {
    auto sourceSubviewOp = ResourceSubviewOp::findSubviewOp(op.getSource());
    auto targetSubviewOp = ResourceSubviewOp::findSubviewOp(op.getTarget());
    if (!sourceSubviewOp && !targetSubviewOp) return failure();
    setInsertionPointToParentExecutionScope(op, rewriter);
    if (sourceSubviewOp) {
      auto fusedLoc =
          rewriter.getFusedLoc({sourceSubviewOp.getLoc(), op.getLoc()});
      auto newOffset = rewriter.createOrFold<arith::AddIOp>(
          fusedLoc, sourceSubviewOp.getSourceOffset(), op.getSourceOffset());
      rewriter.updateRootInPlace(op, [&]() {
        op.getSourceMutable().assign(sourceSubviewOp.getSource());
        op.getSourceSizeMutable().assign(sourceSubviewOp.getSourceSize());
        op.getSourceOffsetMutable().assign(newOffset);
      });
    }
    if (targetSubviewOp) {
      auto fusedLoc =
          rewriter.getFusedLoc({targetSubviewOp.getLoc(), op.getLoc()});
      auto newOffset = rewriter.createOrFold<arith::AddIOp>(
          fusedLoc, targetSubviewOp.getSourceOffset(), op.getTargetOffset());
      rewriter.updateRootInPlace(op, [&]() {
        op.getTargetMutable().assign(targetSubviewOp.getSource());
        op.getTargetSizeMutable().assign(targetSubviewOp.getSourceSize());
        op.getTargetOffsetMutable().assign(newOffset);
      });
    }
    return success();
  }
};

}  // namespace

void CmdCopyOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.insert<FoldSubviewsIntoCmdCopyOp>(context);
}

//===----------------------------------------------------------------------===//
// stream.cmd.collective
//===----------------------------------------------------------------------===//

namespace {

// TODO(benvanik): make this something on the DispatchOpInterface.

// Folds subview ranges into dispatch ranges.
//
// Example:
//  %0 = stream.resource.subview %src[%subview_offset] ... -> {%subview_length}
//  stream.cmd.dispatch ... {
//    rw %0[%offset] ... {%length}
//  }
// ->
//  %new_offset = arith.addi %offset, %subview_offset
//  stream.cmd.dispatch ... {
//    rw %0[%new_offset] ... {%subview_length}
//  }
template <typename Op>
struct FoldSubviewsIntoDispatchOp : public OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;
  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter &rewriter) const override {
    SmallVector<ResourceSubviewOp> resourceSubviewOps;
    resourceSubviewOps.reserve(op.getResources().size());
    bool anySubviewOps = false;
    for (auto operand : op.getResources()) {
      auto subviewOp = ResourceSubviewOp::findSubviewOp(operand);
      if (subviewOp) anySubviewOps = true;
      resourceSubviewOps.push_back(subviewOp);
    }
    if (!anySubviewOps) return failure();
    rewriter.startRootUpdate(op);

    setInsertionPointToParentExecutionScope(op, rewriter);
    for (auto [resourceIndex, subviewOp] :
         llvm::enumerate(resourceSubviewOps)) {
      if (!subviewOp) continue;
      auto fusedLoc = rewriter.getFusedLoc({subviewOp.getLoc(), op.getLoc()});
      auto newOffset = rewriter.createOrFold<arith::AddIOp>(
          fusedLoc, subviewOp.getSourceOffset(),
          op.getResourceOffsets()[resourceIndex]);
      op.getResourcesMutable()
          .slice(resourceIndex, 1)
          .assign(subviewOp.getSource());
      op.getResourceSizesMutable()
          .slice(resourceIndex, 1)
          .assign(subviewOp.getSourceSize());
      op.getResourceOffsetsMutable().slice(resourceIndex, 1).assign(newOffset);
    }

    rewriter.finalizeRootUpdate(op);
    return success();
  }
};

}  // namespace

void CmdCollectiveOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                  MLIRContext *context) {
  results.insert<FoldSubviewsIntoDispatchOp<CmdCollectiveOp>>(context);
}

//===----------------------------------------------------------------------===//
// stream.cmd.dispatch
//===----------------------------------------------------------------------===//

void CmdDispatchOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                MLIRContext *context) {
  results.insert<FoldSubviewsIntoDispatchOp<CmdDispatchOp>>(context);
}

//===----------------------------------------------------------------------===//
// stream.cmd.call
//===----------------------------------------------------------------------===//

namespace {

// TODO(benvanik): make this something on the DispatchOpInterface.
// This duplicates FoldSubviewsIntoDispatchOp to handle the call op until the
// interface can be written.
struct FoldSubviewsIntoCmdCallOp : public OpRewritePattern<CmdCallOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(CmdCallOp op,
                                PatternRewriter &rewriter) const override {
    // Original operand index + the subview.
    SmallVector<std::pair<int, ResourceSubviewOp>> resourceSubviewOps;
    bool anySubviewOps = false;
    for (auto [operandIndex, operand] :
         llvm::enumerate(op.getResourceOperands())) {
      if (operand.getType().isa<IREE::Stream::ResourceType>()) {
        auto subviewOp = ResourceSubviewOp::findSubviewOp(operand);
        if (subviewOp) anySubviewOps = true;
        resourceSubviewOps.push_back({operandIndex, subviewOp});
      }
    }
    if (!anySubviewOps) return failure();
    rewriter.startRootUpdate(op);

    setInsertionPointToParentExecutionScope(op, rewriter);
    for (auto [resourceIndex, resourceSubviewOp] :
         llvm::enumerate(resourceSubviewOps)) {
      auto [operandIndex, subviewOp] = resourceSubviewOp;
      if (!subviewOp) continue;
      auto fusedLoc = rewriter.getFusedLoc({subviewOp.getLoc(), op.getLoc()});
      auto newOffset = rewriter.createOrFold<arith::AddIOp>(
          fusedLoc, subviewOp.getSourceOffset(),
          op.getResourceOperandOffsets()[resourceIndex]);
      op.getResourceOperandsMutable()
          .slice(operandIndex, 1)
          .assign(subviewOp.getSource());
      op.getResourceOperandSizesMutable()
          .slice(resourceIndex, 1)
          .assign(subviewOp.getSourceSize());
      op.getResourceOperandOffsetsMutable()
          .slice(resourceIndex, 1)
          .assign(newOffset);
    }

    rewriter.finalizeRootUpdate(op);
    return success();
  }
};

}  // namespace

void CmdCallOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.insert<FoldSubviewsIntoCmdCallOp>(context);
}

//===----------------------------------------------------------------------===//
// stream.cmd.execute
//===----------------------------------------------------------------------===//

namespace {

// If any operands are sourced from subviews clone those subviews into the
// region and rewrite the operands to point at the original resource. This
// allows us to progressively fold the subviews into the ops consuming them.
//
// Example:
//  %0 = stream.resource.subview %src[%offset] ...
//  %1 = stream.cmd.execute with(%0 as %arg0)
// ->
//  %1 = stream.cmd.execute with(%src as %arg0) {
//    %2 = stream.resource.subview %arg0[%offset] ...
//  }
struct CloneCapturedCmdExecuteSubviewOps
    : public OpRewritePattern<CmdExecuteOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(CmdExecuteOp op,
                                PatternRewriter &rewriter) const override {
    struct SubviewCapture {
      unsigned operandIdx;
      IREE::Stream::ResourceSubviewOp subviewOp;
    };
    SmallVector<SubviewCapture> captures;
    for (auto operand : llvm::enumerate(op.getResourceOperands())) {
      auto subviewOp = ResourceSubviewOp::findSubviewOp(operand.value());
      if (!subviewOp) continue;
      captures.push_back(
          SubviewCapture{static_cast<unsigned>(operand.index()), subviewOp});
    }
    if (captures.empty()) return failure();
    rewriter.startRootUpdate(op);

    auto &entryBlock = op.getBody().front();
    rewriter.setInsertionPointToStart(&entryBlock);
    for (auto &capture : captures) {
      // Replace operand with the source subview resource.
      op.getResourceOperandsMutable()
          .slice(capture.operandIdx, 1)
          .assign(capture.subviewOp.getSource());
      op.getResourceOperandSizesMutable()
          .slice(capture.operandIdx, 1)
          .assign(capture.subviewOp.getSourceSize());

      // Clone the subview into the region and wire it up to take the same
      // range as the original.
      auto arg = entryBlock.getArgument(capture.operandIdx);
      auto newOp = rewriter.create<ResourceSubviewOp>(
          capture.subviewOp.getLoc(), arg, capture.subviewOp.getSourceSize(),
          capture.subviewOp.getSourceOffset(),
          capture.subviewOp.getResultSize());
      rewriter.replaceAllUsesExcept(arg, newOp.getResult(), newOp);
    }

    rewriter.finalizeRootUpdate(op);
    return success();
  }
};

// Elides stream.cmd.execute ops when they have no meaningful work.
// The returned timepoint is replaced with an immediately resolved timepoint.
struct ElideNoOpCmdExecuteOp : public OpRewritePattern<CmdExecuteOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(CmdExecuteOp op,
                                PatternRewriter &rewriter) const override {
    auto &entryBlock = op.getBody().front();
    auto yieldOp = getYieldIfOnlyOp(entryBlock);
    if (!yieldOp.has_value()) {
      // Has non-yield ops.
      return failure();
    }
    if (yieldOp->getNumOperands() != 0) {
      return rewriter.notifyMatchFailure(
          op, "no ops in execute region but still passing through operands");
    }
    rewriter.replaceOpWithNewOp<TimepointImmediateOp>(
        op, op.getResultTimepoint().getType());
    return success();
  }
};

}  // namespace

void CmdExecuteOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                               MLIRContext *context) {
  results.insert<ElideImmediateTimepointWait<CmdExecuteOp>>(context);
  results.insert<ChainDependentAwaits<CmdExecuteOp>>(context);
  results.insert<CloneCapturedCmdExecuteSubviewOps>(context);
  results.insert<ElideNoOpCmdExecuteOp>(context);
  results.insert<IREE::Util::ClosureOptimizationPattern<CmdExecuteOp>>(context);
  results.insert<ElideUnusedOp<CmdExecuteOp>>(context);
}

//===----------------------------------------------------------------------===//
// stream.cmd.serial
//===----------------------------------------------------------------------===//

namespace {

// Elides a region-carrying op when the region is empty.
// Requires no results that need replacement.
template <typename Op>
struct ElideEmptyCmdRegionOp : public OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;
  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter &rewriter) const override {
    auto &entryBlock = op.getBody().front();
    auto yieldOp = getYieldIfOnlyOp(entryBlock);
    if (!yieldOp.has_value()) {
      // Has non-yield ops.
      return failure();
    }
    rewriter.eraseOp(op);
    return success();
  }
};

}  // namespace

void CmdSerialOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.insert<ElideEmptyCmdRegionOp<CmdSerialOp>>(context);
}

//===----------------------------------------------------------------------===//
// stream.cmd.concurrent
//===----------------------------------------------------------------------===//

void CmdConcurrentOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                  MLIRContext *context) {
  results.insert<ElideEmptyCmdRegionOp<CmdConcurrentOp>>(context);
}

//===----------------------------------------------------------------------===//
// stream.timepoint.immediate
//===----------------------------------------------------------------------===//

OpFoldResult TimepointImmediateOp::fold(FoldAdaptor operands) {
  return IREE::Stream::TimepointAttr::get(getContext(), getResult().getType());
}

//===----------------------------------------------------------------------===//
// stream.timepoint.export
//===----------------------------------------------------------------------===//

LogicalResult TimepointExportOp::fold(FoldAdaptor operands,
                                      SmallVectorImpl<OpFoldResult> &results) {
  // If the source timepoint comes from an import op we can fold - but only if
  // the types match.
  if (auto importOp = dyn_cast_or_null<TimepointImportOp>(
          getAwaitTimepoint().getDefiningOp())) {
    if (llvm::equal(importOp.getOperandTypes(), getResultTypes())) {
      llvm::append_range(results, importOp.getOperands());
      return success();
    }
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// stream.timepoint.join
//===----------------------------------------------------------------------===//

OpFoldResult TimepointJoinOp::fold(FoldAdaptor operands) {
  if (llvm::all_of(operands.getAwaitTimepoints(),
                   [](auto operand) { return operand != nullptr; })) {
    // Immediate wait; fold into immediate.
    return IREE::Stream::TimepointAttr::get(getContext(),
                                            getResult().getType());
  } else if (getAwaitTimepoints().size() == 1) {
    // Join of a single timepoint => that timepoint.
    return getAwaitTimepoints().front();
  }
  return {};
}

namespace {

struct ElideImmediateTimepointJoinOperands
    : public OpRewritePattern<TimepointJoinOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(TimepointJoinOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> newTimepoints;
    newTimepoints.reserve(op.getAwaitTimepoints().size());
    for (auto timepoint : op.getAwaitTimepoints()) {
      if (!isa_and_nonnull<TimepointImmediateOp>(timepoint.getDefiningOp())) {
        newTimepoints.push_back(timepoint);
      }
    }
    if (newTimepoints.size() == op.getAwaitTimepoints().size())
      return failure();
    if (newTimepoints.empty()) {
      // Fully immediate; replace entire join with immediate.
      rewriter.replaceOpWithNewOp<TimepointImmediateOp>(
          op, op.getResultTimepoint().getType());
    } else {
      rewriter.updateRootInPlace(
          op, [&]() { op.getAwaitTimepointsMutable().assign(newTimepoints); });
    }
    return success();
  }
};

struct FoldDuplicateTimepointJoinOperands
    : public OpRewritePattern<TimepointJoinOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(TimepointJoinOp op,
                                PatternRewriter &rewriter) const override {
    SetVector<Value> newTimepoints;
    newTimepoints.insert(op.getAwaitTimepoints().begin(),
                         op.getAwaitTimepoints().end());
    if (newTimepoints.size() == op.getAwaitTimepoints().size())
      return failure();
    rewriter.updateRootInPlace(op, [&]() {
      op.getAwaitTimepointsMutable().assign(newTimepoints.takeVector());
    });
    return success();
  }
};

// Expands await timepoints in join ops that come from join ops.
// Local transformations will often insert joins that end up back-to-back:
//   %j0 = stream.timepoint.join max(%tp0, %tp1)
//   %j1 = stream.timepoint.join max(%tp2, %j0, %tp3)
// Which we want to fold and expand:
//   %j1 = stream.timepoint.join max(%tp2, %tp0, %tp1, %tp3)
struct ExpandTimepointJoinOperands : public OpRewritePattern<TimepointJoinOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(TimepointJoinOp op,
                                PatternRewriter &rewriter) const override {
    SetVector<Value> newTimepoints;
    bool didExpand = false;
    for (auto timepoint : op.getAwaitTimepoints()) {
      if (auto sourceJoinOp =
              dyn_cast_or_null<TimepointJoinOp>(timepoint.getDefiningOp())) {
        newTimepoints.insert(sourceJoinOp.getAwaitTimepoints().begin(),
                             sourceJoinOp.getAwaitTimepoints().end());
        didExpand = true;
      } else {
        newTimepoints.insert(timepoint);
      }
    }
    if (!didExpand) return failure();
    rewriter.updateRootInPlace(op, [&]() {
      op.getAwaitTimepointsMutable().assign(newTimepoints.takeVector());
    });
    return success();
  }
};

}  // namespace

void TimepointJoinOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                  MLIRContext *context) {
  // TODO(benvanik): elide operands if timepoint must be satisfied in use-def.
  // TODO(benvanik): sink and pull in other timepoints (join on all needed).
  results.insert<ElideImmediateTimepointJoinOperands>(context);
  results.insert<FoldDuplicateTimepointJoinOperands>(context);
  results.insert<ExpandTimepointJoinOperands>(context);
}

//===----------------------------------------------------------------------===//
// stream.timepoint.barrier
//===----------------------------------------------------------------------===//

namespace {

// Walks up the tied op SSA def chain to find a stream.timepoint.await op that
// produces the resource. Returns nullptr if no await op is found or local
// analysis cannot determine the source (spans across a branch, etc).
static std::pair<IREE::Stream::TimepointAwaitOp, Value> findSourceAwaitOp(
    Value resource) {
  Value baseResource = resource;
  while (auto definingOp = dyn_cast_or_null<IREE::Util::TiedOpInterface>(
             baseResource.getDefiningOp())) {
    if (auto awaitOp = dyn_cast<IREE::Stream::TimepointAwaitOp>(
            baseResource.getDefiningOp())) {
      return {awaitOp, baseResource};
    }
    auto tiedValue = definingOp.getTiedResultOperand(baseResource);
    if (!tiedValue) break;
    baseResource = tiedValue;
  }
  return {nullptr, nullptr};
}

// Tries to find awaits that feed into signals and then chains execution by
// propagating the original timepoint forward.
//
// Example:
//  %r0a = stream.timepoint.await %t0 => %source
//  %r0b, %t1 = stream.timepoint.barrier %r0a
// ->
//  %r0b = %source
//  %t1 = %t0
struct ChainTimepoints : public OpRewritePattern<TimepointBarrierOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(TimepointBarrierOp barrierOp,
                                PatternRewriter &rewriter) const override {
    // Try to find an await op. This may traverse through any number of tied ops
    // along the way.
    auto [awaitOp, baseResource] = findSourceAwaitOp(barrierOp.getResource());
    if (!awaitOp) return failure();

    // TODO(benvanik): move this to a pass that can do IPO. Local analysis is
    // insufficient for this. For now we conservatively ignore any case where
    // the await does not feed directly into the signal.
    if (baseResource != barrierOp.getResource()) {
      return rewriter.notifyMatchFailure(
          barrierOp, "ops exist between await and signal, not yet matching");
    }

    // Rewrite such that consumers of the signal op wait on the prior
    // timepoint.
    rewriter.replaceOp(barrierOp,
                       {
                           awaitOp.getTiedResultOperand(baseResource),
                           awaitOp.getAwaitTimepoint(),
                       });
    return success();
  }
};

}  // namespace

void TimepointBarrierOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                     MLIRContext *context) {
  results.insert<ChainTimepoints>(context);
}

//===----------------------------------------------------------------------===//
// stream.timepoint.await
//===----------------------------------------------------------------------===//

LogicalResult TimepointAwaitOp::fold(FoldAdaptor operands,
                                     SmallVectorImpl<OpFoldResult> &results) {
  if (operands.getAwaitTimepoint()) {
    // Immediate wait; fold to all captured operands.
    results.append(getResourceOperands().begin(), getResourceOperands().end());
    return success();
  }
  return failure();
}

namespace {

struct ElideImmediateHostAwaits : public OpRewritePattern<TimepointAwaitOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(TimepointAwaitOp op,
                                PatternRewriter &rewriter) const override {
    if (isa_and_nonnull<TimepointImmediateOp>(
            op.getAwaitTimepoint().getDefiningOp())) {
      rewriter.replaceOp(op, op.getResourceOperands());
      return success();
    }
    return failure();
  }
};

// Sinks an await down to the first consumer of any resource. Note that there
// may be multiple resources guarded by the await.
struct SinkAwaitToFirstConsumer : public OpRewritePattern<TimepointAwaitOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(TimepointAwaitOp op,
                                PatternRewriter &rewriter) const override {
    // TODO(benvanik): amortize this dominance calculation.
    DominanceInfo domInfo(op->getParentOp());

    // Gather all direct users of the awaited resources and find the common
    // dominator block across all uses. This may be the entry block itself.
    SetVector<Operation *> allUsers;
    Block *commonDominator = nullptr;
    for (auto result : op.getResults()) {
      for (auto &use : result.getUses()) {
        if (allUsers.insert(use.getOwner())) {
          auto *userBlock = use.getOwner()->getBlock();
          commonDominator = commonDominator
                                ? domInfo.findNearestCommonDominator(
                                      commonDominator, userBlock)
                                : userBlock;
        }
      }
    }
    if (!commonDominator) return failure();

    // Find the first use within the dominator block (if any) so that we
    // can sink down to it.
    Operation *firstUserInDominator = commonDominator->getTerminator();
    for (auto *user : allUsers) {
      if (user->getBlock() == commonDominator) {
        if (user->isBeforeInBlock(firstUserInDominator)) {
          firstUserInDominator = user;
        }
      }
    }

    // If sinking to `firstUserInDominator` could result in patterns
    // fighting each other, then don't sink.
    if (!canStablySinkTo(op, firstUserInDominator)) return failure();

    rewriter.updateRootInPlace(op,
                               [&]() { op->moveBefore(firstUserInDominator); });
    return success();
  }
};

// Moves stream.resource.subview ops across to results of an await.
// This allows us to pass-through the subviews to consumers that can hopefully
// fold the range.
struct SinkSubviewsAcrossAwaits : public OpRewritePattern<TimepointAwaitOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(TimepointAwaitOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.startRootUpdate(op);
    bool didChange = false;
    for (auto operand : llvm::enumerate(op.getResourceOperands())) {
      auto subviewOp =
          operand.value().getDefiningOp<IREE::Stream::ResourceSubviewOp>();
      if (!subviewOp) continue;
      didChange = true;
      unsigned operandIdx = static_cast<unsigned>(operand.index());

      // Create a new subview op matching the original on our result and swap
      // users to it.
      auto result = op.getResults()[operandIdx];
      auto newOp = rewriter.create<IREE::Stream::ResourceSubviewOp>(
          subviewOp.getLoc(), result, subviewOp.getSourceSize(),
          subviewOp.getSourceOffset(), subviewOp.getResultSize());
      rewriter.replaceAllUsesExcept(result, newOp.getResult(), newOp);

      // Update our bound size to the subview source size (not the subrange).
      op.getResourceOperandSizesMutable()
          .slice(operandIdx, 1)
          .assign(subviewOp.getSourceSize());

      // Replace our resource usage with the source of the subview op.
      op.getResourceOperandsMutable()
          .slice(operandIdx, 1)
          .assign(subviewOp.getSource());
    }
    if (didChange) {
      rewriter.finalizeRootUpdate(op);
      return success();
    } else {
      rewriter.cancelRootUpdate(op);
      return failure();
    }
  }
};

// Returns true if all operands of |op| are defined before |insertionPoint| in
// the containing block.
static bool areAllOperandsDefinedBy(Operation *op, Operation *insertionPoint,
                                    DominanceInfo &dominanceInfo) {
  for (auto operand : op->getOperands()) {
    if (!dominanceInfo.dominates(operand, insertionPoint)) return false;
  }
  return true;
}

// Finds timepoint awaits on the same timepoint within the same domination
// paths and groups them together.
//
// Example:
//  %6 = stream.timepoint.await %tp => %3 : !stream.resource<external>{%c4000}
//  %7 = stream.tensor.export %6 ...
//  %8 = stream.timepoint.await %tp => %4 : !stream.resource<external>{%c4000}
//  %9 = stream.tensor.export %8 ...
// ->
//  %6:2 = stream.timepoint.await %tp => %3, %4 :
//      !stream.resource<external>{%c4000}, !stream.resource<external>{%c4000}
//  %7 = stream.tensor.export %6#0 ...
//  %9 = stream.tensor.export %6#1 ...
struct GroupAwaitsByTimepoint : public OpRewritePattern<TimepointAwaitOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(TimepointAwaitOp op,
                                PatternRewriter &rewriter) const override {
    DominanceInfo dominanceInfo(op->getParentOp());
    SmallVector<TimepointAwaitOp> coveredOps;
    for (auto &use : op.getAwaitTimepoint().getUses()) {
      // TODO(benvanik): make this handle joins/ties; today we get blocked
      // there. We rely on other canonicalizers to sink things such that
      // (hopefully) we get them directly accessible here.
      if (use.getOwner() == op) continue;
      if (op->getBlock() != use.getOwner()->getBlock()) continue;
      if (dominanceInfo.dominates(use.getOwner(), op)) continue;
      auto awaitOp = dyn_cast<TimepointAwaitOp>(use.getOwner());
      if (!awaitOp ||
          !AffinityAttr::areCompatible(
              op.getAffinityAttr().dyn_cast_or_null<AffinityAttr>(),
              awaitOp.getAffinityAttr().dyn_cast_or_null<AffinityAttr>())) {
        // Can't combine if the affinities differ as the wait semantics are
        // load-bearing. Probably. They really shouldn't be.
        // TODO(benvanik): remove affinity from stream.timepoint.await.
        continue;
      }
      // Ensure all dependencies of the await op are available.
      if (!areAllOperandsDefinedBy(awaitOp, op, dominanceInfo)) {
        // One or more operands is defined after op so we can't merge.
        continue;
      }
      coveredOps.push_back(awaitOp);
    }
    if (coveredOps.empty()) return failure();
    coveredOps.push_back(op);

    // Sort the ops by their definition order; this gives us a deterministic
    // operand ordering regardless of the order the patterns are run.
    llvm::sort(coveredOps, [&](TimepointAwaitOp lhs, TimepointAwaitOp rhs) {
      return lhs->isBeforeInBlock(rhs);
    });

    // Combine all awaits into a single one.
    SmallVector<Value> newOperands;
    SmallVector<Value> newOperandSizes;
    for (auto coveredOp : coveredOps) {
      llvm::append_range(newOperands, coveredOp.getResourceOperands());
      llvm::append_range(newOperandSizes, coveredOp.getResourceOperandSizes());
    }
    auto newOp = rewriter.create<TimepointAwaitOp>(
        op.getLoc(), newOperands, newOperandSizes, op.getAwaitTimepoint());
    if (op.getAffinity().has_value()) {
      newOp.setAffinityAttr(op.getAffinityAttr());
    }

    // Replace covered ops with the new results.
    unsigned resultIdx = 0;
    for (auto coveredOp : coveredOps) {
      for (auto result : coveredOp.getResults()) {
        rewriter.replaceAllUsesWith(result, newOp.getResults()[resultIdx++]);
      }
      rewriter.eraseOp(coveredOp);
    }
    return success();
  }
};

// Folds duplicate resources passing through an await op.
//
// Example:
//  %1:4 = stream.timepoint.await %tp => %1, %1, %2, %2
// ->
//  %1:2 = stream.timepoint.await %tp => %1, %2
struct FoldDuplicateAwaitResources : public OpRewritePattern<TimepointAwaitOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(TimepointAwaitOp op,
                                PatternRewriter &rewriter) const override {
    DenseMap<Value, unsigned> baseMap;
    SmallVector<std::pair<Value, unsigned>> replacements;
    SmallVector<Value> newOperands;
    SmallVector<Value> newOperandSizes;
    for (auto [operand, operandSize, result] :
         llvm::zip_equal(op.getResourceOperands(), op.getResourceOperandSizes(),
                         op.getResults())) {
      auto insertion =
          baseMap.insert(std::make_pair(operand, newOperands.size()));
      if (insertion.second) {
        // Inserted as a new unique operand.
        newOperands.push_back(operand);
        newOperandSizes.push_back(operandSize);
      }
      unsigned resultIdx = insertion.first->second;
      replacements.push_back(std::make_pair(result, resultIdx));
    }
    if (newOperands.size() == op.getResourceOperands().size()) {
      return failure();  // No change.
    }

    // Create replacement op with deduped operands/results.
    auto newOp = rewriter.create<IREE::Stream::TimepointAwaitOp>(
        op.getLoc(), newOperands, newOperandSizes, op.getAwaitTimepoint());
    if (op.getAffinity().has_value()) {
      newOp.setAffinityAttr(op.getAffinityAttr());
    }

    // Replace all duplicate results with the base results.
    for (auto &replacement : replacements) {
      auto oldResult = replacement.first;
      auto newResult = newOp.getResults()[replacement.second];
      rewriter.replaceAllUsesWith(oldResult, newResult);
    }
    rewriter.eraseOp(op);
    return success();
  }
};

}  // namespace

void TimepointAwaitOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                   MLIRContext *context) {
  // TODO(benvanik): elide waits if timepoint must be satisfied in use-def.
  results.insert<ElideImmediateHostAwaits>(context);
  results.insert<SinkAwaitToFirstConsumer>(context);
  results.insert<SinkSubviewsAcrossAwaits>(context);
  results.insert<GroupAwaitsByTimepoint>(context);
  results.insert<FoldDuplicateAwaitResources>(context);
  results.insert<ElideUnusedOp<TimepointAwaitOp>>(context);
}

}  // namespace Stream
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

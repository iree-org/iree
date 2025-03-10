// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::Stream {

#define GEN_PASS_DEF_AUTOMATICREFERENCECOUNTINGPASS
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h.inc"

namespace {

// Rules:
//   stream.tensor.import:
//     - retains by default
//     - callers can opt-in to consume behavior to avoid retain and transfer
//   stream.tensor.export:
//     - consumes by default
//   arith.select:
//     - both operands retained and the unselected one is released
//   CFG blocks:
//     - branches retain for target blocks (as with calls)
//     - cond branches need to release any non-selected values
//   Call ops:
//     - callers retain for callees
//     - callers can transfer ownership (callee "consumes") by not retaining
//     - callees do not retain for callers (return "consumes")
//     - tied operands on calls are considered as "consuming"

//===----------------------------------------------------------------------===//
// Utilities for managing ref counting ops
//===----------------------------------------------------------------------===//

// Returns true if the argument or result at |idx| has the `stream.consume`
// attribute in its arg/result attr dictionary.
static bool hasConsumeAttr(ArrayAttr attrsAttr, unsigned idx) {
  if (auto attrs =
          dyn_cast_if_present<DictionaryAttr>(attrsAttr.getValue()[idx])) {
    return attrs.contains("stream.consume");
  }
  return false;
}

// Attempts to find a timepoint indicating when the given |resourceValue| is
// available for use. If no timepoint is found using local analysis a barrier
// will be inserted with the given affinity.
// Returns nullptr if the resource is known to be immediately available.
static Value
queryBarrierTimepoint(Location loc, Value resourceValue, Value resourceSize,
                      IREE::Stream::AffinityAttr consumerAffinityAttr,
                      OpBuilder &builder) {
  // Simple check to try to find the latest timepoint.
  // This does not look across CFG edges and thus does not have the chance to
  // be ambiguous - the defining op either has a timepoint or we have to insert
  // a barrier.
  if (auto *definingOp = resourceValue.getDefiningOp()) {
    if (auto awaitOp = dyn_cast<IREE::Stream::TimepointAwaitOp>(definingOp)) {
      // Sync points are never removed and let us know the resource is
      // immediately available.
      if (awaitOp.getSync()) {
        return {}; // immediate
      } else {
        return awaitOp.getAwaitTimepoint();
      }
    } else if (auto timelineOp =
                   dyn_cast<IREE::Stream::TimelineOpInterface>(definingOp)) {
      return timelineOp.getResultTimepoint();
    }
  }

  // No timepoint clearly visible so insert a barrier. This relies on timepoint
  // propagation and elision to clean up the barrier in cases where there's a
  // timepoint elsewhere in the program (global stores -> global loads, etc).
  auto barrierOp = builder.create<IREE::Stream::TimepointBarrierOp>(
      loc, resourceValue, resourceSize, consumerAffinityAttr);
  return barrierOp.getResultTimepoint();
}

// Inserts a new `stream.async.retain` op for the given |resourceValue|.
static Value createRetain(Operation *forOp, Value resourceValue,
                          OpBuilder &builder) {
  Value resourceSize = IREE::Util::SizeAwareTypeInterface::queryValueSize(
      forOp->getLoc(), resourceValue, builder);
  auto retainOp = builder.create<IREE::Stream::AsyncRetainOp>(
      forOp->getLoc(), resourceValue, resourceSize);
  return retainOp.getResult();
}

// Inserts a new `stream.async.release` op for the given |resourceValue|.
// The resource will be waited on (to get the wait timepoint), the release will
// be issued, and then the result of that will be waited on (to get the signal
// timepoint). It's expected that timepoint propagation will handle cleaning up
// the timeline.
static Value createSyncRelease(Operation *forOp, Value resourceValue,
                               OpBuilder &builder) {
  Value resourceSize = IREE::Util::SizeAwareTypeInterface::queryValueSize(
      forOp->getLoc(), resourceValue, builder);
  auto affinityAttr = IREE::Stream::AffinityAttr::lookup(forOp);
  Value barrierTimepoint = queryBarrierTimepoint(
      forOp->getLoc(), resourceValue, resourceSize, affinityAttr, builder);
  auto releaseOp = builder.create<IREE::Stream::AsyncReleaseOp>(
      forOp->getLoc(), resourceValue, resourceSize, barrierTimepoint,
      affinityAttr);
  auto awaitOp = builder.create<IREE::Stream::TimepointAwaitOp>(
      forOp->getLoc(), releaseOp.getResult(), resourceSize,
      releaseOp.getResultTimepoint());
  return awaitOp.getResult(0);
}

struct TimepointResource {
  Value timepoint;
  Value resource;
};

// Inserts a new `stream.async.release` op for the given |resourceValue| that
// waits on |barrierTimepoint|. Returns a pair of
// `{releaseTimepoint, releaseValue}` that must be used by callers to ensure
// they order operations after the release occurs.
static TimepointResource createAsyncRelease(Operation *forOp,
                                            Value barrierTimepoint,
                                            Value resourceValue,
                                            Value resourceSize,
                                            OpBuilder &builder) {
  auto affinityAttr = IREE::Stream::AffinityAttr::lookup(forOp);
  auto releaseOp = builder.create<IREE::Stream::AsyncReleaseOp>(
      forOp->getLoc(), resourceValue, resourceSize, barrierTimepoint,
      affinityAttr);
  return TimepointResource{
      releaseOp.getResultTimepoint(),
      releaseOp.getResult(),
  };
}

//
static Value createResourceFence(Operation *forOp, ValueRange resourceValues,
                                 OpBuilder &builder) {
  if (resourceValues.empty()) {
    //
  } else if (resourceValues.size() == 1) {
    //
  } else {
    //
  }
}

//===----------------------------------------------------------------------===//
// Last-use deallocation
//===----------------------------------------------------------------------===//

// last use

// export as last use: consume vs retain

// call as last user:
// set stream.consume on arg? if multiple args use same value then only on one?

// return as last user:
// forward?/consume?

// cond/branch as last user

LogicalResult addDropRefInDivergentLivenessSuccessor(Liveness &liveness,
                                                     Value value) {
  using BlockSet = llvm::SmallPtrSet<Block *, 4>;

  OpBuilder builder(value.getContext());

  // If a block has successors with different `liveIn` property of the `value`,
  // record block successors that do not thave the `value` in the `liveIn` set.
  llvm::SmallDenseMap<Block *, BlockSet> divergentLivenessBlocks;

  // Because we only add `drop_ref` operations to the region that defines the
  // `value` we can only process CFG for the same region.
  Region *definingRegion = value.getParentRegion();

  // Collect blocks with successors with mismatching `liveIn` sets.
  for (Block &block : definingRegion->getBlocks()) {
    const LivenessBlockInfo *blockLiveness = liveness.getLiveness(&block);

    // Skip the block if value is not in the `liveOut` set.
    if (!blockLiveness || !blockLiveness->isLiveOut(value))
      continue;

    BlockSet liveInSuccessors;   // `value` is in `liveIn` set
    BlockSet noLiveInSuccessors; // `value` is not in the `liveIn` set

    // Collect successors that do not have `value` in the `liveIn` set.
    for (Block *successor : block.getSuccessors()) {
      const LivenessBlockInfo *succLiveness = liveness.getLiveness(successor);
      if (succLiveness && succLiveness->isLiveIn(value))
        liveInSuccessors.insert(successor);
      else
        noLiveInSuccessors.insert(successor);
    }

    // Block has successors with different `liveIn` property of the `value`.
    if (!liveInSuccessors.empty() && !noLiveInSuccessors.empty())
      divergentLivenessBlocks.try_emplace(&block, noLiveInSuccessors);
  }

  // Try to insert `dropRef` operations to handle blocks with divergent liveness
  // in successors blocks.
  for (auto kv : divergentLivenessBlocks) {
    Block *block = kv.getFirst();
    BlockSet &successors = kv.getSecond();

    // DO NOT SUBMIT
    // yield op trait?
    // Coroutine suspension is a special case terminator for wich we do not
    // need to create additional reference counting (see details above).
    Operation *terminator = block->getTerminator();
    // if (isa<CoroSuspendOp>(terminator))
    // continue;

    // We only support successor blocks with empty block argument list.
    auto hasArgs = [](Block *block) { return !block->getArguments().empty(); };
    if (llvm::any_of(successors, hasArgs))
      return terminator->emitOpError()
             << "successor have different `liveIn` property of the reference "
                "counted value";

    // Make sure that `dropRef` operation is called when branched into the
    // successor block without `value` in the `liveIn` set.
    for (Block *successor : successors) {
      // If successor has a unique predecessor, it is safe to create `dropRef`
      // operations directly in the successor block.
      //
      // Otherwise we need to create a special block for reference counting
      // operations, and branch from it to the original successor block.
      Block *refCountingBlock = nullptr;

      if (successor->getUniquePredecessor() == block) {
        refCountingBlock = successor;
      } else {
        refCountingBlock = &successor->getParent()->emplaceBlock();
        refCountingBlock->moveBefore(successor);
        OpBuilder builder = OpBuilder::atBlockEnd(refCountingBlock);
        builder.create<cf::BranchOp>(value.getLoc(), successor);
      }

      OpBuilder builder = OpBuilder::atBlockBegin(refCountingBlock);
      createSyncRelease(terminator, value, builder);

      // No need to update the terminator operation.
      if (successor == refCountingBlock)
        continue;

      // Update terminator `successor` block to `refCountingBlock`.
      for (const auto &pair : llvm::enumerate(terminator->getSuccessors()))
        if (pair.value() == successor)
          terminator->setSuccessor(refCountingBlock, pair.index());
    }
  }

  return success();
}
static LogicalResult insertLastUseReleases(CallableOpInterface funcOp) {
  Liveness liveness(funcOp);

  SmallVector<std::function<void()>> pendingWork;

  auto insertForValue = [&](Value value) -> LogicalResult {
    // DO NOT SUBMIT drop if no uses
    if (value.getUses().empty()) {
      // Set insertion point after the operation producing a value, or at the
      // beginning of the block if the value defined by the block argument.
      auto *parentBlock = value.getParentBlock();
      pendingWork.push_back([=]() {
        OpBuilder b(value.getContext());
        if (Operation *op = value.getDefiningOp()) {
          b.setInsertionPointAfter(op);
          createSyncRelease(op, value, b);
        } else {
          b.setInsertionPointToStart(parentBlock);
          createSyncRelease(funcOp, value, b);
        }
      });
      return success();
    }

    // DO NOT SUBMIT addDropRefAfterLastUse
    Region *definingRegion = value.getParentRegion();
    SmallPtrSet<Operation *, 4> lastUsers;
    llvm::DenseMap<Block *, Operation *> usersInTheBlocks;
    for (Operation *user : value.getUsers()) {
      Block *userBlock = user->getBlock();
      Block *ancestor = definingRegion->findAncestorBlockInRegion(*userBlock);
      usersInTheBlocks[ancestor] = ancestor->findAncestorOpInBlock(*user);
      assert(ancestor && "ancestor block must be not null");
      assert(usersInTheBlocks[ancestor] && "ancestor op must be not null");
    }
    for (auto &blockAndUser : usersInTheBlocks) {
      Block *block = blockAndUser.getFirst();
      Operation *userInTheBlock = blockAndUser.getSecond();

      const LivenessBlockInfo *blockLiveness = liveness.getLiveness(block);

      // Value must be in the live input set or defined in the block.
      assert(blockLiveness->isLiveIn(value) ||
             blockLiveness->getBlock() == value.getParentBlock());

      // If value is in the live out set, it means it doesn't "die" in the
      // block.
      if (blockLiveness->isLiveOut(value))
        continue;

      // At this point we proved that `value` dies in the `block`. Find the last
      // use of the `value` inside the `block`, this is where it "dies".
      Operation *lastUser =
          blockLiveness->getEndOperation(value, userInTheBlock);
      assert(lastUsers.count(lastUser) == 0 && "last users must be unique");
      lastUsers.insert(lastUser);
    }
    // Process all the last users of the `value` inside each block where the
    // value dies.
    for (Operation *lastUser : lastUsers) {
      // Return like operations forward reference count.
      if (lastUser->hasTrait<OpTrait::ReturnLike>())
        continue;

      // We can't currently handle other types of terminators.
      if (lastUser->hasTrait<OpTrait::IsTerminator>()) {
        return lastUser->emitError() << "async reference counting can't handle "
                                        "terminators that are not ReturnLike";
      }

      // Add a drop_ref immediately after the last user.
      pendingWork.push_back([=]() {
        OpBuilder builder(value.getContext());
        builder.setInsertionPointAfter(lastUser);
        createSyncRelease(lastUser, value, builder);
      });
    }
    // DO NOT SUBMIT addDropRefInDivergentLivenessSuccessor
    return addDropRefInDivergentLivenessSuccessor(liveness, value);
  };

  if (funcOp
          .walk([&](Block *block) -> WalkResult {
            if (block->getParent() != funcOp.getCallableRegion() &&
                !isa<scf::IfOp>(block->getParentOp())) {
              return WalkResult::advance();
            }
            for (auto arg : block->getArguments()) {
              if (isa<IREE::Stream::ResourceType>(arg.getType())) {
                if (failed(insertForValue(arg))) {
                  return WalkResult::interrupt();
                }
              }
            }
            return WalkResult::advance();
          })
          .wasInterrupted()) {
    return failure();
  }

  std::function<LogicalResult(Operation * rootOp)> walkOps;
  walkOps = [&](Operation *rootOp) -> LogicalResult {
    for (auto &region : rootOp->getRegions()) {
      for (auto &block : llvm::reverse(region)) {
        for (auto &op : block) {
          for (auto result : op.getResults()) {
            if (isa<IREE::Stream::ResourceType>(result.getType())) {
              if (failed(insertForValue(result))) {
                return failure();
              }
            }
          }
          // DO NOT SUBMIT have to recurse
          if (isa<scf::IfOp>(op) || isa<scf::ForOp>(op) ||
              isa<scf::WhileOp>(op)) {
            if (failed(walkOps(&op))) {
              return failure();
            }
          }
        }
      }
    }
    return success();
  };

  if (failed(walkOps(funcOp.getOperation()))) {
    return failure();
  }

  for (auto &work : pendingWork) {
    work();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Reference counting
//===----------------------------------------------------------------------===//

// Retains all resource operands on the op to balance out last-use releases.
// Tied operands are not retained as they do not receive releases on their
// last use by an op that ties them.
template <typename OpT>
struct RetainOperandsPattern final : public OpRewritePattern<OpT> {
  using OpRewritePattern<OpT>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpT op,
                                PatternRewriter &rewriter) const override {
    // Gather a list of operands that need to be retained, if any.
    SmallVector<OpOperand *> retainOperands;
    auto tiedOp = dyn_cast<IREE::Util::TiedOpInterface>(op.getOperation());
    for (auto &operand : op->getOpOperands()) {
      if (isa<IREE::Stream::ResourceType>(operand.get().getType()) &&
          (!tiedOp || !tiedOp.isOperandTied(operand.getOperandNumber()))) {
        retainOperands.push_back(&operand);
      }
    }
    if (retainOperands.empty()) {
      return failure(); // nothing to update
    }

    // Retain each operand and pass the retained value to the op.
    rewriter.modifyOpInPlace(op, [&]() {
      for (auto *operand : retainOperands) {
        operand->set(createRetain(op, operand->get(), rewriter));
      }
    });

    return success();
  }
};

// Retains all resource results on the op that acts as an origin for an SSA
// use-def chain.
template <typename OpT>
struct RetainResultsPattern final : public OpRewritePattern<OpT> {
  using OpRewritePattern<OpT>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpT op,
                                PatternRewriter &rewriter) const override {
    // Gather a list of results that need to be retained, if any.
    SmallVector<OpResult *> retainResults;
    auto tiedOp = dyn_cast<IREE::Util::TiedOpInterface>(op.getOperation());
    for (auto &result : op->getOpResults()) {
      if (isa<IREE::Stream::ResourceType>(result.getType()) &&
          (!tiedOp ||
           !tiedOp.getTiedResultOperandIndex(result.getResultNumber())
                .has_value())) {
        retainResults.push_back(&result);
      }
    }
    if (retainResults.empty()) {
      return failure(); // nothing to update
    }

    // Retain each result and pass the retained value to consumers.
    // Note that because we change the uses of the result as we retain it we
    // need to cache the list of original uses prior to any modification.
    rewriter.setInsertionPointAfter(op);
    for (auto *result : retainResults) {
      auto uses = llvm::map_to_vector(
          result->getUses(), [](OpOperand &operand) { return &operand; });
      Value retainedResult = createRetain(op, *result, rewriter);
      for (auto *use : uses) {
        rewriter.modifyOpInPlace(use->getOwner(),
                                 [&]() { use->set(retainedResult); });
      }
    }

    return success();
  }
};

// Execution happening on the timeline needs to retain all resources consumed
// for the duration of the execution and then release them. This is in contrast
// to most other ops that only deal with retaining and leave releases to the
// last use cleanup as timeline operations happen out-of-band with host code.
//
// A sequence of:
//   %operand = ... produce ...
//   execute(%operand)
//   release-at-last-use %operand
//
// Is turned into:
//   %operand = ... produce ...
//   retain-for-execution %operand
//   execute(%operand)
//   release-for-execution %operand
//   release-at-last-use %operand
//
// We rely on whole-program analysis to remove the redundant retain/release in
// cases where analysis indicates the timeline allows for it. In cases where
// timelines diverge (a fork) the last-use release may be ambiguous from the
// perspective of the devices executing the work.
//
// Tied operands are treated special in that we are creating a new SSA value for
// them even though they are the same resource. Because of this we have to
// retain them explicitly for future consumers but do not need to release them.
// Technically a tied operand sequence becomes:
// - retain tied operand
// - execute
// - retain tied result
// - release tied operand
// but to avoid extra IR we elide the retain/release and only preserve the
// retain on the result. Last-use insertion should release the operand if it
// was the last use and will take care of releasing the result as normal.
struct AsyncExecuteOpPattern final
    : public OpRewritePattern<IREE::Stream::AsyncExecuteOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(IREE::Stream::AsyncExecuteOp executeOp,
                                PatternRewriter &rewriter) const override {
    // Gather a list of operands that need to be retained/released, if any.
    SmallVector<OpOperand *> retainableOperands;
    for (auto &operand : executeOp->getOpOperands()) {
      if (isa<IREE::Stream::ResourceType>(operand.get().getType()) &&
          !executeOp.isOperandTied(operand.getOperandNumber())) {
        retainableOperands.push_back(&operand);
      }
    }

    // Gather a list of tied results that need to be retained, if any.
    SmallVector<OpResult *> retainResults;
    for (auto &result : executeOp.getResults()) {
      assert(isa<IREE::Stream::ResourceType>(result.getType()));
      if (executeOp.getTiedResultOperandIndex(result.getResultNumber())
              .has_value()) {
        retainResults.push_back(&result);
      }
    }

    if (retainableOperands.empty() && retainResults.empty()) {
      return failure(); // nothing to update
    }

    // Retain each operand and pass the retained value to the op.
    rewriter.modifyOpInPlace(executeOp, [&]() {
      for (auto *operand : retainableOperands) {
        operand->set(createRetain(executeOp, operand->get(), rewriter));
      }
    });

    // Retain each tied result and pass the retained value to consumers.
    // Note that because we change the uses of the result as we retain it we
    // need to cache the list of original uses prior to any modification.
    rewriter.setInsertionPointAfter(executeOp);
    for (auto *result : retainResults) {
      auto uses = llvm::map_to_vector(
          result->getUses(), [](OpOperand &operand) { return &operand; });
      Value retainedResult = createRetain(executeOp, *result, rewriter);
      for (auto *use : uses) {
        rewriter.modifyOpInPlace(use->getOwner(),
                                 [&]() { use->set(retainedResult); });
      }
    }

    return success();
  }
};

// Imports by default assume the caller has not retained the value (as
// opposed to how we treat internal calls) and that we must. This allows
// naive calling code to function but prevents ownership transfer by the
// caller without additional annotations.
struct TensorImportOpPattern final
    : public OpRewritePattern<IREE::Stream::TensorImportOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(IREE::Stream::TensorImportOp importOp,
                                PatternRewriter &rewriter) const override {
    if (importOp.getConsume()) {
      return rewriter.notifyMatchFailure(
          importOp, "import consumes the operand and does not need a retain");
    }
    rewriter.setInsertionPointAfter(importOp);
    Value importedValue = importOp.getResult();
    Value retainedValue = createRetain(importOp, importedValue, rewriter);
    rewriter.replaceAllUsesExcept(importedValue, retainedValue,
                                  retainedValue.getDefiningOp());
    return success();
  }
};

// Exports by default assume the caller wants a retained value that they will
// need to balance. If the export happens to be the last use of the value in the
// program it'll have a +1 reference count indicating a transfer of ownership.
struct TensorExportOpPattern final
    : public OpRewritePattern<IREE::Stream::TensorExportOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(IREE::Stream::TensorExportOp exportOp,
                                PatternRewriter &rewriter) const override {
    Value exportedValue = exportOp.getSource();
    Value retainedValue =
        createRetain(exportOp, exportOp.getSource(), rewriter);
    rewriter.modifyOpInPlace(
        exportOp, [&]() { exportOp.getSourceMutable().set(retainedValue); });
    return success();
  }
};

// Callers retain operands for the callee so that we can omit the retain
// when transferring ownership to the callee. Results are assumed to have
// been retained by the callee prior to returning such that it can decide
// to transfer ownership to the caller. Tied operands are treated as
// consumed on both sides to avoid additional overhead as we know the
// result is the same as the operand.
//
// This is roughly the same as RetainOperandsPattern but handles custom arg
// attributes for indicating operand consumption (where we skip retains).
struct CallOpPattern final : public OpInterfaceRewritePattern<CallOpInterface> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;
  LogicalResult matchAndRewrite(CallOpInterface callOp,
                                PatternRewriter &rewriter) const override {
    // Gather a list of operands that need to be retained, if any.
    SmallVector<OpOperand *> retainOperands;
    ArrayAttr argAttrs = callOp.getArgAttrsAttr();
    auto tiedOp = dyn_cast<IREE::Util::TiedOpInterface>(callOp.getOperation());
    for (auto &operand : callOp.getArgOperandsMutable()) {
      if (isa<IREE::Stream::ResourceType>(operand.get().getType()) &&
          !hasConsumeAttr(argAttrs, operand.getOperandNumber()) &&
          (!tiedOp || !tiedOp.isOperandTied(operand.getOperandNumber()))) {
        retainOperands.push_back(&operand);
      }
    }
    if (retainOperands.empty()) {
      return failure(); // nothing to update
    }

    // Retain each operand and pass the retained value to the callee.
    rewriter.modifyOpInPlace(callOp, [&]() {
      for (auto *operand : retainOperands) {
        operand->set(createRetain(callOp, operand->get(), rewriter));
      }
    });

    return success();
  }
};

//===----------------------------------------------------------------------===//
// --iree-stream-automatic-reference-counting
//===----------------------------------------------------------------------===//

struct AutomaticReferenceCountingPass
    : public IREE::Stream::impl::AutomaticReferenceCountingPassBase<
          AutomaticReferenceCountingPass> {
  FrozenRewritePatternSet patterns;

  LogicalResult initialize(MLIRContext *context) override {
    RewritePatternSet patternSet(context);

    // Reference counting insertion patterns.
    patternSet.add<AsyncExecuteOpPattern>(context);
    patternSet.add<TensorImportOpPattern>(context);
    patternSet.add<TensorExportOpPattern>(context);
    patternSet.add<CallOpPattern>(context);

    // Ops without special handling that just need to ensure they retain their
    // operands or results to balance out last-use releases.
    patternSet.add<RetainResultsPattern<IREE::Util::OptimizationBarrierOp>>(
        context);
    patternSet.add<RetainResultsPattern<arith::SelectOp>>(context);
    patternSet.add<RetainResultsPattern<IREE::Util::GlobalLoadOp>>(context);
    patternSet.add<RetainOperandsPattern<IREE::Util::GlobalStoreOp>>(context);

    patterns = std::move(patternSet);
    return success();
  }

  void runOnOperation() override {
    if (getOperation()->getNumRegions() == 0) {
      return; // no-op on externs
    }

    // TODO(benvanik): verify that no retain/release ops exist (as we can't
    // handle them).

    // Insert releases for all SSA values when they leave scope.
    if (failed(insertLastUseReleases(getOperation()))) {
      // DO NOT SUBMIT make this a verification step
      return signalPassFailure();
    }

    // Insert retains to balance the last-use releases that have been inserted.
    // We do this after inserting the releases so that we have a clear use-def
    // chain for last-use analysis.
    walkAndApplyPatterns(getOperation(), patterns);
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Stream

// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Stream/Analysis/Partitioning.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-stream-schedule-execution"

namespace mlir::iree_compiler::IREE::Stream {

#define GEN_PASS_DEF_SCHEDULEEXECUTIONPASS
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h.inc"

namespace {

// Incremental builder for a partitioned region of executable work.
// Must be constructed in a topological order of all partitions.
struct ExecutePartitionBuilder {
  explicit ExecutePartitionBuilder(Block *parentBlock, size_t ordinal,
                                   Partition *partition,
                                   IRMapping &parentMapping,
                                   MLIRContext *context)
      : ordinal(ordinal), partition(partition), builder(context) {
    // Fuse the location of all ops we'll be putting in the partition.
    SmallVector<Location> locs;
    for (auto *op : partition->ops) {
      locs.push_back(op->getLoc());
    }
    auto fusedLoc = FusedLoc::get(context, locs);

    // Find the insertion point in the parent block.
    // This is at the last op in the partition.
    Operation *insertionPt = nullptr;
    for (auto *op : partition->ops) {
      if (op->getBlock() != parentBlock)
        continue;
      if (!insertionPt) {
        insertionPt = op; // first defining op
      } else if (insertionPt->isBeforeInBlock(op)) {
        insertionPt = op; // moving insertion point down
      }
    }
    OpBuilder parentBuilder(context);
    if (insertionPt) {
      parentBuilder.setInsertionPointAfter(insertionPt);
    } else {
      parentBuilder.setInsertionPointToStart(parentBlock);
    }

    // Gather operands and result types from the declared partition I/O.
    // These are values from the original block. Note that because we are
    // constructing in order we know that any results of prior partitions are
    // in the |parentMapping|.
    SmallVector<Type> resultTypes;
    SmallVector<Value> resultSizes;
    resultTypes.reserve(partition->outs.size());
    resultSizes.reserve(partition->outs.size());
    for (auto out : partition->outs) {
      resultTypes.push_back(out.getType());
      auto resultSize = IREE::Util::SizeAwareTypeInterface::queryValueSize(
          fusedLoc, out, parentBuilder);
      if (resultSize)
        resultSizes.push_back(resultSize);
    }
    SmallVector<Value> operands;
    SmallVector<Type> operandTypes;
    SmallVector<Value> operandSizes;
    operands.reserve(partition->ins.size());
    operandTypes.reserve(partition->ins.size());
    operandSizes.reserve(partition->ins.size());
    for (auto in : partition->ins) {
      if (!isa<IREE::Stream::ResourceType>(in.getType()))
        continue;
      operands.push_back(in);
      operandTypes.push_back(in.getType());
      auto operandSize = IREE::Util::SizeAwareTypeInterface::queryValueSize(
          fusedLoc, in, parentBuilder);
      if (operandSize)
        operandSizes.push_back(operandSize);
    }

    // TODO(benvanik): tie operands, or leave to canonicalization.
    SmallVector<int64_t> tiedOperands;
    executeOp = IREE::Stream::AsyncExecuteOp::create(
        parentBuilder, fusedLoc, resultTypes, resultSizes,
        /*awaitTimepoint=*/Value{}, operands, operandSizes, tiedOperands);
    if (partition->affinity) {
      executeOp.setAffinityAttr(partition->affinity);
    }

    // Add entry block and arguments.
    auto &entryBlock = executeOp.getBody().emplaceBlock();
    SmallVector<Location> operandLocs(operandTypes.size(), executeOp.getLoc());
    for (auto [operand, arg] : llvm::zip_equal(
             operands, entryBlock.addArguments(operandTypes, operandLocs))) {
      mapping.map(operand, arg);
    }
    builder = OpBuilder::atBlockBegin(&entryBlock);

    // Remap results for escaping outputs.
    for (auto [operand, result] :
         llvm::zip_equal(partition->outs, executeOp.getResults())) {
      parentMapping.map(operand, result);
    }
  }

  // Visits a block operation and clones it into the partition, if desired.
  //
  // Slightly suboptimal to be calling this on each op for each partition,
  // however we only walk the block once and constructing a multimap would be
  // way worse.
  //
  // Returns true if the operation was cloned into the partition.
  bool visit(Operation *op) {
    if (!partition->ops.contains(op))
      return false;

    // Clone the op into the partition and remap it.
    auto *clonedOp = builder.clone(*op, mapping);
    (void)clonedOp;
    LLVM_DEBUG({
      llvm::dbgs() << "Cloned op into partition " << ordinal << ": ";
      clonedOp->dump();
    });

    // If the op has the same affinity as the partition region we can strip it.
    // Note that some ops may have affinities that are more specific and we
    // want to preserve those as long as possible.
    if (auto transferOp = dyn_cast<IREE::Stream::AsyncTransferOp>(clonedOp)) {
      if (transferOp.getSourceAffinityAttr() == partition->affinity) {
        transferOp.setSourceAffinityAttr(nullptr);
      }
      if (transferOp.getTargetAffinityAttr() == partition->affinity) {
        transferOp.setTargetAffinityAttr(nullptr);
      }
    } else if (auto affinityOp =
                   dyn_cast<IREE::Stream::AffinityOpInterface>(clonedOp)) {
      if (affinityOp.getAffinityAttr() == partition->affinity) {
        affinityOp.setAffinityAttr(nullptr);
      }
    }

    return true;
  }

  IREE::Stream::AsyncExecuteOp finish() {
    // Gather results mapped into the SSA values we've cloned.
    SmallVector<Value> results;
    SmallVector<Value> resultSizes;
    results.reserve(partition->outs.size());
    resultSizes.reserve(partition->outs.size());
    for (auto oldResult : partition->outs) {
      auto newResult = mapping.lookup(oldResult);
      results.push_back(newResult);
      auto resultSize = IREE::Util::SizeAwareTypeInterface::queryValueSize(
          executeOp.getLoc(), newResult, builder);
      if (resultSize)
        resultSizes.push_back(resultSize);
    }
    IREE::Stream::YieldOp::create(builder, executeOp.getLoc(), results,
                                  resultSizes);
    return executeOp;
  }

  size_t ordinal = -1;
  Partition *partition = nullptr;
  IREE::Stream::AsyncExecuteOp executeOp;
  OpBuilder builder;
  IRMapping mapping;
};

// Sorts blocks in dominance order such that the entry block is first and
// all of the following blocks are dominated only by blocks that have come
// before them in the list.
static SmallVector<Block *, 8> sortBlocksInDominanceOrder(Region &region) {
  if (region.getBlocks().size() == 1) {
    // Dominance info cannot be computed for regions with one block.
    return {&region.getBlocks().front()};
  }

  DominanceInfo dominanceInfo(region.getParentOp());
  llvm::SmallSetVector<Block *, 8> unmarkedBlocks;
  for (auto &block : region.getBlocks()) {
    unmarkedBlocks.insert(&block);
  }
  llvm::SmallSetVector<Block *, 8> markedBlocks;
  std::function<void(Block *)> visit = [&](Block *block) {
    if (markedBlocks.count(block) > 0)
      return;
    for (auto *childBlock : dominanceInfo.getNode(block)->children()) {
      visit(childBlock->getBlock());
    }
    markedBlocks.insert(block);
  };
  while (!unmarkedBlocks.empty()) {
    visit(unmarkedBlocks.pop_back_val());
  }
  auto orderedBlocks = markedBlocks.takeVector();
  std::reverse(orderedBlocks.begin(), orderedBlocks.end());
  return orderedBlocks;
}

// Returns true if an operation can be placed into a partition (execute region).
// Only streamable operations should be partitioned; all other operations stay
// in the parent block and have their nested regions recursively processed.
//
// TODO(benvanik): allow limited scf ops inside the regions when we can ensure
// they rely on only constant values (in the future we can do indirect stuff).
static bool canPlaceInPartition(Operation *op) {
  // Only StreamableOpInterface ops can be placed into partitions.
  // This includes ops like stream.async.slice, stream.async.clone, etc.
  // but excludes stream.async.execute (which is the partition container itself)
  // and control flow ops like scf.for.
  return isa<IREE::Stream::StreamableOpInterface>(op);
}

// Tries to find a timeline-aware consumer of |value| that can provide a
// timepoint for synchronization. This looks through stream.tensor.export ops
// since timeline-aware ops (like util.call with fences) typically work with
// HAL types rather than stream resources.
// Returns the timepoint if found and otherwise returns null.
static Value tryGetConsumerTimepoint(Value value) {
  SmallVector<Operation *> usersToCheck;
  usersToCheck.append(value.getUsers().begin(), value.getUsers().end());

  // Look through tensor exports to find timeline-aware ops.
  for (auto *user : value.getUsers()) {
    if (auto exportOp = dyn_cast<IREE::Stream::TensorExportOp>(user)) {
      usersToCheck.append(exportOp.getResult().getUsers().begin(),
                          exportOp.getResult().getUsers().end());
    }
  }

  for (auto *user : usersToCheck) {
    if (auto awareOp = dyn_cast<IREE::Stream::TimelineAwareOpInterface>(user)) {
      if (awareOp.participatesInTimeline()) {
        // Let the timeline-aware op build its result timepoint.
        OpBuilder awareBuilder(user);
        awareBuilder.setInsertionPointAfter(user);
        if (auto timepoint = awareOp.buildResultTimepoint(awareBuilder)) {
          return timepoint;
        }
      }
    }
  }

  return nullptr;
}

LogicalResult processRegion(Location loc, MLIRContext *context, Region &region,
                            const PartitioningConfigAttr &configAttr) {
  for (auto *block : sortBlocksInDominanceOrder(region)) {
    // Compute a set of partitions covering all of the streamable ops in the
    // block.
    auto partitionSet = partitionStreamableOps(configAttr, block);
    if (partitionSet.empty()) {
      continue;
    }

    if (failed(partitionSet.verify(loc))) {
      return failure();
    }

    // Create partition builders for each partition.
    // We'll clone ops into each and insert them into the block at the
    // appropriate position (first use... probably).
    IRMapping mapping;
    SmallVector<ExecutePartitionBuilder> partitionBuilders;
    partitionBuilders.reserve(partitionSet.size());
    for (auto partition : llvm::enumerate(partitionSet.partitions)) {
      partitionBuilders.push_back(ExecutePartitionBuilder(
          block, partition.index(), &partition.value(), mapping, context));
    }

    // Walk over each op in the original block and find those that need to be
    // partitioned. Each partition builder may clone the op into itself. The
    // op will always be left in the original block and we'll rely on DCE to
    // remove the ones no longer required. This is not a good approach as it
    // creates a lot of new IR (up to O(op*partitions)).
    SetVector<Operation *> deadOps;
    for (auto &op : *block) {
      if (op.hasTrait<OpTrait::IsTerminator>())
        continue;
      for (auto &partitionBuilder : partitionBuilders) {
        partitionBuilder.visit(&op);
      }
      if (isa<IREE::Stream::StreamableOpInterface>(op)) {
        deadOps.insert(&op);
      }
    }

    // Apply remapping for values captured/escaping partitions.
    // We must do this per block as we'll be updating dominated block values.
    for (auto &partitionBuilder : partitionBuilders) {
      // Finish construction and insert the yield.
      auto executeOp = partitionBuilder.finish();

      OpBuilder builder(executeOp);
      builder.setInsertionPointAfter(executeOp);
      for (auto [oldResult, newResult, newResultSize] : llvm::zip_equal(
               partitionBuilder.partition->outs, executeOp.getResults(),
               executeOp.getResultSizes())) {
        // Check if any consumer is timeline-aware and can provide its own
        // timepoint - if so we reuse that to guarantee we don't ever wait.
        Value deferredTimepoint = tryGetConsumerTimepoint(oldResult);
        if (deferredTimepoint) {
          // Timeline-aware op provided a timepoint - use it.
          auto awaitOp = IREE::Stream::TimepointAwaitOp::create(
              builder, executeOp.getLoc(), newResult, newResultSize,
              deferredTimepoint);
          deadOps.insert(oldResult.getDefiningOp());
          Value toReplace = oldResult;
          toReplace.replaceAllUsesWith(awaitOp.getResults().front());
        } else {
          // Normal case - immediate await on the execute's timepoint.
          auto awaitOp = IREE::Stream::TimepointAwaitOp::create(
              builder, executeOp.getLoc(), newResult, newResultSize,
              executeOp.getResultTimepoint());
          deadOps.insert(oldResult.getDefiningOp());
          Value toReplace = oldResult;
          toReplace.replaceAllUsesWith(awaitOp.getResults().front());
        }
      }
    }
    // Before erasing preferCloneToConsumers ops, re-materialize them in nested
    // regions that reference them.
    DenseMap<Operation *, Operation *> rematerializedOps;
    for (auto *deadOp : deadOps) {
      auto streamableOp = dyn_cast<IREE::Stream::StreamableOpInterface>(deadOp);
      if (!streamableOp || !streamableOp.preferCloneToConsumers()) {
        continue;
      }

      // Find all uses in nested regions and re-materialize.
      for (auto &use : llvm::make_early_inc_range(deadOp->getUses())) {
        Operation *user = use.getOwner();
        // Only look at users in other regions.
        if (user->getBlock() == deadOp->getBlock()) {
          continue;
        }

        // Re-materialize the op at the start of the user's block.
        Block *nestedBlock = user->getBlock();
        auto *clonedOp = deadOp->clone();
        nestedBlock->getOperations().insert(nestedBlock->begin(), clonedOp);

        // Replace this use with the cloned op's result.
        Value oldValue = use.get();
        auto resultValue = dyn_cast<OpResult>(oldValue);
        unsigned resultIndex = resultValue.getResultNumber();
        use.set(clonedOp->getResult(resultIndex));
      }
    }

    for (auto *deadOp : llvm::reverse(deadOps)) {
      if (!deadOp->use_empty()) {
        // Keep ops that are still used (shouldn't happen after
        // re-materialization).
        continue;
      }
      deadOp->erase();
    }

    // Sort the ops in the execution region. This is safe because we are
    // still unaliased and SSA values imply ordering.
    mlir::sortTopologically(block);

    LLVM_DEBUG({
      llvm::dbgs() << "\nPartitions constructed:\n";
      block->dump();
    });
  }

  // Process nested regions AFTER partitioning the current block.
  // This allows nested regions to use the remapped values from parent
  // partitions. We only process operations that were NOT placed into
  // partitions above (e.g., control flow operations like scf.for).
  for (auto *block : sortBlocksInDominanceOrder(region)) {
    for (auto &op : *block) {
      // Skip ops that were placed into partitions (they're being moved).
      if (canPlaceInPartition(&op)) {
        continue;
      }

      // Process any op with regions that supports control flow, except for
      // stream.async.execute and stream.async.concurrent which are the
      // partition containers created by this pass.
      if (isa<RegionBranchOpInterface>(op)) {
        // Don't recurse into the partition containers we just created.
        if (isa<IREE::Stream::AsyncExecuteOp>(op) ||
            isa<IREE::Stream::AsyncConcurrentOp>(op)) {
          continue;
        }

        for (auto &subregion : op.getRegions()) {
          if (failed(processRegion(loc, context, subregion, configAttr)))
            return failure();
        }
      }
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// --iree-stream-schedule-execution
//===----------------------------------------------------------------------===//

struct RemoveBarriers : public OpRewritePattern<IREE::Stream::AsyncBarrierOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(IREE::Stream::AsyncBarrierOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, op.getOperand(0));
    return success();
  }
};

struct ScheduleExecutionPass
    : public IREE::Stream::impl::ScheduleExecutionPassBase<
          ScheduleExecutionPass> {
  void runOnOperation() override {
    auto *context = &getContext();
    mlir::CallableOpInterface parentOp = getOperation();
    if (!parentOp.getCallableRegion() ||
        parentOp.getCallableRegion()->empty()) {
      return;
    }

    // Lookup the optional config used to control partitioning.
    auto configAttr = IREE::Stream::PartitioningConfigAttr::lookup(parentOp);

    // Partition each block on its own. We could try to partition with the CFG
    // however that's much more complex - it's easier to handle partitioning
    // structured control flow (scf) ops. Note that we do this in dominance
    // order so that we are sure if we replace values that dominate other blocks
    // they see the correct values.
    auto &region = *parentOp.getCallableRegion();
    if (failed(processRegion(parentOp.getLoc(), context, region, configAttr)))
      return signalPassFailure();

    // Cleanup the dead ops.
    // TODO(benvanik): less work here - maybe no patterns to just force folding?
    RewritePatternSet patterns(context);
    for (auto *dialect : context->getLoadedDialects()) {
      dialect->getCanonicalizationPatterns(patterns);
    }
    for (auto op : context->getRegisteredOperations()) {
      op.getCanonicalizationPatterns(patterns, context);
    }

    // Barriers are used only for analysis and can be removed as part of
    // cleanup.
    patterns.insert<RemoveBarriers>(context);

    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    if (failed(applyPatternsGreedily(getOperation(), frozenPatterns))) {
      return signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Stream

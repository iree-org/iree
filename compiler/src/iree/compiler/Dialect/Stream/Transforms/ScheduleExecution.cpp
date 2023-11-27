// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Stream/Analysis/Partitioning.h"
#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "iree/compiler/Dialect/Stream/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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
#include "mlir/Transforms/TopologicalSortUtils.h"

#define DEBUG_TYPE "iree-stream-schedule-execution"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Stream {
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
      if (!llvm::isa<IREE::Stream::ResourceType>(in.getType()))
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
    executeOp = parentBuilder.create<IREE::Stream::AsyncExecuteOp>(
        fusedLoc, resultTypes, resultSizes, /*awaitTimepoint=*/Value{},
        operands, operandSizes, tiedOperands);
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
    if (auto affinityOp =
            dyn_cast<IREE::Stream::AffinityOpInterface>(clonedOp)) {
      if (affinityOp.getAffinity() == partition->affinity) {
        affinityOp.setAffinity(nullptr);
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
    builder.create<IREE::Stream::YieldOp>(executeOp.getLoc(), results,
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

LogicalResult processRegion(Location loc, MLIRContext *context, Region &region,
                            const PartitioningConfigAttr &configAttr) {
  for (auto *block : sortBlocksInDominanceOrder(region)) {
    // Compute a set of partitions covering all of the streamable ops in the
    // block.
    auto partitionSet = partitionStreamableOps(configAttr, block);
    if (partitionSet.empty())
      continue;
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
        // Insert one await per result. We could batch them all but that would
        // prematurely tie their lifetimes together. By having unique awaits
        // we allow propagation to move the waits further to where the values
        // are used (including right into other execution regions).
        auto awaitOp = builder.create<IREE::Stream::TimepointAwaitOp>(
            executeOp.getLoc(), newResult, newResultSize,
            executeOp.getResultTimepoint());
        if (executeOp.getAffinity().has_value()) {
          awaitOp.setAffinityAttr(executeOp.getAffinityAttr());
        }

        // Explicitly copy the Value since it is marked as const.
        Value toBeDeleted = oldResult;

        toBeDeleted.replaceAllUsesWith(awaitOp.getResults().front());
        deadOps.insert(oldResult.getDefiningOp());
      }

      // Sort the ops in the execution region. This is safe because we are
      // still unaliased and SSA values imply ordering.
      mlir::sortTopologically(block);
    }
    for (auto *deadOp : llvm::reverse(deadOps)) {
      deadOp->erase();
    }

    LLVM_DEBUG({
      llvm::dbgs() << "\nPartitions constructed:\n";
      block->dump();
    });
  }

  for (auto *block : sortBlocksInDominanceOrder(region)) {
    for (auto &op : *block) {
      if (isa<scf::SCFDialect>(op.getDialect())) {
        for (auto &subregion : op.getRegions()) {
          if (failed(processRegion(loc, context, subregion, configAttr)))
            return failure();
        }
      }
    }
  }

  return success();
}

class ScheduleExecutionPass
    : public ScheduleExecutionBase<ScheduleExecutionPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Stream::StreamDialect>();
    registry.insert<IREE::Util::UtilDialect>();
  }

  void runOnOperation() override {
    auto *context = &getContext();
    auto parentOp = getOperation();
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
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(getOperation(), frozenPatterns))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<InterfacePass<CallableOpInterface>>
createScheduleExecutionPass() {
  return std::make_unique<ScheduleExecutionPass>();
}

} // namespace Stream
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir

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
#include "llvm/ADT/BitVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/TopologicalSortUtils.h"

#define DEBUG_TYPE "iree-stream-schedule-concurrency"

namespace mlir::iree_compiler::IREE::Stream {
namespace {

// TODO(benvanik): deduplicate this with ScheduleExecution - almost all of this
// is identical.

// Incremental builder for a partitioned region of executable work.
// Must be constructed in a topological order of all partitions.
struct WavePartitionBuilder {
  explicit WavePartitionBuilder(Block *parentBlock, size_t ordinal,
                                Partition *partition, IRMapping &parentMapping,
                                MLIRContext *context)
      : ordinal(ordinal), partition(partition), builder(context) {
    // Fuse the location of all ops we'll be putting in the partition.
    SmallVector<Location> locs;
    for (auto *op : partition->ops) {
      locs.push_back(op->getLoc());
    }
    auto fusedLoc = FusedLoc::get(context, locs);

    // Find the insertion point in the parent block.
    // This is at the last op defining an input as all inputs must be available.
    Operation *insertionPt = nullptr;
    for (auto in : partition->ins) {
      auto *definingOp = in.getDefiningOp();
      if (!definingOp)
        continue;
      if (definingOp->getBlock() != parentBlock)
        continue;
      if (!insertionPt) {
        insertionPt = definingOp; // first defining op
      } else if (insertionPt->isBeforeInBlock(definingOp)) {
        insertionPt = definingOp; // moving insertion point down
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
    concurrentOp = parentBuilder.create<IREE::Stream::AsyncConcurrentOp>(
        fusedLoc, resultTypes, resultSizes, operands, operandSizes,
        tiedOperands);

    // Add entry block and arguments.
    auto &entryBlock = concurrentOp.getBody().emplaceBlock();
    SmallVector<Location> operandLocs(operandTypes.size(),
                                      concurrentOp.getLoc());
    for (auto [operand, arg] : llvm::zip_equal(
             operands, entryBlock.addArguments(operandTypes, operandLocs))) {
      mapping.map(operand, arg);
    }
    builder = OpBuilder::atBlockBegin(&entryBlock);

    // Remap results for escaping outputs.
    for (auto [operand, result] :
         llvm::zip_equal(partition->outs, concurrentOp.getResults())) {
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

    return true;
  }

  void finish() {
    // Gather results mapped into the SSA values we've cloned.
    SmallVector<Value> results;
    SmallVector<Value> resultSizes;
    results.reserve(partition->outs.size());
    resultSizes.reserve(partition->outs.size());
    for (auto oldResult : partition->outs) {
      auto newResult = mapping.lookup(oldResult);
      results.push_back(newResult);
      auto resultSize = IREE::Util::SizeAwareTypeInterface::queryValueSize(
          concurrentOp.getLoc(), newResult, builder);
      if (resultSize)
        resultSizes.push_back(resultSize);
    }
    builder.create<IREE::Stream::YieldOp>(concurrentOp.getLoc(), results,
                                          resultSizes);
  }

  size_t ordinal = -1;
  Partition *partition = nullptr;
  IREE::Stream::AsyncConcurrentOp concurrentOp;
  OpBuilder builder;
  IRMapping mapping;
};

class ScheduleConcurrencyPass
    : public ScheduleConcurrencyBase<ScheduleConcurrencyPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Stream::StreamDialect>();
    registry.insert<IREE::Util::UtilDialect>();
  }

  void runOnOperation() override {
    auto parentOp = getOperation();
    if (!parentOp.getCallableRegion() ||
        parentOp.getCallableRegion()->empty()) {
      return;
    }
    for (auto executeOp :
         parentOp.getCallableRegion()->getOps<IREE::Stream::AsyncExecuteOp>()) {
      if (failed(runOnRegion(executeOp)))
        return signalPassFailure();
    }
  }

  LogicalResult runOnRegion(IREE::Stream::AsyncExecuteOp parentOp) {
    if (parentOp.getBody().empty()) {
      return success();
    }
    auto *block = &parentOp.getBody().front();

    // Lookup the optional config used to control partitioning.
    auto configAttr = IREE::Stream::PartitioningConfigAttr::lookup(parentOp);

    // Compute a set of partitions covering all of the streamable ops in the
    // execution region.
    auto waveSet = partitionRegionConcurrency(configAttr, block);
    if (waveSet.empty())
      return success();
    if (failed(waveSet.verify(parentOp.getLoc())))
      return failure();

    // Create partition builders for each partition.
    // We'll clone ops into each and insert them into the block at the
    // appropriate position (first use... probably).
    IRMapping mapping;
    SmallVector<WavePartitionBuilder> partitionBuilders;
    partitionBuilders.reserve(waveSet.size());
    for (auto partition : llvm::enumerate(waveSet.partitions)) {
      if (partition.value().ops.size() == 1)
        continue;
      partitionBuilders.push_back(WavePartitionBuilder(block, partition.index(),
                                                       &partition.value(),
                                                       mapping, &getContext()));
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
      bool handled = false;
      for (auto &partitionBuilder : partitionBuilders) {
        handled = partitionBuilder.visit(&op) || handled;
      }
      if (handled) {
        deadOps.insert(&op);
      }
    }

    // Apply remapping for values captured/escaping partitions.
    // We must do this per block as we'll be updating dominated block values.
    for (auto &partitionBuilder : partitionBuilders) {
      for (auto [oldResult, newResult] :
           llvm::zip_equal(partitionBuilder.partition->outs,
                           partitionBuilder.concurrentOp.getResults())) {
        // Explicitly copy the Value since the original is marked as const.
        Value toBeDeleted = oldResult;

        toBeDeleted.replaceAllUsesWith(newResult);
        deadOps.insert(oldResult.getDefiningOp());
      }
      partitionBuilder.finish();
    }
    for (auto *deadOp : llvm::reverse(deadOps)) {
      deadOp->erase();
    }

    // Sort the ops in the execution region as they may have gotten out of order
    // during partitioning. This is safe because we are still unaliased and SSA
    // values imply ordering.
    mlir::sortTopologically(block);

    LLVM_DEBUG({
      llvm::dbgs() << "\nWaves constructed:\n";
      block->dump();
    });
    return success();
  }
};

} // namespace

std::unique_ptr<InterfacePass<CallableOpInterface>>
createScheduleConcurrencyPass() {
  return std::make_unique<ScheduleConcurrencyPass>();
}

} // namespace mlir::iree_compiler::IREE::Stream

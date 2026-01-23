// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VM/Analysis/LinearScan/LiveIntervals.h"

#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/Dialect/VM/Analysis/ValueLiveness.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"

#define DEBUG_TYPE "iree-vm-live-intervals"

namespace mlir::iree_compiler {

static Attribute getStrArrayAttr(Builder &builder,
                                 ArrayRef<std::string> values) {
  return builder.getStrArrayAttr(llvm::map_to_vector<8>(
      values, [](const std::string &value) { return StringRef(value); }));
}

// static
LogicalResult LiveIntervals::annotateIR(IREE::VM::FuncOp funcOp) {
  LiveIntervals liveIntervals;
  if (failed(liveIntervals.build(funcOp))) {
    return funcOp.emitOpError() << "failed to build live intervals";
  }

  Builder builder(funcOp.getContext());

  // Annotate each block with its instruction range.
  for (auto *block : liveIntervals.getBlockOrder()) {
    if (block->empty()) {
      continue;
    }

    uint32_t blockStart = liveIntervals.getInstructionIndex(&block->front());
    uint32_t blockEnd = liveIntervals.getInstructionIndex(&block->back());

    // Annotate first op with block range.
    block->front().setAttr(
        "block_range",
        builder.getStringAttr(
            llvm::formatv("[{0}, {1}]", blockStart, blockEnd).str()));
  }

  // Annotate each operation with its instruction index and result intervals.
  for (auto &block : funcOp.getBlocks()) {
    for (auto &op : block.getOperations()) {
      uint32_t opIndex = liveIntervals.getInstructionIndex(&op);
      op.setAttr("op_index", builder.getI32IntegerAttr(opIndex));

      if (op.getNumResults() == 0) {
        continue;
      }

      SmallVector<std::string, 4> intervalStrs;
      for (auto result : op.getResults()) {
        const LiveInterval *interval = liveIntervals.getInterval(result);
        if (interval) {
          intervalStrs.push_back(llvm::formatv("[{0}, {1}]{2}", interval->start,
                                               interval->end,
                                               interval->isRef ? " ref" : "")
                                     .str());
        } else {
          intervalStrs.push_back("none");
        }
      }
      op.setAttr("result_intervals", getStrArrayAttr(builder, intervalStrs));
    }
  }

  // Also annotate block arguments.
  for (auto &block : funcOp.getBlocks()) {
    SmallVector<std::string, 4> argIntervalStrs;
    for (auto blockArg : block.getArguments()) {
      const LiveInterval *interval = liveIntervals.getInterval(blockArg);
      if (interval) {
        argIntervalStrs.push_back(llvm::formatv("[{0}, {1}]{2}",
                                                interval->start, interval->end,
                                                interval->isRef ? " ref" : "")
                                      .str());
      } else {
        argIntervalStrs.push_back("none");
      }
    }
    if (!argIntervalStrs.empty()) {
      block.front().setAttr("block_arg_intervals",
                            getStrArrayAttr(builder, argIntervalStrs));
    }
  }

  return success();
}

LogicalResult LiveIntervals::build(IREE::VM::FuncOp funcOp) {
  intervals_.clear();
  sortedByStart_.clear();
  valueToInterval_.clear();
  opToIndex_.clear();
  blockOrder_.clear();
  instructionCount_ = 0;

  // Run liveness analysis first.
  ValueLiveness liveness;
  if (failed(liveness.recalculate(funcOp))) {
    return funcOp.emitError() << "failed to compute value liveness";
  }

  // Sort blocks in dominance order (reverse post-order).
  sortBlocksInDominanceOrder(funcOp);

  // Number all instructions.
  numberInstructions();

  // Build intervals from liveness.
  buildIntervals(liveness);

  // Sort intervals by start position.
  sortedByStart_.clear();
  sortedByStart_.reserve(intervals_.size());
  for (const auto &interval : intervals_) {
    sortedByStart_.push_back(&interval);
  }
  llvm::stable_sort(sortedByStart_,
                    [](const LiveInterval *a, const LiveInterval *b) {
                      return a->start < b->start;
                    });

  LLVM_DEBUG({
    llvm::dbgs() << "=== Live Intervals for " << funcOp.getName() << " ===\n";
    dump();
  });

  return success();
}

const LiveInterval *LiveIntervals::getInterval(Value value) const {
  auto it = valueToInterval_.find(value);
  if (it == valueToInterval_.end()) {
    return nullptr;
  }
  return &intervals_[it->second];
}

uint32_t LiveIntervals::getInstructionIndex(Operation *op) const {
  auto it = opToIndex_.find(op);
  assert(it != opToIndex_.end() && "operation not in index map");
  return it->second;
}

void LiveIntervals::sortBlocksInDominanceOrder(IREE::VM::FuncOp funcOp) {
  blockOrder_.clear();

  if (funcOp.getBlocks().size() == 1) {
    // Dominance info cannot be computed for regions with one block.
    blockOrder_.push_back(&funcOp.getBlocks().front());
    return;
  }

  DominanceInfo dominanceInfo(funcOp);
  llvm::SmallSetVector<Block *, 8> unmarkedBlocks;
  for (auto &block : funcOp.getBlocks()) {
    unmarkedBlocks.insert(&block);
  }
  llvm::SmallSetVector<Block *, 8> markedBlocks;
  std::function<void(Block *)> visit = [&](Block *block) {
    if (markedBlocks.contains(block)) {
      return;
    }
    for (auto *childBlock : dominanceInfo.getNode(block)->children()) {
      visit(childBlock->getBlock());
    }
    markedBlocks.insert(block);
  };
  while (!unmarkedBlocks.empty()) {
    visit(unmarkedBlocks.pop_back_val());
  }
  blockOrder_ = markedBlocks.takeVector();
  std::reverse(blockOrder_.begin(), blockOrder_.end());
}

void LiveIntervals::numberInstructions() {
  opToIndex_.clear();
  instructionCount_ = 0;

  for (auto *block : blockOrder_) {
    for (auto &op : block->getOperations()) {
      opToIndex_[&op] = instructionCount_++;
    }
  }
}

void LiveIntervals::buildIntervals(ValueLiveness &liveness) {
  intervals_.clear();
  valueToInterval_.clear();

  // Process blocks in dominance order.
  for (auto *block : blockOrder_) {
    // Process block arguments.
    for (auto blockArg : block->getArguments()) {
      if (valueToInterval_.contains(blockArg)) {
        continue;
      }

      // Block arguments are "defined" at the start of the block.
      // We use the first op's index as the start.
      uint32_t start = block->empty() ? 0 : opToIndex_[&block->front()];
      uint32_t end = findLastUse(blockArg, liveness);

      LiveInterval interval;
      interval.value = blockArg;
      interval.start = start;
      interval.end = std::max(start, end);
      interval.isRef = isa<IREE::VM::RefType>(blockArg.getType());
      interval.byteWidth =
          interval.isRef
              ? 0
              : IREE::Util::getRoundedElementByteWidth(blockArg.getType());

      valueToInterval_[blockArg] = intervals_.size();
      intervals_.push_back(interval);
    }

    // Process operation results.
    for (auto &op : block->getOperations()) {
      uint32_t opIndex = opToIndex_[&op];

      for (auto result : op.getResults()) {
        if (valueToInterval_.contains(result)) {
          continue;
        }

        uint32_t start = opIndex;
        uint32_t end = findLastUse(result, liveness);

        LiveInterval interval;
        interval.value = result;
        interval.start = start;
        interval.end = std::max(start, end);
        interval.isRef = isa<IREE::VM::RefType>(result.getType());
        interval.byteWidth =
            interval.isRef
                ? 0
                : IREE::Util::getRoundedElementByteWidth(result.getType());

        valueToInterval_[result] = intervals_.size();
        intervals_.push_back(interval);
      }
    }
  }
}

uint32_t LiveIntervals::findLastUse(Value value, ValueLiveness &liveness) {
  // Start with definition point.
  uint32_t lastIndex = 0;
  if (auto defOp = value.getDefiningOp()) {
    lastIndex = opToIndex_[defOp];
  } else if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    Block *block = blockArg.getOwner();
    if (!block->empty()) {
      lastIndex = opToIndex_[&block->front()];
    }
  }

  // Find the maximum use index, INCLUDING discards.
  //
  // We must include discards because they operate on registers, not SSA values.
  // After register allocation, two SSA values may share the same register if
  // their intervals don't overlap. If we excluded discards from the interval,
  // a register could be reused before the discard executes, causing the discard
  // to kill the wrong value.
  //
  // Example: %a ends at inst 50, %b starts at inst 54, discard(%a) at inst 54.
  // Without including discards: %a interval=[..50], %b interval=[54..], both
  // get r0, discard(r0) kills %b instead of %a.
  //
  // TODO: For optimal register usage, move discard insertion to post-regalloc
  // where we can reason about register liveness instead of SSA value liveness.
  for (auto &use : value.getUses()) {
    Operation *useOp = use.getOwner();
    auto it = opToIndex_.find(useOp);
    if (it != opToIndex_.end()) {
      lastIndex = std::max(lastIndex, it->second);
    }
  }

  // If the value is live-out of any block, extend to the end of that block.
  // This handles cross-block liveness (e.g., values used in successors).
  for (auto *block : blockOrder_) {
    // Check if value is live-in to this block (means it's live-out of
    // predecessor).
    for (auto liveIn : liveness.getBlockLiveIns(block)) {
      if (liveIn == value) {
        // Value is live into this block, so it must be live until some use
        // in this block or a successor. Find the last use in this block,
        // including discards (see comment above about why).
        for (auto &op : block->getOperations()) {
          for (auto &operand : op.getOpOperands()) {
            if (operand.get() == value) {
              lastIndex = std::max(lastIndex, opToIndex_[&op]);
            }
          }
        }
        // If the value is still needed (live-out), extend to block terminator.
        // We check by seeing if any successor also has this value in liveIn.
        Operation *terminator = block->getTerminator();
        for (auto *successor : terminator->getSuccessors()) {
          for (auto succLiveIn : liveness.getBlockLiveIns(successor)) {
            if (succLiveIn == value) {
              // Value is live-out of this block.
              lastIndex = std::max(lastIndex, opToIndex_[terminator]);
              break;
            }
          }
        }
      }
    }
  }

  return lastIndex;
}

void LiveIntervals::dump() const {
  llvm::dbgs() << "Block order: ";
  for (auto *block : blockOrder_) {
    llvm::dbgs() << block << " ";
  }
  llvm::dbgs() << "\n";

  llvm::dbgs() << "Total instructions: " << instructionCount_ << "\n";
  llvm::dbgs() << "Intervals (sorted by start):\n";

  for (const auto *interval : sortedByStart_) {
    llvm::dbgs() << "  " << interval->value << " [" << interval->start << ", "
                 << interval->end << "]" << (interval->isRef ? " ref" : " i32")
                 << "\n";
  }
}

} // namespace mlir::iree_compiler

// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VM/Analysis/ValueLiveness.h"

#include <algorithm>
#include <cstring>

#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace iree_compiler {

// static
LogicalResult ValueLiveness::annotateIR(IREE::VM::FuncOp funcOp) {
  ValueLiveness liveness;
  if (failed(liveness.recalculate(funcOp))) {
    funcOp.emitOpError()
        << "failed to compute value liveness information for function";
    return failure();
  }

  // Build mapping of Operations to values live during them.
  // This is not needed normally so we calculate it only in this debugging case.
  DenseMap<Operation *, llvm::SmallSetVector<Value, 8>> livePerOp;
  for (auto &liveRange : liveness.liveRanges_) {
    auto value = liveRange.getFirst();
    auto &bitVector = liveRange.getSecond();
    for (int opIndex : bitVector.set_bits()) {
      auto *op = liveness.opsInOrder_[opIndex];
      livePerOp[op].insert(value);
    }
  }

  // Block names are their order in the function.
  DenseMap<Block *, int> blockOrdinals;
  for (auto &block : funcOp.getBlocks()) {
    int ordinal = blockOrdinals.size();
    blockOrdinals[std::addressof(block)] = ordinal;
  }

  // Keep asm state to make getting the SSA value names fast.
  OpPrintingFlags printingFlags;
  printingFlags.elideLargeElementsAttrs(1);
  AsmState asmState(funcOp, printingFlags);

  // Gather attributes for each op before we actually add them. We do this so
  // that it's easier to slice out the results for printing without also
  // including the attributes we ourselves are trying to add.
  Builder builder(funcOp.getContext());
  DenseMap<Operation *, NamedAttrList> livenessAttrs(livePerOp.size());
  auto addValuesAttr = [&](Operation &op, StringRef attrName,
                           const llvm::SmallSetVector<Value, 8> &values) {
    SmallVector<StringAttr, 8> valueNames;
    for (auto value : values) {
      std::string str;
      if (auto blockArg = llvm::dyn_cast<BlockArgument>(value)) {
        if (blockArg.getOwner()->isEntryBlock()) {
          str = llvm::formatv("%arg{0}", blockArg.getArgNumber());
        } else {
          str =
              llvm::formatv("%bb{0}_arg{1}", blockOrdinals[blockArg.getOwner()],
                            blockArg.getArgNumber());
        }
      } else {
        llvm::raw_string_ostream os(str);
        value.print(os, asmState);
        str = os.str();
      }

      int equalsIndex = str.find(" =");
      if (equalsIndex != std::string::npos) {  // heh
        auto results = str.substr(0, equalsIndex);
        valueNames.push_back(builder.getStringAttr(results));
      } else {
        valueNames.push_back(builder.getStringAttr(str));
      }
    }

    // Sort attributes by name (as SmallSetVector is unordered).
    std::sort(
        valueNames.begin(), valueNames.end(),
        +[](const StringAttr &a, const StringAttr &b) {
          return a.getValue().compare_insensitive(b.getValue()) < 0;
        });
    SmallVector<Attribute, 8> valueNameAttrs;
    for (auto attr : valueNames) {
      valueNameAttrs.push_back(attr);
    }

    livenessAttrs[&op].set(attrName, builder.getArrayAttr(valueNameAttrs));
  };

  for (auto &block : funcOp.getBlocks()) {
    auto &blockLiveness = liveness.blockLiveness_[&block];

    addValuesAttr(block.front(), "block_live_in", blockLiveness.liveIn);
    addValuesAttr(block.front(), "block_live", blockLiveness.live);
    addValuesAttr(block.front(), "block_live_out", blockLiveness.liveOut);
    addValuesAttr(block.front(), "block_defined", blockLiveness.defined);

    // Add per-op live values.
    for (auto &op : block.getOperations()) {
      addValuesAttr(op, "live", livePerOp[&op]);
    }
  }

  // Markup all ops with their attributes.
  for (auto &opAttrs : livenessAttrs) {
    for (auto nameAttr : opAttrs.getSecond().getAttrs()) {
      opAttrs.getFirst()->setAttr(nameAttr.getName(), nameAttr.getValue());
    }
  }

  return success();
}

LogicalResult ValueLiveness::recalculate(IREE::VM::FuncOp funcOp) {
  opsInOrder_.clear();
  opOrdering_.clear();
  blockLiveness_.clear();
  liveRanges_.clear();

  calculateOpOrdering(funcOp);
  if (failed(computeLivenessSets(funcOp))) {
    return funcOp.emitError() << "failed to compute liveness sets";
  }
  if (failed(computeLiveIntervals(funcOp))) {
    return funcOp.emitError() << "failed to compute live intervals";
  }

  return success();
}

void ValueLiveness::calculateOpOrdering(IREE::VM::FuncOp funcOp) {
  int nextOrdinal = 0;
  for (auto &block : funcOp.getBlocks()) {
    for (auto &op : block.getOperations()) {
      opOrdering_[&op] = nextOrdinal++;
      opsInOrder_.push_back(&op);
    }
  }
}

LogicalResult ValueLiveness::computeInitialLivenessSets(
    IREE::VM::FuncOp funcOp) {
  for (auto &block : funcOp.getBlocks()) {
    auto &blockSets = blockLiveness_[&block];

    // Block arguments are defined within the block, technically.
    for (auto blockArg : block.getArguments()) {
      blockSets.defined.insert(blockArg);
    }

    // Add all operands as live uses and all results as definitions.
    for (auto &op : block.getOperations()) {
      for (auto &operand : op.getOpOperands()) {
        blockSets.live.insert(operand.get());
      }
      for (auto result : op.getResults()) {
        blockSets.defined.insert(result);
        for (auto &use : result.getUses()) {
          if (use.getOwner()->getBlock() != &block) {
            // Value escapes this block.
            blockSets.liveOut.insert(result);
          }
        }
      }
    }
  }
  return success();
}

LogicalResult ValueLiveness::computeLivenessSets(IREE::VM::FuncOp funcOp) {
  if (failed(computeInitialLivenessSets(funcOp))) {
    return failure();
  }

  llvm::SmallSetVector<Block *, 32> worklist;
  worklist.insert(&funcOp.getBlocks().front());

  // Compute live-in set for each block.
  for (auto &block : funcOp.getBlocks()) {
    auto &blockSets = blockLiveness_[&block];
    blockSets.liveIn = blockSets.live;
    blockSets.liveIn.set_union(blockSets.liveOut);
    blockSets.liveIn.set_subtract(blockSets.defined);

    // If there are live-ins they may need to be propagated to predecessors.
    if (!blockSets.liveIn.empty()) {
      worklist.insert(block.getPredecessors().begin(),
                      block.getPredecessors().end());
    }
  }

  // Propagate liveness until a fixed point is reached.
  while (!worklist.empty()) {
    Block *block = worklist.pop_back_val();
    auto &blockSets = blockLiveness_[block];

    // Compute a new live-out set for the block.
    auto liveOut = blockSets.liveOut;
    for (Block *successor : block->getSuccessors()) {
      liveOut.set_union(blockLiveness_[successor].liveIn);
    }
    blockSets.liveOut = liveOut;

    // Compute the live-in set for the block.
    auto liveIn = liveOut;
    liveIn.set_union(blockSets.live);
    liveIn.set_subtract(blockSets.defined);

    // Propagate the liveness to predecessors if the live-in set changed.
    if (blockSets.liveIn.size() != liveIn.size()) {
      blockSets.liveIn = liveIn;
      worklist.insert(block->getPredecessors().begin(),
                      block->getPredecessors().end());
    }
  }

  return success();
}

LogicalResult ValueLiveness::computeLiveIntervals(IREE::VM::FuncOp funcOp) {
  // Adds a live range for |value| from |start| to |end|.
  // Both |start| and |end| must be within the same Block.
  auto addLiveRange = [this](Value value, Operation *start, Operation *end) {
    assert(start->getBlock() == end->getBlock());
    auto &bitVector = liveRanges_[value];
    bitVector.resize(opOrdering_.size());
    bitVector.set(opOrdering_[start], opOrdering_[end] + 1);
  };

  for (auto &block : funcOp.getBlocks()) {
    auto &blockSets = blockLiveness_[&block];

    // Handle values that escape the block.
    for (auto value : blockSets.liveOut) {
      if (blockSets.liveIn.count(value)) {
        // Live in and live out covers the entire block.
        addLiveRange(value, &block.front(), &block.back());
      } else {
        // Live out but not live in implies defined in the block.
        Operation *firstUse =
            value.getDefiningOp() ? value.getDefiningOp() : &block.front();
        addLiveRange(value, firstUse, &block.back());
      }
    }

    // Handle values entering the block and dying within.
    for (auto value : blockSets.liveIn) {
      if (blockSets.liveOut.count(value)) continue;
      Operation *lastUse = &block.front();
      for (auto &use : value.getUses()) {
        if (use.getOwner()->getBlock() != &block) continue;
        if (lastUse == use.getOwner()) continue;
        if (lastUse->isBeforeInBlock(use.getOwner())) {
          lastUse = use.getOwner();
        }
      }
      addLiveRange(value, &block.front(), lastUse);
    }

    // Handle values defined within the block and not escaping.
    for (auto value : blockSets.defined) {
      if (blockSets.liveOut.count(value)) continue;
      Operation *firstUse =
          value.getDefiningOp() ? value.getDefiningOp() : &block.front();
      Operation *lastUse = firstUse;
      for (auto &use : value.getUses()) {
        if (use.getOwner()->getBlock() != &block) continue;
        if (lastUse->isBeforeInBlock(use.getOwner())) {
          lastUse = use.getOwner();
        }
      }
      addLiveRange(value, firstUse, lastUse);
    }
  }

  return success();
}

ArrayRef<Value> ValueLiveness::getBlockLiveIns(Block *block) {
  auto &blockSets = blockLiveness_[block];
  return blockSets.liveIn.getArrayRef();
}

bool ValueLiveness::isLastValueUse(Value value, Operation *useOp) {
  auto &blockSets = blockLiveness_[useOp->getBlock()];
  if (blockSets.liveOut.count(value)) {
    // Value is escapes the block the useOp is in so it is definitely not the
    // last use.
    return false;
  }
  int opOrdinal = opOrdering_[useOp];
  auto &liveRange = liveRanges_[value];
  if (!useOp->hasTrait<OpTrait::IsTerminator>() &&
      liveRange.test(opOrdinal + 1)) {
    // The value is still live within the block after the useOp.
    return false;
  }
  return true;
}

bool ValueLiveness::isLastValueUse(Value value, Operation *useOp,
                                   int operandIndex) {
  if (!isLastValueUse(value, useOp)) {
    return false;
  }
  for (auto &operand : llvm::reverse(useOp->getOpOperands())) {
    if (operand.get() == value) {
      // Compare the queried operand index with the last use index.
      return operandIndex >= operand.getOperandNumber();
    }
  }
  assert(false && "value not used by operand");
  return false;
}

}  // namespace iree_compiler
}  // namespace mlir

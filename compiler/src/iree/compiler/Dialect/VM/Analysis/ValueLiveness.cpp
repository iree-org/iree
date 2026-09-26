// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VM/Analysis/ValueLiveness.h"

#include <algorithm>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"

namespace mlir::iree_compiler {

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
  auto addLiveRange = [&](Value value) {
    for (Operation *op : liveness.liveness_->resolveLiveness(value)) {
      livePerOp[op].insert(value);
    }
  };
  for (auto &block : funcOp.getBlocks()) {
    for (auto blockArg : block.getArguments()) {
      addLiveRange(blockArg);
    }
    for (auto &op : block.getOperations()) {
      for (auto result : op.getResults()) {
        addLiveRange(result);
      }
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
      if (auto blockArg = dyn_cast<BlockArgument>(value)) {
        if (blockArg.getOwner()->isEntryBlock()) {
          str = llvm::formatv("%arg{}", blockArg.getArgNumber());
        } else {
          str = llvm::formatv("%bb{}_arg{}", blockOrdinals[blockArg.getOwner()],
                              blockArg.getArgNumber());
        }
      } else {
        llvm::raw_string_ostream os(str);
        value.print(os, asmState);
        str = os.str();
      }

      int equalsIndex = str.find(" =");
      if (equalsIndex != std::string::npos) { // heh
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
    const auto *blockInfo = liveness.liveness_->getLiveness(&block);

    // Values live on block entry/exit come from the upstream analysis.
    llvm::SmallSetVector<Value, 8> liveIn(blockInfo->in().begin(),
                                          blockInfo->in().end());
    llvm::SmallSetVector<Value, 8> liveOut(blockInfo->out().begin(),
                                           blockInfo->out().end());

    // Local sets: all values defined within the block (block args and op
    // results) and all values used by ops within the block.
    llvm::SmallSetVector<Value, 8> defined;
    llvm::SmallSetVector<Value, 8> live;
    for (auto blockArg : block.getArguments()) {
      defined.insert(blockArg);
    }
    for (auto &op : block.getOperations()) {
      for (auto operand : op.getOperands()) {
        live.insert(operand);
      }
      for (auto result : op.getResults()) {
        defined.insert(result);
      }
    }

    addValuesAttr(block.front(), "block_live_in", liveIn);
    addValuesAttr(block.front(), "block_live", live);
    addValuesAttr(block.front(), "block_live_out", liveOut);
    addValuesAttr(block.front(), "block_defined", defined);

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
  liveness_.emplace(funcOp.getOperation());
  return success();
}

bool ValueLiveness::isLiveIn(Block *block, Value value) {
  const auto *blockInfo = liveness_->getLiveness(block);
  assert(blockInfo && "block not covered by the liveness analysis (was the "
                      "IR modified after recalculate?)");
  return blockInfo->isLiveIn(value);
}

bool ValueLiveness::isLiveOut(Block *block, Value value) {
  const auto *blockInfo = liveness_->getLiveness(block);
  assert(blockInfo && "block not covered by the liveness analysis (was the "
                      "IR modified after recalculate?)");
  return blockInfo->isLiveOut(value);
}

bool ValueLiveness::isLastValueUse(Value value, Operation *useOp) {
  assert(liveness_->getLiveness(useOp->getBlock()) &&
         "block not covered by the liveness analysis (was the IR modified "
         "after recalculate?)");
  return liveness_->isDeadAfter(value, useOp);
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

// Determines if an operand is the last "real" use of a ref value.
//
// "Real" uses exclude vm.discard_refs operations, which are cleanup markers
// rather than actual uses. This distinction is critical for MOVE bit logic:
// - If the only uses after a branch are discard ops, the branch IS the last
//   real use and should get MOVE semantics.
// - Without this, we'd incorrectly retain refs that are about to be discarded.
//
// Returns true when all conditions are met:
// 1. useOp is not a discard op (discards are never "real" uses)
// 2. value is not in the block's live-out set (doesn't escape)
// 3. No non-discard uses exist after useOp in the same block
// 4. operandIndex is >= the last operand using this value (handles multi-use)
//
// The multi-use check (condition 4) ensures that for ops like:
//   vm.call @foo(%ref, %ref)  // same ref used twice
// Only the LAST operand (index 1) is considered the "last use", so MOVE
// is only applied once.
bool ValueLiveness::isLastRealValueUse(Value value, Operation *useOp,
                                       int operandIndex) {
  // Discards are never "real" uses - they just clean up.
  if (isa<IREE::VM::DiscardRefsOp>(useOp)) {
    return false;
  }

  // Check if value escapes block.
  if (isLiveOut(useOp->getBlock(), value)) {
    // Value is in liveOut. For non-terminators, this means the value has uses
    // after this block, so it's not at last use.
    if (!useOp->hasTrait<OpTrait::IsTerminator>()) {
      return false;
    }
    // For branch terminators where the value is forwarded as a successor
    // operand, discards in successors are just cleanup - they don't count as
    // "real" uses since the branch transfers ownership. But for other
    // terminators (like vm.call.yieldable), the value is NOT forwarded -
    // discards in successors ARE real uses of the original value.
    bool valueIsSuccessorOperand = false;
    if (auto branchOp = dyn_cast<BranchOpInterface>(useOp)) {
      for (unsigned i = 0; i < useOp->getNumSuccessors(); ++i) {
        SuccessorOperands succOperands = branchOp.getSuccessorOperands(i);
        for (Value operand : succOperands.getForwardedOperands()) {
          if (operand == value) {
            valueIsSuccessorOperand = true;
            break;
          }
        }
        if (valueIsSuccessorOperand) {
          break;
        }
      }
    }
    // Check if the value escapes to any successor blocks.
    // Only check uses in immediate successor blocks. Uses in other blocks
    // (e.g. predecessors or the definition block) are on prior control-flow
    // edges and don't affect whether MOVE is safe at this branch point.
    SmallPtrSet<Block *, 4> successorBlocks;
    for (unsigned i = 0; i < useOp->getNumSuccessors(); ++i) {
      Block *succBlock = useOp->getSuccessor(i);
      successorBlocks.insert(succBlock);
      // If the value flows THROUGH a successor (both liveIn and liveOut),
      // it reaches non-immediate successors we can't enumerate here.
      // Conservatively prevent MOVE. This is precise for current VM pipeline
      // invariants: MaterializeRefDiscards places exactly one discard per
      // path at value death points, so liveIn && liveOut implies real
      // (non-discard) uses downstream.
      if (isLiveIn(succBlock, value) && isLiveOut(succBlock, value)) {
        return false;
      }
    }
    for (auto &use : value.getUses()) {
      Operation *userOp = use.getOwner();
      Block *userBlock = userOp->getBlock();
      if (userBlock == useOp->getBlock()) {
        continue;
      }
      // Skip uses in non-successor blocks (e.g. predecessor or definition
      // blocks). These are on prior control-flow edges, not forward ones.
      if (!successorBlocks.contains(userBlock)) {
        continue;
      }
      // Use is in a successor block. If it's a discard AND the value was
      // forwarded as a successor operand, skip it (ownership transferred).
      // Otherwise, it's a real use and the value escapes.
      bool isDiscardOfForwardedValue =
          isa<IREE::VM::DiscardRefsOp>(userOp) && valueIsSuccessorOperand;
      if (!isDiscardOfForwardedValue) {
        return false;
      }
    }
    // All uses in other blocks were discards of forwarded values. Fall through
    // to check if this is the last use within the block.
  }

  // Walk forward to see if any non-discard uses exist after this op.
  for (auto it = ++Block::iterator(useOp); it != useOp->getBlock()->end();
       ++it) {
    for (OpOperand &operand : it->getOpOperands()) {
      if (operand.get() == value && !isa<IREE::VM::DiscardRefsOp>(&*it)) {
        return false; // Real use found after this op
      }
    }
  }

  // Handle same-value multiple operands (only last operand is "last use").
  for (auto &operand : llvm::reverse(useOp->getOpOperands())) {
    if (operand.get() == value) {
      return operandIndex >= operand.getOperandNumber();
    }
  }
  return false;
}

} // namespace mlir::iree_compiler

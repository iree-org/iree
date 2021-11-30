// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Stream/Analysis/Partitioning.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/PatternMatch.h"

#define DEBUG_TYPE "iree-stream-partitioning"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Stream {

#ifndef NDEBUG

void dumpPartition(Partition &partition, AsmState &state) {
  llvm::dbgs() << " INS:\n  ";
  llvm::interleaveComma(partition.ins, llvm::dbgs(), [&](Value in) {
    in.printAsOperand(llvm::dbgs(), state);
  });
  llvm::dbgs() << "\n OUTS:\n  ";
  llvm::interleaveComma(partition.outs, llvm::dbgs(), [&](Value out) {
    out.printAsOperand(llvm::dbgs(), state);
  });
  llvm::dbgs() << "\n OPS:\n";
  for (auto *op : llvm::reverse(partition.ops)) {
    llvm::dbgs() << "  ";
    op->print(llvm::dbgs(), state);
    llvm::dbgs() << "\n";
  }
}

void Partition::dump(Operation *parentOp) {
  AsmState state(parentOp);
  dumpPartition(*this, state);
}

void PartitionSet::dump(Operation *parentOp) {
  AsmState state(parentOp);
  for (auto partition : llvm::enumerate(partitions)) {
    llvm::dbgs() << "PARTITION[" << partition.index() << "]:\n";
    dumpPartition(partition.value(), state);
  }
}

#else
void Partition::dump(Operation *parentOp) {}
void PartitionSet::dump(Operation *parentOp) {}
#endif  // !NDEBUG

LogicalResult Partition::verify(Location loc) {
  // Ensure values are defined either by other ops in the partition or are
  // declared as inputs.
  SetVector<Value> defValues;
  for (auto *op : ops) {
    for (auto result : op->getResults()) {
      defValues.insert(result);
    }
  }
  for (auto *op : ops) {
    for (auto operand : op->getOperands()) {
      if (!ins.contains(operand) && !defValues.contains(operand)) {
        return mlir::emitError(loc)
               << "operand not declared in partition inputs or by an op within "
                  "the partition";
      }
    }
  }

  // Ensure all outputs come from ops in the partition (or are pass-through
  // operands, though those are silly).
  for (auto out : outs) {
    if (!ins.contains(out) && !defValues.contains(out)) {
      return mlir::emitError(loc) << "output not defined by an op within the "
                                     "partition (or captured)";
    }
  }

  return success();
}

LogicalResult PartitionSet::verify(Location loc) {
  // Verify each partition is consistent.
  for (auto &partition : partitions) {
    if (failed(partition.verify(loc))) return failure();
  }

  // Ensure no partitions duplicate escaping values as we need a single def to
  // remap the value in the parent block.
  SetVector<Value> outs;
  for (auto &partition : partitions) {
    for (auto out : partition.outs) {
      if (outs.contains(out)) {
        return mlir::emitError(loc)
               << "duplicate value found in partition set outputs";
      }
      outs.insert(out);
    }
  }

  // Ensure a correct topological order of partitions. This only checks the
  // order of the partitions and not any ops that aren't covered. We do this
  // by walking backwards and checking that no partition captures values
  // escaping any partitions after it.
  SetVector<Value> declaredBelow;
  for (auto &partition : llvm::reverse(partitions)) {
    for (auto in : partition.ins) {
      if (declaredBelow.contains(in)) {
        return mlir::emitError(loc) << "partition set out of order; value "
                                       "captured declared as escaping below";
      }
    }
    for (auto out : partition.outs) {
      declaredBelow.insert(out);
    }
  }

  return success();
}

void PartitionSet::topologicalSort() {
  if (partitions.empty()) return;

  SetVector<Partition *> unsortedSet;
  DenseMap<Value, SmallVector<Partition *>> consumers;
  for (auto &partition : partitions) {
    unsortedSet.insert(&partition);
    for (auto in : partition.ins) {
      consumers[in].push_back(&partition);
    }
  }

  struct DFSState {
    SmallVector<Partition *, 16> topologicalCounts;
    DenseSet<Partition *> seen;
  } state;
  std::function<void(Partition *)> postorderWalk;
  postorderWalk = [&](Partition *current) {
    for (auto out : current->outs) {
      for (auto *consumer : consumers[out]) {
        postorderWalk(consumer);
      }
    }
    auto it = state.seen.insert(current);
    if (/*inserted=*/it.second) {
      if (unsortedSet.contains(current)) {
        state.topologicalCounts.push_back(current);
      }
    }
  };
  for (auto *partition : unsortedSet) postorderWalk(partition);

  SmallVector<Partition> sortedSet;
  sortedSet.reserve(partitions.size());
  for (auto *partition : llvm::reverse(state.topologicalCounts)) {
    sortedSet.push_back(std::move(*partition));
  }
  partitions = std::move(sortedSet);
}

PartitionSet partitionStreamableOps(IREE::Stream::PartitioningConfigAttr config,
                                    Block *block) {
  // Only one algorithm today.
  return partitionStreamableOpsReference(config, block);
}

PartitionSet partitionRegionConcurrency(
    IREE::Stream::PartitioningConfigAttr config, Block *block) {
  // Only one algorithm today.
  return partitionRegionConcurrencyReference(config, block);
}

}  // namespace Stream
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

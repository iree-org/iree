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
#include "mlir/Interfaces/ControlFlowInterfaces.h"

#define DEBUG_TYPE "iree-stream-partitioning"

namespace mlir::iree_compiler::IREE::Stream {

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

// Collects all values consumed by an operation, including those used in nested
// regions (e.g. scf.for bodies). This is a conservative analysis that walks all
// regions to find consumed values.
void collectConsumedValues(Operation *rootOp,
                           SetVector<Value> &consumedValues) {
  SmallVector<Operation *> worklist;
  DenseSet<Operation *> visitedOps;
  worklist.push_back(rootOp);
  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();

    // Skip if already visited.
    if (!visitedOps.insert(op).second) {
      continue;
    }

    // Collect direct operands.
    for (auto operand : op->getOperands()) {
      consumedValues.insert(operand);
    }

    // Conservatively walk all regions to find consumed values.
    // We can't use RegionBranchOpInterface::getEntrySuccessorRegions because
    // it requires compile-time constant operands to determine reachability,
    // but we need to handle runtime values. For hazard detection we need to be
    // conservative and assume all regions may execute.
    for (auto &region : op->getRegions()) {
      for (auto &block : region) {
        for (auto &nestedOp : block) {
          worklist.push_back(&nestedOp);
        }
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Partition data structures
//===----------------------------------------------------------------------===//

#ifndef NDEBUG

void dumpPartition(Partition &partition, AsmState &asmState) {
  if (partition.affinity) {
    llvm::dbgs() << " AFFINITY: ";
    partition.affinity.dump();
    llvm::dbgs() << "\n";
  }
  llvm::dbgs() << " INS:\n    ";
  llvm::interleave(
      partition.ins, llvm::dbgs(),
      [&](Value in) { in.print(llvm::dbgs(), asmState); }, "\n    ");
  llvm::dbgs() << "\n OUTS:\n    ";
  llvm::interleave(
      partition.outs, llvm::dbgs(),
      [&](Value out) { out.print(llvm::dbgs(), asmState); }, "\n    ");
  llvm::dbgs() << "\n OPS:\n";
  for (auto *op : llvm::reverse(partition.ops)) {
    llvm::dbgs() << "  ";
    op->print(llvm::dbgs(), asmState);
    llvm::dbgs() << "\n";
  }
}

void Partition::dump(AsmState &asmState) { dumpPartition(*this, asmState); }

void PartitionSet::dump(AsmState &asmState) {
  for (auto partition : llvm::enumerate(partitions)) {
    llvm::dbgs() << "PARTITION[" << partition.index() << "]:\n";
    dumpPartition(partition.value(), asmState);
  }
}

#else
void Partition::dump(AsmState &asmState) {}
void PartitionSet::dump(AsmState &asmState) {}
#endif // !NDEBUG

LogicalResult Partition::verify(Location loc) {
  // Ensure all ops are compatible with the partition affinity.
  for (auto *op : ops) {
    if (auto affinityOp = dyn_cast<IREE::Stream::AffinityOpInterface>(op)) {
      if (!IREE::Stream::AffinityAttr::areCompatible(
              affinity, affinityOp.getAffinityAttr())) {
        return op->emitError("op affinity ")
               << affinityOp.getAffinityAttr()
               << " is not compatible with the partition affinity " << affinity;
      }
    }
  }

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

  // Check for circular dependencies: input ops using partition outputs.
  // This can happen if an operation with nested regions (like scf.for) is
  // incorrectly placed as a partition input when its nested regions actually
  // consume partition outputs.
  for (auto in : ins) {
    // Only check ops, not bare values.
    auto definingOp = in.getDefiningOp();
    if (!definingOp)
      continue;

    // Collect all values used by this input op (including nested regions).
    SetVector<Value> inputConsumedValues;
    collectConsumedValues(definingOp, inputConsumedValues);

    // Check if any consumed value comes from partition outputs.
    //
    // NOTE: We only check outputs, not all defValues. When operations are
    // duplicated across partitions (e.g., clones with preferCloneToConsumers),
    // the same operation may define values in multiple partitions. An input
    // operation might consume a value that is defined in THIS partition's
    // defValues, but if that value is not exported (not in outputs), the input
    // operation must be using a copy from another partition that also defines
    // it. Checking defValues would create false positives for such cases.
    for (auto consumedValue : inputConsumedValues) {
      if (outs.contains(consumedValue)) {
        return mlir::emitError(loc)
               << "circular dependency: input operation uses partition output "
               << consumedValue
               << " - this indicates the operation should be inside the "
                  "partition";
      }
    }
  }

  return success();
}

LogicalResult PartitionSet::verify(Location loc) {
  // Verify each partition is consistent.
  for (auto &partition : partitions) {
    if (failed(partition.verify(loc)))
      return failure();
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
                                       "captured declared as escaping below: "
                                    << in;
      }
    }
    for (auto out : partition.outs) {
      declaredBelow.insert(out);
    }
  }

  return success();
}

void PartitionSet::topologicalSort() {
  if (partitions.empty())
    return;

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
  for (auto *partition : unsortedSet)
    postorderWalk(partition);

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

PartitionSet
partitionRegionConcurrency(IREE::Stream::PartitioningConfigAttr config,
                           Block *block) {
  // Only one algorithm today.
  return partitionRegionConcurrencyReference(config, block);
}

} // namespace mlir::iree_compiler::IREE::Stream

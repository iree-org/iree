// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Stream/Analysis/Partitioning.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/PatternMatch.h"

#define DEBUG_TYPE "iree-stream-partitioning"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Stream {

// This is terrible. See Stream/Analysis/Partition.h for a description of what
// a real implementation would do. We want cost modeling for tie breakers when
// an op could be in multiple partitions, cloning for ops that are not worth
// spanning partitions (like splats), etc.
PartitionSet partitionStreamableOpsReference(
    IREE::Stream::PartitioningConfigAttr config, Block *block) {
  PartitionSet partitionSet;

  struct PartitionBuilder {
    unsigned ordinal;
    // Affinity of the partition.
    IREE::Stream::AffinityAttr affinity;
    // Ops present in the partition; ops may be present in multiple partitions.
    SetVector<Operation *> ops;
  };
  SmallVector<std::unique_ptr<PartitionBuilder>> builders;
  llvm::BitVector usableBuilders;

  struct OpInfo {
    // Which partitions the op is contained within.
    llvm::BitVector membership;
    // Which partitions transitively depend on this operation.
    llvm::BitVector hazards;
  };
  DenseMap<Operation *, OpInfo> opInfos;

  for (auto &op : llvm::reverse(*block)) {
    // Skip constants; they just add noise (and since they are heavily CSE'd
    // they have lots of users to test).
    if (op.hasTrait<OpTrait::ConstantLike>()) {
      LLVM_DEBUG(llvm::dbgs() << "(ignoring constant)\n");
      continue;
    } else if (!isa<IREE::Stream::StreamableOpInterface>(op)) {
      // Not a streamable op. If it has side-effects then we force a hazard on
      // all builders so that we don't move ops across it.
      if (!mlir::wouldOpBeTriviallyDead(&op)) {
        LLVM_DEBUG({
          llvm::dbgs() << "Side-effecting op forcing flush and freeze:\n";
          op.dump();
        });
        usableBuilders.reset();
      }
      // Even though not a streamable op we still want to track it below.
    }

    // Initialize op info for this op - whether streamable or not. We track
    // transitive hazards on each op. Note that thanks to the ordering of ops
    // in SSA form (_reversed here!_) we know that once we visit this op no
    // partition created after it can ever depend on it if it doesn't here. This
    // lets us keep the bitvectors small.
    auto &opInfo = opInfos[&op];
    opInfo.hazards.reserve(builders.size() + 1);
    opInfo.hazards.resize(builders.size(), /*t=*/false);

    IREE::Stream::AffinityAttr affinityAttr;
    if (auto affinityOp = dyn_cast<IREE::Stream::AffinityOpInterface>(op)) {
      affinityAttr = affinityOp.getAffinity();
    }

    LLVM_DEBUG({
      llvm::dbgs() << "====\nPartitioning op:\n";
      op.dump();
    });

    // Set bits for each partition this op may be able to be placed into.
    // We prune the set based on whether the users are part of a transitive
    // dependency chain down the use-def chain to a partition.
    llvm::BitVector consumers(builders.size(), /*t=*/false);
    for (auto user : op.getUsers()) {
      auto &userInfo = opInfos[user];
      LLVM_DEBUG({
        llvm::dbgs() << "Testing user:\n";
        user->dump();
        for (auto membershipOrdinal : userInfo.membership.set_bits()) {
          llvm::dbgs() << "  member of partition " << membershipOrdinal << "\n";
        }
        for (auto hazardOrdinal : userInfo.hazards.set_bits()) {
          llvm::dbgs() << "  hazard w/ partition " << hazardOrdinal << "\n";
        }
      });
      consumers |= userInfo.membership;
      opInfo.hazards |= userInfo.membership;
      opInfo.hazards |= userInfo.hazards;
    }
    llvm::BitVector candidates(builders.size(), /*t=*/true);
    candidates ^= opInfo.hazards;
    candidates |= consumers;
    candidates &= usableBuilders;

    // Prune candidates that do not have a compatible affinity.
    for (auto ordinal : candidates.set_bits()) {
      if (!IREE::Stream::AffinityAttr::areCompatible(
              affinityAttr, builders[ordinal]->affinity)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Candidate partition " << ordinal << " incompatible\n");
        candidates.reset(ordinal);
      }
    }

    // If this op is not streamable then bail here; we've still setup the hazard
    // map for following iteration.
    auto streamableOp = dyn_cast<IREE::Stream::StreamableOpInterface>(op);
    if (!streamableOp) {
      LLVM_DEBUG(llvm::dbgs() << "Not streamable (skip)\n");
      continue;
    }

    // First see which partitions are consuming this that we can also safely
    // move in to.
    consumers &= candidates;

    opInfo.membership.reserve(builders.size() + 1);
    opInfo.membership.resize(builders.size(), /*t=*/false);

    // If we have one or more consumers we should go into those first.
    if (consumers.any()) {
      // If we are a clonable op (like splat) clone us into every partition.
      // Otherwise we just pick the first we find (probably a bad heuristic).
      bool shouldClone = streamableOp.preferCloneToConsumers();
      for (auto consumerOrdinal : consumers.set_bits()) {
        LLVM_DEBUG(llvm::dbgs() << "Cloning into consumer partition "
                                << consumerOrdinal << "\n");
        builders[consumerOrdinal]->ops.insert(&op);
        opInfo.membership.set(consumerOrdinal);
        opInfo.hazards.reset(consumerOrdinal);
        if (!shouldClone) break;
      }
      LLVM_DEBUG(llvm::dbgs() << "Handled streamable (continue)\n");
      continue;
    }

    // No consumers - if there's any candidate then we'll go into that.
    int firstCandidateOrdinal = candidates.find_first();
    if (firstCandidateOrdinal != -1) {
      LLVM_DEBUG(llvm::dbgs() << "Moving to first candidate partition "
                              << firstCandidateOrdinal << " (continue)\n");
      builders[firstCandidateOrdinal]->ops.insert(&op);
      opInfo.membership.set(firstCandidateOrdinal);
      opInfo.hazards.reset(firstCandidateOrdinal);
      continue;
    }

    // Mark the op as having hazards against all other partitions.
    if (!builders.empty()) {
      opInfo.hazards.set(0, builders.size() - 1);
    }

    // Create a new partition just for this op.
    opInfo.membership.resize(opInfo.membership.size() + 1, /*t=*/true);
    auto builder = std::make_unique<PartitionBuilder>();
    builder->ordinal = builders.size();
    builder->affinity = affinityAttr;
    builder->ops.insert(&op);
    LLVM_DEBUG(llvm::dbgs()
               << "Created partition " << builder->ordinal << "\n");
    builders.push_back(std::move(builder));
    usableBuilders.resize(builders.size(), /*t=*/true);
  }

  // Emit partitions in forward order (as they are topologically sorted in
  // reverse order from our bottom-up walk).
  for (auto &builder : llvm::reverse(builders)) {
    Partition partition;

    SetVector<Value> consumedValues;
    SetVector<Value> producedValues;
    SetVector<Value> escapingValues;
    for (auto *op : llvm::reverse(builder->ops)) {
      for (auto operand : op->getOperands()) {
        consumedValues.insert(operand);
      }
      for (auto result : op->getResults()) {
        producedValues.insert(result);
        // TODO(benvanik): optimize this - creates n^2/nlogn behavior.
        for (auto user : result.getUsers()) {
          if (!builder->ops.contains(user)) {
            escapingValues.insert(result);
          }
        }
      }
    }
    consumedValues.set_subtract(producedValues);
    partition.ins = consumedValues;
    partition.outs = escapingValues;

    partition.ops = std::move(builder->ops);
    partitionSet.partitions.push_back(std::move(partition));
  }

  LLVM_DEBUG(partitionSet.dump(block->getParentOp()));

  return partitionSet;
}

// This looks to extract a single level of concurrency; we should be recursively
// dividing the block to identify both serial and concurrent regions.
PartitionSet partitionRegionConcurrencyReference(
    IREE::Stream::PartitioningConfigAttr config, Block *block) {
  PartitionSet waveSet;

  auto favor = config ? config.getFavor().getValue()
                      : IREE::Stream::Favor::MinPeakMemory;
  if (favor == IREE::Stream::Favor::Debug) {
    // Disable partitioning when favoring debugability.
    return waveSet;
  }

  struct PartitionBuilder {
    unsigned ordinal;
    // Ops present in the wave; ops may be present in multiple waves.
    SetVector<Operation *> ops;
  };
  SmallVector<std::unique_ptr<PartitionBuilder>> builders;

  struct OpInfo {
    // Which waves the op is contained within.
    llvm::BitVector membership;
    // Which waves transitively depend on this operation.
    llvm::BitVector hazards;
  };
  DenseMap<Operation *, OpInfo> opInfos;

  for (auto &op : llvm::reverse(*block)) {
    // Skip constants; they just add noise (and since they are heavily CSE'd
    // they have lots of users to test).
    if (op.hasTrait<OpTrait::ConstantLike>()) {
      LLVM_DEBUG(llvm::dbgs() << "(ignoring constant)\n");
      continue;
    }

    // Initialize op info for this op - whether streamable or not. We track
    // transitive hazards on each op. Note that thanks to the ordering of ops
    // in SSA form (_reversed here!_) we know that once we visit this op no
    // wave created after it can ever depend on it if it doesn't here. This
    // lets us keep the bitvectors small.
    auto &opInfo = opInfos[&op];
    opInfo.hazards.reserve(builders.size() + 1);
    opInfo.hazards.resize(builders.size(), /*t=*/false);

    LLVM_DEBUG({
      llvm::dbgs() << "====\nPartitioning op:\n";
      op.dump();
    });

    // Set bits for each wave this op may be able to be placed into.
    // We prune the set based on whether the users are part of a transitive
    // dependency chain down the use-def chain to a wave.
    llvm::BitVector consumers(builders.size(), /*t=*/false);
    for (auto user : op.getUsers()) {
      auto &userInfo = opInfos[user];
      LLVM_DEBUG({
        llvm::dbgs() << "Testing user:\n";
        user->dump();
        for (auto membershipOrdinal : userInfo.membership.set_bits()) {
          llvm::dbgs() << "  member of wave " << membershipOrdinal << "\n";
        }
        int lastHazardOrdinal = userInfo.hazards.find_last();
        if (lastHazardOrdinal != -1) {
          llvm::dbgs() << "  hazard w/ waves 0-" << lastHazardOrdinal << "\n";
        }
      });
      consumers |= userInfo.membership;
      opInfo.hazards |= userInfo.membership;
      opInfo.hazards |= userInfo.hazards;
    }
    llvm::BitVector candidates(builders.size(), /*t=*/true);
    candidates ^= opInfo.hazards;

    // If this op is not streamable then bail here; we've still setup the hazard
    // map for following iteration.
    auto streamableOp = dyn_cast<IREE::Stream::StreamableOpInterface>(op);
    if (!streamableOp || streamableOp.isMetadata()) {
      LLVM_DEBUG(llvm::dbgs() << "Not streamable/is subview (skip)\n");
      continue;
    }

    opInfo.membership.reserve(builders.size() + 1);
    opInfo.membership.resize(builders.size(), /*t=*/false);

    // No consumers - if there's any candidate then we'll go into that.
    int firstCandidateOrdinal = favor == IREE::Stream::Favor::MinPeakMemory
                                    ? candidates.find_first()
                                    : candidates.find_last();
    if (firstCandidateOrdinal != -1) {
      LLVM_DEBUG(llvm::dbgs() << "Moving to last candidate wave "
                              << firstCandidateOrdinal << " (continue)\n");
      builders[firstCandidateOrdinal]->ops.insert(&op);
      opInfo.membership.set(firstCandidateOrdinal);
      opInfo.hazards.set(0, firstCandidateOrdinal);
      opInfo.hazards.reset(firstCandidateOrdinal);
      continue;
    }

    // Mark the op as having hazards against all other waves.
    opInfo.hazards.set(0, builders.size());

    // Create a new wave just for this op.
    opInfo.membership.resize(opInfo.membership.size() + 1, /*t=*/true);
    auto builder = std::make_unique<PartitionBuilder>();
    builder->ordinal = builders.size();
    builder->ops.insert(&op);
    LLVM_DEBUG(llvm::dbgs() << "Created wave " << builder->ordinal << "\n");
    builders.push_back(std::move(builder));
  }

  // Emit waves in forward order (as they are topologically sorted in
  // reverse order from our bottom-up walk).
  for (auto &builder : llvm::reverse(builders)) {
    Partition wave;

    SetVector<Value> consumedValues;
    SetVector<Value> producedValues;
    SetVector<Value> escapingValues;
    for (auto *op : llvm::reverse(builder->ops)) {
      for (auto operand : op->getOperands()) {
        consumedValues.insert(operand);
      }
      for (auto result : op->getResults()) {
        producedValues.insert(result);
        // TODO(benvanik): optimize this - creates n^2/nlogn behavior.
        for (auto user : result.getUsers()) {
          if (!builder->ops.contains(user)) {
            escapingValues.insert(result);
          }
        }
      }
    }
    consumedValues.set_subtract(producedValues);
    wave.ins = consumedValues;
    wave.outs = escapingValues;

    wave.ops = std::move(builder->ops);
    waveSet.partitions.push_back(std::move(wave));
  }

  LLVM_DEBUG(waveSet.dump(block->getParentOp()));

  return waveSet;
}

}  // namespace Stream
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Stream/Analysis/Partitioning.h"
#include "iree/compiler/Dialect/Stream/Analysis/ResourceHazards.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/PatternMatch.h"

#define DEBUG_TYPE "iree-stream-partitioning"

namespace mlir::iree_compiler::IREE::Stream {

// Returns an AsmState at the ancestor to |block| that is isolated from above.
// Returns nullptr if debug dumps of partitioning is disabled.
static std::unique_ptr<AsmState> getRootAsmState(Block *block) {
  LLVM_DEBUG({
    auto *rootOp = block->getParentOp();
    while (auto parentOp = rootOp->getParentOp()) {
      if (!isa<IREE::Stream::TimelineOpInterface>(parentOp) &&
          parentOp->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
        rootOp = parentOp;
        break;
      }
      rootOp = parentOp;
    }
    return std::make_unique<AsmState>(rootOp);
  });
  return nullptr;
}

// This is terrible. See Stream/Analysis/Partition.h for a description of what
// a real implementation would do. We want cost modeling for tie breakers when
// an op could be in multiple partitions, cloning for ops that are not worth
// spanning partitions (like splats), etc.
PartitionSet
partitionStreamableOpsReference(IREE::Stream::PartitioningConfigAttr config,
                                Block *block) {
  PartitionSet partitionSet;

  struct OpInfo {
    // Which partitions the op is contained within.
    llvm::BitVector membership;
    // Which partitions transitively depend on this operation.
    llvm::BitVector hazards;
  };
  DenseMap<Operation *, OpInfo> opInfos;

  struct PartitionBuilder {
    unsigned ordinal;
    // Affinity of the partition.
    IREE::Stream::AffinityAttr affinity;
    // Ops present in the partition; ops may be present in multiple partitions.
    SetVector<Operation *> ops;
    // Ops that were cloned and are known not to have their values escape.
    DenseSet<Operation *> clonedOps;
    // Which partitions transitively depend on this partition.
    llvm::BitVector hazards;
    void insert(Operation *op, OpInfo &opInfo) {
      if (auto affinityOp = dyn_cast<IREE::Stream::AffinityOpInterface>(op)) {
        affinity = affinity ? affinity.joinAND(affinityOp.getAffinityAttr())
                            : affinityOp.getAffinityAttr();
      }
      opInfo.membership.set(ordinal);
      if (opInfo.hazards.size() > ordinal)
        opInfo.hazards.reset(ordinal);
      ops.insert(op);
      hazards |= opInfo.hazards;
    }
  };
  SmallVector<std::unique_ptr<PartitionBuilder>> builders;
  llvm::BitVector usableBuilders;

  auto willCreateCircularDependencyBetweenPartitions =
      [&](unsigned sourceOrdinal, unsigned targetOrdinal) -> bool {
    // Returns:
    // If we are to make partition with ordinal targetOrdinal to
    // depend on partition with ordinal sourceOrdinal,
    // will this create a circular dependency.
    if (sourceOrdinal == targetOrdinal)
      return false;
    return builders[sourceOrdinal]->hazards.size() > targetOrdinal &&
           builders[sourceOrdinal]->hazards[targetOrdinal];
  };

  auto canAddOpToPartition = [&](Operation &op, OpInfo &opInfo,
                                 unsigned partitionOrdinal) {
    auto streamableOp = dyn_cast<IREE::Stream::StreamableOpInterface>(op);
    if (!streamableOp)
      return false;
    IREE::Stream::AffinityAttr affinityAttr;
    if (auto affinityOp = dyn_cast<IREE::Stream::AffinityOpInterface>(op))
      affinityAttr = affinityOp.getAffinityAttr();
    if (!IREE::Stream::AffinityAttr::canExecuteTogether(
            affinityAttr, builders[partitionOrdinal]->affinity))
      return false;

    bool preferCloneToConsumers = streamableOp.preferCloneToConsumers();
    llvm::BitVector *opHazards = nullptr;
    llvm::BitVector opHazardsInCandidatePartition;
    if (preferCloneToConsumers) {
      // If we are cloning we care only about users that are a part of the
      // candidate partition.
      // Here we would need to walk further down the users if a user is also
      // cloned into the partition. This will be useful if we have a block of
      // cloneable ops. If left like that, other than the inefficiency,
      // it should not produce invalid partitioning.
      opHazards = &opHazardsInCandidatePartition;
      for (auto user : op.getUsers()) {
        if (builders[partitionOrdinal]->ops.contains(user))
          opHazardsInCandidatePartition |= opInfos[user].hazards;
      }
    } else
      opHazards = &opInfo.hazards;

    for (auto opHazardOrdinal : opHazards->set_bits()) {
      if (partitionOrdinal < opHazardOrdinal) {
        // Reject partition ordering that would require partition sorting.
        // TODO: It is probably more optimal to reorder the partitions after
        // their formation based on their dependency graph instead of rejecting
        // here. Since this is considered not a good partitioning algorithm
        // and will probably get removed, we leave it like that.
        return false;
      }
      // Check for formation of circular dependency between partitions.
      if (willCreateCircularDependencyBetweenPartitions(opHazardOrdinal,
                                                        partitionOrdinal))
        return false;
    }
    return true;
  };

  auto asmState = getRootAsmState(block);

  llvm::DenseMap<Operation *, llvm::SmallVector<Operation *>> syncOps;

  for (auto &op : llvm::reverse(*block)) {
    // Skip constants; they just add noise (and since they are heavily CSE'd
    // they have lots of users to test).
    if (op.hasTrait<OpTrait::ConstantLike>()) {
      LLVM_DEBUG(llvm::dbgs() << "(ignoring constant)\n");
      continue;
    } else if (isa<IREE::Util::GlobalStoreOpInterface>(op)) {
      // We ignore global stores as they are unobservable within an execution
      // region - we must still block on loads though.
      LLVM_DEBUG(llvm::dbgs() << "(ignoring global store)\n");
      continue;
    } else if (!isa<IREE::Stream::StreamableOpInterface>(op)) {
      // Not a streamable op. If it has side-effects then we force a hazard on
      // all builders so that we don't move ops across it.
      if (!mlir::wouldOpBeTriviallyDead(&op)) {
        LLVM_DEBUG({
          llvm::dbgs() << "Side-effecting op forcing flush and freeze:\n";
          op.print(llvm::dbgs(), *asmState);
          llvm::dbgs() << "\n";
        });
        usableBuilders.reset();
      }
      // Even though not a streamable op we still want to track it below.
    }

    // Synchronizing operations should join with their producers if the producer
    // is streamable.
    if (dyn_cast<IREE::Stream::AsyncBarrierOp>(op) ||
        dyn_cast<IREE::Stream::AsyncTransferOp>(op)) {
      auto producer = op.getOperand(0).getDefiningOp();
      auto streamable =
          dyn_cast_or_null<IREE::Stream::StreamableOpInterface>(producer);
      if (streamable) {
        if (!syncOps.contains(producer))
          syncOps[producer] = llvm::SmallVector<Operation *>();
        syncOps[producer].push_back(&op);
        continue;
      }
    }

    // Initialize op info for this op - whether streamable or not. We track
    // transitive hazards on each op. Note that thanks to the ordering of ops
    // in SSA form (_reversed here!_) we know that once we visit this op no
    // partition created after it can ever depend on it if it doesn't here. This
    // lets us keep the bitvectors small.
    auto &opInfo = opInfos[&op];
    opInfo.hazards.reserve(builders.size() + 1);
    opInfo.hazards.resize(builders.size(), /*t=*/false);

    LLVM_DEBUG({
      llvm::dbgs() << "====\nPartitioning op:\n";
      op.print(llvm::dbgs(), *asmState);
      llvm::dbgs() << "\n";
    });

    // Set bits for each partition this op may be able to be placed into.
    // We prune the set based on whether the users are part of a transitive
    // dependency chain down the use-def chain to a partition.
    llvm::BitVector consumers(builders.size(), /*t=*/false);
    for (auto user : op.getUsers()) {
      auto userInfoIt = opInfos.find(user);
      if (userInfoIt == opInfos.end())
        continue;
      auto &userInfo = userInfoIt->second;
      LLVM_DEBUG({
        llvm::dbgs() << "Testing user:\n";
        user->print(llvm::dbgs(), *asmState);
        llvm::dbgs() << "\n";
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

    for (auto syncOp : syncOps[&op]) {
      for (auto user : syncOp->getUsers()) {
        auto userInfoIt = opInfos.find(user);
        if (userInfoIt == opInfos.end())
          continue;
        auto &userInfo = userInfoIt->second;
        opInfo.hazards |= userInfo.membership;
        opInfo.hazards |= userInfo.hazards;
        consumers.reset();
      }
    }

    // For any sync ops not use this ops results we need to put in a
    // non-consumer block:
    llvm::BitVector candidates(builders.size(), /*t=*/true);
    candidates ^= opInfo.hazards;
    candidates |= consumers;
    candidates &= usableBuilders;

    // Prune candidates that do not have a compatible affinity.
    for (auto ordinal : candidates.set_bits()) {
      if (!canAddOpToPartition(op, opInfo, ordinal)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Candidate partition " << ordinal << " incompatible\n");
        candidates.reset(ordinal);
      }
    }

    for (auto syncOp : syncOps[&op]) {
      for (auto ordinal : candidates.set_bits()) {
        if (!canAddOpToPartition(*syncOp, opInfo, ordinal)) {
          LLVM_DEBUG(llvm::dbgs()
                     << "Candidate partition " << ordinal << " incompatible\n");
          candidates.reset(ordinal);
        }
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
    if (consumers.any())
      candidates = consumers;

    opInfo.membership.reserve(builders.size() + 1);
    opInfo.membership.resize(builders.size(), /*t=*/false);

    // No consumers - if there's any candidate then we'll go into that.
    int firstCandidateOrdinal = candidates.find_first();
    if (firstCandidateOrdinal == -1) {
      // Mark the op as having hazards against all other partitions.
      // It is better to be safe than incorrect, especially with our current
      // minimal test coverage. It's not always safe to reorder things - if
      // anything we are unlikely to be conservative enough here - for example,
      // if there's a stream.resource.load of a resource or a global we can't
      // move anything that may affect that resource or global. This
      // partitioning was designed to be conservative because debugging such
      // issues is really difficult.
      if (!builders.empty()) {
        opInfo.hazards.set(0, builders.size() - 1);
      }

      // Create a new partition just for this op.
      opInfo.membership.resize(opInfo.membership.size() + 1, /*t=*/true);
      auto builder = std::make_unique<PartitionBuilder>();
      builder->ordinal = builders.size();
      LLVM_DEBUG(llvm::dbgs()
                 << "Created partition " << builder->ordinal << "\n");
      builders.push_back(std::move(builder));
      usableBuilders.resize(builders.size(), /*t=*/true);
      firstCandidateOrdinal = builders.size() - 1;
    }

    auto &builder = builders[firstCandidateOrdinal];

    // If we have synchronization operations we can place in the last block:
    for (auto syncOp : syncOps[&op]) {
      builder->insert(syncOp, opInfo);
    }

    LLVM_DEBUG(llvm::dbgs() << "Moving to first candidate partition "
                            << firstCandidateOrdinal << " (continue)\n");
    // If we are a clonable op (like splat) clone us into every partition.
    // Otherwise we just pick the first we find (probably a bad heuristic).
    if (consumers.count() > 1 && streamableOp.preferCloneToConsumers()) {
      for (auto consumerOrdinal : consumers.set_bits()) {
        LLVM_DEBUG(llvm::dbgs() << "Cloning into consumer partition "
                                << consumerOrdinal << "\n");
        auto &consumerBuilder = builders[consumerOrdinal];
        consumerBuilder->insert(&op, opInfo);
        consumerBuilder->clonedOps.insert(&op);
      }
    } else {
      builder->insert(&op, opInfo);
    }
  }

  // Ops cloned into multiple partitions may still escape if there are
  // non-streamable consumers. We need to make sure we only let one result
  // escape.
  DenseSet<Operation *> clonedEscapingOps;

  // Emit partitions in forward order (as they are topologically sorted in
  // reverse order from our bottom-up walk).
  for (auto &builder : llvm::reverse(builders)) {
    Partition partition;

    SetVector<Value> consumedValues;
    SetVector<Value> producedValues;
    SetVector<Value> escapingValues;
    for (auto *op : llvm::reverse(builder->ops)) {
      bool didCloneEscape = false;
      for (auto operand : op->getOperands()) {
        consumedValues.insert(operand);
      }
      for (auto result : op->getResults()) {
        producedValues.insert(result);

        // Cloned ops default to local usage but may still have users outside
        // of any partition and need to escape.
        if (builder->clonedOps.contains(op)) {
          // We only want to have one partition produce the value and track ones
          // we've already produced via clonedEscapingOps.
          if (!clonedEscapingOps.contains(op)) {
            for (auto user : result.getUsers()) {
              if (!isa<IREE::Stream::StreamableOpInterface>(user)) {
                escapingValues.insert(result);
                didCloneEscape = true;
                break;
              }
            }
          }
        } else {
          // TODO(benvanik): optimize this - creates n^2/nlogn behavior.
          for (auto user : result.getUsers()) {
            if (!builder->ops.contains(user)) {
              escapingValues.insert(result);
            }
          }
        }
      }
      if (didCloneEscape) {
        clonedEscapingOps.insert(op);
      }
    }
    consumedValues.set_subtract(producedValues);
    partition.affinity = builder->affinity;
    partition.ins = consumedValues;
    partition.outs = escapingValues;

    partition.ops = std::move(builder->ops);
    partitionSet.partitions.push_back(std::move(partition));
  }

  LLVM_DEBUG(partitionSet.dump(*asmState));

  return partitionSet;
}

// This looks to extract a single level of concurrency; we should be recursively
// dividing the block to identify both serial and concurrent regions.
PartitionSet
partitionRegionConcurrencyReference(IREE::Stream::PartitioningConfigAttr config,
                                    Block *block) {
  PartitionSet waveSet;

  auto favor = config.getFavor().getValue();
  if (favor == IREE::Stream::Favor::Debug) {
    // Disable partitioning when favoring debuggability.
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

  auto asmState = getRootAsmState(block);

  // Run analysis - if it fails then we'll just be conservative.
  IREE::Stream::ResourceHazardAnalysis hazardAnalysis(block->getParentOp());
  if (failed(hazardAnalysis.run())) {
    LLVM_DEBUG(llvm::dbgs() << "WARNING: resource hazard analysis failed; "
                               "conservatively scheduling\n");
  }

  for (auto &op : llvm::reverse(*block)) {
    // Skip constants; they just add noise (and since they are heavily CSE'd
    // they have lots of users to test).
    if (op.hasTrait<OpTrait::ConstantLike>()) {
      LLVM_DEBUG(llvm::dbgs() << "(ignoring constant)\n");
      continue;
    }

    // NOTE: it's ok if this op is not streamable as we still need to track the
    // hazards for other ops that it may use/may use it.
    auto streamableOp = dyn_cast<IREE::Stream::StreamableOpInterface>(op);

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
      op.print(llvm::dbgs(), *asmState);
      llvm::dbgs() << "\n";
    });

    // Set bits for each wave this op may be able to be placed into.
    // We prune the set based on whether the users are part of a transitive
    // dependency chain down the use-def chain to a wave.
    for (auto user : op.getUsers()) {
      auto userInfoIt = opInfos.find(user);
      if (userInfoIt == opInfos.end())
        continue;
      auto &userInfo = userInfoIt->second;
      LLVM_DEBUG({
        llvm::dbgs() << "Testing user:\n";
        user->print(llvm::dbgs(), *asmState);
        llvm::dbgs() << "\n";
        for (auto membershipOrdinal : userInfo.membership.set_bits()) {
          llvm::dbgs() << "  member of wave " << membershipOrdinal << "\n";
        }
        int lastHazardOrdinal = userInfo.hazards.find_last();
        if (lastHazardOrdinal != -1) {
          llvm::dbgs() << "  hazard w/ waves 0-" << lastHazardOrdinal << "\n";
        }
      });
      bool hazardPresent = hazardAnalysis.hasHazard(streamableOp, user);
      if (hazardPresent) {
        // Hazard with existing op usage - prevent concurrent scheduling.
        opInfo.hazards |= userInfo.membership;
      } else {
        LLVM_DEBUG(llvm::dbgs() << "  $ hazard analysis says ok to schedule\n");
      }
      // Always inherit hazards whether merging or not.
      opInfo.hazards |= userInfo.hazards;
    }

    // Additional exhaustive testing for users of tied operands.
    // For each resource operand of this op we scan back through previously
    // created waves to see if there are any partitioned ops that have a hazard.
    for (auto operand : op.getOperands()) {
      if (!isa<IREE::Stream::ResourceType>(operand.getType()))
        continue;
      for (auto user : operand.getUsers()) {
        if (user == &op || user->getBlock() != block ||
            user->isBeforeInBlock(&op))
          continue;
        auto tiedOp = dyn_cast<IREE::Util::TiedOpInterface>(user);
        if (!tiedOp || !tiedOp.hasAnyTiedUses(operand))
          continue;
        auto userInfoIt = opInfos.find(user);
        if (userInfoIt == opInfos.end())
          continue;
        auto &userInfo = userInfoIt->second;
        LLVM_DEBUG({
          llvm::dbgs() << "Testing tied user:\n";
          user->print(llvm::dbgs(), *asmState);
          llvm::dbgs() << "\n";
          for (auto membershipOrdinal : userInfo.membership.set_bits()) {
            llvm::dbgs() << "  member of wave " << membershipOrdinal << "\n";
          }
          int lastHazardOrdinal = userInfo.hazards.find_last();
          if (lastHazardOrdinal != -1) {
            llvm::dbgs() << "  hazard w/ waves 0-" << lastHazardOrdinal << "\n";
          }
        });
        bool hazardPresent = hazardAnalysis.hasHazard(streamableOp, user);
        if (hazardPresent) {
          // Hazard with existing op usage - prevent concurrent scheduling.
          opInfo.hazards |= userInfo.membership;
        } else {
          LLVM_DEBUG(llvm::dbgs()
                     << "  $ hazard analysis says ok to schedule with tied\n");
        }
        // Always inherit hazards whether merging or not.
        opInfo.hazards |= userInfo.hazards;
      }
    }

    llvm::BitVector candidates(builders.size(), /*t=*/true);
    candidates ^= opInfo.hazards;

    // If this op is not streamable then bail here; we've still setup the hazard
    // map for following iteration.
    if (!streamableOp || streamableOp.isMetadata()) {
      LLVM_DEBUG(llvm::dbgs() << "Not streamable/is subview (skip)\n");
      continue;
    }

    opInfo.membership.reserve(builders.size() + 1);
    opInfo.membership.resize(builders.size(), /*t=*/false);

    // No consumers - if there's any candidate then we'll go into that.
    int firstCandidateOrdinal = favor == IREE::Stream::Favor::MaxConcurrency
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

  LLVM_DEBUG(waveSet.dump(*asmState));

  return waveSet;
}

} // namespace mlir::iree_compiler::IREE::Stream

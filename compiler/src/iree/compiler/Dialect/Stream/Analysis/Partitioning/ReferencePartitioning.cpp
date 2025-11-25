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

struct OpInfo {
  // Which partitions the op is contained within.
  llvm::BitVector membership;
  // Which partitions transitively depend on this operation.
  llvm::BitVector hazards;
  // Hazards specifically from nested region captures creating circular
  // dependencies. These must always be respected, even for operations with
  // preferCloneToConsumers.
  llvm::BitVector nestedRegionHazards;
};

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
    hazards |= opInfo.nestedRegionHazards;
  }
};

// Find the effective OpInfo for a user operation.
// Returns the OpInfo* to use for hazard tracking, or nullptr if the user
// should be skipped (either because hazards were already set or not found).
static OpInfo *getEffectiveUserInfo(Operation *user, Block *block,
                                    DenseMap<Operation *, OpInfo> &opInfos,
                                    OpInfo &opInfo) {
  // Check if user is in the same block we're partitioning.
  if (user->getBlock() == block) {
    // Direct lookup - user in same block.
    LLVM_DEBUG(llvm::dbgs() << "  User in same block\n");
    auto userInfoIt = opInfos.find(user);
    if (userInfoIt != opInfos.end()) {
      return &userInfoIt->second;
    }
    return nullptr;
  }

  LLVM_DEBUG(llvm::dbgs() << "  User in different block (nested region)\n");

  // User is in a nested region - find the containing operation in our block.
  // This handles cases like scf.for where operations inside the loop body
  // use values from the parent block. We treat the containing operation
  // (scf.for) as a proxy for all its nested users.
  Operation *containing = user->getParentOp();
  while (containing && containing->getBlock() != block) {
    containing = containing->getParentOp();
  }

  if (!containing) {
    return nullptr;
  }

  // Use the containing op's info as proxy for nested users.
  auto containingInfoIt = opInfos.find(containing);
  if (containingInfoIt != opInfos.end()) {
    // Only use containing op's info if it's streamable.
    // Non-streamable ops (scf.for, scf.if) need special circular dependency
    // tracking via nestedRegionHazards to prevent grouping captured values
    // with operations that consume the containing op's results.
    if (isa<IREE::Stream::StreamableOpInterface>(containing)) {
      return &containingInfoIt->second;
    }
    // Fall through to set nestedRegionHazards for non-streamable containing
    // ops.
  }

  // The containing operation is not streamable (e.g., scf.for).
  // Values captured into it create circular dependency hazards with any
  // partitions that use the containing operation's results.
  for (auto result : containing->getResults()) {
    for (auto resultUser : result.getUsers()) {
      if (resultUser->getBlock() == block) {
        auto resultUserInfoIt = opInfos.find(resultUser);
        if (resultUserInfoIt != opInfos.end()) {
          opInfo.nestedRegionHazards |= resultUserInfoIt->second.membership;
          opInfo.nestedRegionHazards |= resultUserInfoIt->second.hazards;
          LLVM_DEBUG({
            llvm::dbgs() << "Setting nestedRegionHazards for captured value, "
                         << "hazard count: "
                         << opInfo.nestedRegionHazards.count() << "\n";
          });
        }
      }
    }
  }
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

  DenseMap<Operation *, OpInfo> opInfos;

  SmallVector<std::unique_ptr<PartitionBuilder>> builders;
  llvm::BitVector usableBuilders;

  auto willCreateCircularDependencyBetweenPartitions =
      [&](unsigned sourceOrdinal, unsigned targetOrdinal) -> bool {
    // Returns:
    // If we are to make partition with ordinal targetOrdinal to
    // depend on partition with ordinal sourceOrdinal,
    // will this create a circular dependency.
    if (sourceOrdinal == targetOrdinal) {
      return false;
    }
    return builders[sourceOrdinal]->hazards.size() > targetOrdinal &&
           builders[sourceOrdinal]->hazards[targetOrdinal];
  };

  auto canAddOpToPartition = [&](Operation &op, OpInfo &opInfo,
                                 unsigned partitionOrdinal,
                                 bool check_for_clones = true) {
    auto streamableOp = dyn_cast<IREE::Stream::StreamableOpInterface>(op);
    if (!streamableOp) {
      return false;
    }

    // Most ops should have affinity at this point. If they do not then we allow
    // them to be placed anywhere (and whatever performance implications that
    // has is on the higher layers for not explicitly saying).
    IREE::Stream::AffinityAttr affinityAttr;
    if (auto affinityOp = dyn_cast<IREE::Stream::AffinityOpInterface>(op)) {
      affinityAttr = affinityOp.getAffinityAttr();
    }
    if (!IREE::Stream::AffinityAttr::canExecuteTogether(
            affinityAttr, builders[partitionOrdinal]->affinity)) {
      return false;
    }

    bool preferCloneToConsumers =
        check_for_clones && streamableOp.preferCloneToConsumers();
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
        if (builders[partitionOrdinal]->ops.contains(user)) {
          opHazardsInCandidatePartition |= opInfos[user].hazards;
        }
      }
      // For preferCloneToConsumers ops, we DON'T include nestedRegionHazards
      // because cloning into multiple partitions solves the circular
      // dependency. The op will be duplicated in each consumer partition, so
      // there's no shared state causing circular dependencies.
    } else {
      // For non-clonable ops, nestedRegionHazards must be respected to prevent
      // circular dependencies from nested region captures. Unlike normal
      // hazards which propagate through use-def chains, nestedRegionHazards may
      // not be fully captured in opInfo.hazards when the containing operation
      // (e.g., scf.for) is not streamable and returns nullptr from
      // getEffectiveUserInfo.
      opHazardsInCandidatePartition = opInfo.hazards;
      opHazardsInCandidatePartition |= opInfo.nestedRegionHazards;
      opHazards = &opHazardsInCandidatePartition;
    }

    // Check nestedRegionHazards separately first. Unlike normal hazards
    // (which indicate dependencies), nestedRegionHazards indicate partitions
    // the op CANNOT join due to circular dependency constraints.
    for (auto nestedHazardOrdinal : opInfo.nestedRegionHazards.set_bits()) {
      if (nestedHazardOrdinal == partitionOrdinal) {
        // This op cannot be in this partition due to nested region capture
        // circular dependency constraints.
        return false;
      }
    }

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
                                                        partitionOrdinal)) {
        return false;
      }
    }
    return true;
  };

  auto asmState = getRootAsmState(block);

  // Build a map from values to operations that consume them in nested regions.
  // This allows us to detect when an operation with nested regions (like
  // scf.for) consumes values produced by other operations that may create
  // hazards even though there's no direct use-def edge.
  DenseMap<Value, SmallVector<Operation *>> nestedConsumers;
  for (auto &op : *block) {
    // Only check operations with regions that are not streamable.
    // Streamable ops are handled normally but non-streamable ops with regions
    // (like scf.for, scf.if, scf.while) need special handling.
    if (op.getNumRegions() == 0 ||
        isa<IREE::Stream::StreamableOpInterface>(op)) {
      continue;
    }

    // Collect all values consumed by this operation (including in nested
    // regions).
    SetVector<Value> consumedValues;
    collectConsumedValues(&op, consumedValues);

    // For each consumed value record this op as a nested consumer.
    for (auto value : consumedValues) {
      // Only track values defined in the same block (not block arguments).
      auto definingOp = value.getDefiningOp();
      if (definingOp && definingOp->getBlock() == block) {
        nestedConsumers[value].push_back(&op);
      }
    }
  }

  llvm::DenseMap<Operation *, llvm::SmallVector<Operation *>> syncOps;
  for (auto &op : llvm::reverse(*block)) {

    LLVM_DEBUG({
      llvm::dbgs() << "====\nPartitioning op:\n";
      op.print(llvm::dbgs(), *asmState);
      llvm::dbgs() << "\n";
    });

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
          dyn_cast_if_present<IREE::Stream::StreamableOpInterface>(producer);
      if (streamable) {
        if (!syncOps.contains(producer)) {
          syncOps[producer] = llvm::SmallVector<Operation *>();
        }
        syncOps[producer].push_back(&op);
        LLVM_DEBUG({
          llvm::dbgs() << "Skipping sync op for now \n";
          op.print(llvm::dbgs(), *asmState);
          llvm::dbgs() << "\n";
        });
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

    // Set bits for each partition this op may be able to be placed into.
    // We prune the set based on whether the users are part of a transitive
    // dependency chain down the use-def chain to a partition.
    llvm::BitVector consumers(builders.size(), /*t=*/false);

    // Process direct users (common case).
    for (auto user : op.getUsers()) {
      OpInfo *effectiveUserInfo =
          getEffectiveUserInfo(user, block, opInfos, opInfo);
      if (!effectiveUserInfo) {
        continue;
      }

      auto &userInfo = *effectiveUserInfo;
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

    // Process indirect users (operations that consume this op's results in
    // their nested regions e.g. scf.for using a value in its body).
    for (auto result : op.getResults()) {
      auto nestedConsumersIt = nestedConsumers.find(result);
      if (nestedConsumersIt != nestedConsumers.end()) {
        for (auto *nestedUser : nestedConsumersIt->second) {
          auto userInfoIt = opInfos.find(nestedUser);
          if (userInfoIt == opInfos.end()) {
            continue;
          }
          auto &userInfo = userInfoIt->second;
          LLVM_DEBUG({
            llvm::dbgs() << "Testing nested region user:\n";
            nestedUser->print(llvm::dbgs(), *asmState);
            llvm::dbgs() << "\n";
            for (auto membershipOrdinal : userInfo.membership.set_bits()) {
              llvm::dbgs() << "  member of partition " << membershipOrdinal
                           << "\n";
            }
            for (auto hazardOrdinal : userInfo.hazards.set_bits()) {
              llvm::dbgs() << "  hazard w/ partition " << hazardOrdinal << "\n";
            }
          });
          // For nested consumers we only propagate hazards and not membership.
          // The nested consumer itself is not streamable and won't be in a
          // partition but it creates a dependency barrier.
          opInfo.hazards |= userInfo.membership;
          opInfo.hazards |= userInfo.hazards;
        }
      }
    }

    for (auto syncOp : syncOps[&op]) {
      for (auto user : syncOp->getUsers()) {
        OpInfo *effectiveUserInfo =
            getEffectiveUserInfo(user, block, opInfos, opInfo);
        if (!effectiveUserInfo) {
          continue;
        }

        auto &userInfo = *effectiveUserInfo;
        LLVM_DEBUG({
          llvm::dbgs() << "Testing sync user:\n";
          user->print(llvm::dbgs(), *asmState);
          llvm::dbgs() << "\n";
          for (auto membershipOrdinal : userInfo.membership.set_bits()) {
            llvm::dbgs() << "  member of partition " << membershipOrdinal
                         << "\n";
          }
          for (auto hazardOrdinal : userInfo.hazards.set_bits()) {
            llvm::dbgs() << "  hazard w/ partition " << hazardOrdinal << "\n";
          }
        });
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

    // If we prefer to clone to our consumers, but we are
    // only cloning to a subset, we have to re-check our
    // partitions as they may generate cycles.
    if (streamableOp.preferCloneToConsumers()) {
      auto tempCandidates = candidates;
      tempCandidates &= consumers;
      if (tempCandidates.count() != consumers.count()) {
        // Prune candidates that do not have a compatible affinity.
        for (auto ordinal : candidates.set_bits()) {
          if (!canAddOpToPartition(op, opInfo, ordinal, false)) {
            LLVM_DEBUG(llvm::dbgs() << "Candidate partition " << ordinal
                                    << " incompatible for clone\n");
            candidates.reset(ordinal);
          }
        }

        for (auto syncOp : syncOps[&op]) {
          for (auto ordinal : candidates.set_bits()) {
            if (!canAddOpToPartition(*syncOp, opInfo, ordinal, false)) {
              LLVM_DEBUG(llvm::dbgs() << "Candidate partition " << ordinal
                                      << " incompatible for clone\n");
              candidates.reset(ordinal);
            }
          }
        }
      }
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
      LLVM_DEBUG(llvm::dbgs() << "Moving sync to candidate partition "
                              << firstCandidateOrdinal << ":\n    ");
      LLVM_DEBUG(syncOp->print(llvm::dbgs(), *asmState));
      LLVM_DEBUG(llvm::dbgs() << "\n");
      builder->insert(syncOp, opInfo);
    }

    LLVM_DEBUG(llvm::dbgs() << "Moving to first candidate partition "
                            << firstCandidateOrdinal << " (continue)\n");
    // If we are a clonable op (like splat) clone us into every partition.
    // We also clone if there are nested region hazards, indicating consumers
    // in nested regions that will need their own copy.
    // Otherwise we just pick the first we find (probably a bad heuristic).
    bool shouldClone =
        streamableOp.preferCloneToConsumers() &&
        (consumers.count() > 1 || opInfo.nestedRegionHazards.any());
    if (shouldClone && consumers.any()) {
      for (auto consumerOrdinal : consumers.set_bits()) {
        LLVM_DEBUG(llvm::dbgs() << "Cloning into consumer partition "
                                << consumerOrdinal << "\n");
        auto &consumerBuilder = builders[consumerOrdinal];
        consumerBuilder->insert(&op, opInfo);
        consumerBuilder->clonedOps.insert(&op);
      }
    } else if (shouldClone && !consumers.any()) {
      // We want to clone but have no streamable consumers. Check if the op
      // has any uses in the same block - if so, it needs to be in a partition.
      // If all uses are in nested regions, leave it for re-materialization.
      bool hasUsesInSameBlock = false;
      for (auto result : op.getResults()) {
        for (auto user : result.getUsers()) {
          if (user->getBlock() == block) {
            hasUsesInSameBlock = true;
            break;
          }
        }
        if (hasUsesInSameBlock) {
          break;
        }
      }
      if (hasUsesInSameBlock) {
        // Has uses in same block, insert into first candidate partition.
        builder->insert(&op, opInfo);
      }
      // Else: all uses are in nested regions, leave for re-materialization.
    } else {
      // Not a clonable op, insert into first candidate partition.
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
      // Collect direct operands only. Do NOT recursively collect values from
      // nested regions of operations that define our operands - that would
      // incorrectly add values consumed inside control flow ops (scf.for, etc)
      // to partition inputs, creating spurious circular dependencies.
      // The circular dependency check in Partition::verify() will correctly
      // detect actual circular dependencies using collectConsumedValues.
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
      if (userInfoIt == opInfos.end()) {
        continue;
      }
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
      if (!isa<IREE::Stream::ResourceType>(operand.getType())) {
        continue;
      }
      for (auto user : operand.getUsers()) {
        if (user == &op || user->getBlock() != block ||
            user->isBeforeInBlock(&op)) {
          continue;
        }
        auto tiedOp = dyn_cast<IREE::Util::TiedOpInterface>(user);
        if (!tiedOp || !tiedOp.hasAnyTiedUses(operand)) {
          continue;
        }
        auto userInfoIt = opInfos.find(user);
        if (userInfoIt == opInfos.end()) {
          continue;
        }
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

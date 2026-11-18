// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VM/Analysis/RegisterAllocation.h"

#include <algorithm>
#include <utility>

#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/Dialect/VM/Analysis/LinearScan/LiveIntervals.h"
#include "iree/compiler/Dialect/VM/Analysis/LinearScan/RegisterBank.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"

namespace mlir::iree_compiler {

static Attribute getStrArrayAttr(Builder &builder,
                                 ArrayRef<std::string> values) {
  return builder.getStrArrayAttr(llvm::map_to_vector<8>(
      values, [](const std::string &value) { return StringRef(value); }));
}

// static
LogicalResult RegisterAllocation::annotateIR(IREE::VM::FuncOp funcOp) {
  RegisterAllocation registerAllocation;
  if (failed(registerAllocation.recalculate(funcOp))) {
    funcOp.emitOpError() << "failed to allocate registers for function";
    return failure();
  }

  Builder builder(funcOp.getContext());
  for (auto &block : funcOp.getBlocks()) {
    SmallVector<std::string, 8> blockRegStrs;
    blockRegStrs.reserve(block.getNumArguments());
    for (auto blockArg : block.getArguments()) {
      auto reg = registerAllocation.map_[blockArg];
      blockRegStrs.push_back(reg.toString());
    }
    block.front().setAttr("block_registers",
                          getStrArrayAttr(builder, blockRegStrs));

    for (auto &op : block.getOperations()) {
      // Emit operand registers with MOVE bits for ref operands.
      if (op.getNumOperands() > 0) {
        SmallVector<std::string, 8> operandRegStrs;
        operandRegStrs.reserve(op.getNumOperands());

        // Use the same per-successor MOVE bits as branch register remapping,
        // including operands coalesced with their destination registers.
        llvm::DenseMap<unsigned, Register> branchOperandRegs;
        if (auto branchOp = dyn_cast<BranchOpInterface>(&op)) {
          for (unsigned i = 0; i < op.getNumSuccessors(); ++i) {
            auto succOperands =
                branchOp.getSuccessorOperands(i).getForwardedOperands();
            if (succOperands.empty()) {
              continue;
            }
            unsigned baseIdx = succOperands.getBeginOperandIndex();
            auto operandRegs =
                registerAllocation.getSuccessorOperandRegisters(&op, i);
            for (auto [localIdx, reg] : llvm::enumerate(operandRegs)) {
              branchOperandRegs[baseIdx + localIdx] = reg;
            }
          }
        }

        for (auto &operand : op.getOpOperands()) {
          unsigned idx = operand.getOperandNumber();
          Register reg;
          if (branchOperandRegs.count(idx)) {
            // Use the pre-computed register with correct MOVE bit.
            reg = branchOperandRegs[idx];
          } else {
            // Use standard mapUseToRegister for non-branch operands.
            reg = registerAllocation.mapUseToRegister(operand.get(), &op, idx);
          }
          operandRegStrs.push_back(reg.toString());
        }
        op.setAttr("operand_registers",
                   getStrArrayAttr(builder, operandRegStrs));
      }
      if (op.getNumResults() == 0) {
        continue;
      }
      SmallVector<std::string, 8> regStrs;
      regStrs.reserve(op.getNumResults());
      for (auto result : op.getResults()) {
        auto reg = registerAllocation.map_[result];
        regStrs.push_back(reg.toString());
      }
      op.setAttr("result_registers", getStrArrayAttr(builder, regStrs));
    }

    Operation *terminatorOp = block.getTerminator();
    if (terminatorOp->getNumSuccessors() > 0) {
      SmallVector<Attribute, 2> successorAttrs;
      successorAttrs.reserve(terminatorOp->getNumSuccessors());
      for (int i = 0; i < terminatorOp->getNumSuccessors(); ++i) {
        auto srcDstRegs =
            registerAllocation.remapSuccessorRegisters(terminatorOp, i);
        SmallVector<std::string, 8> remappingStrs;
        for (auto &srcDstReg : srcDstRegs) {
          remappingStrs.push_back(llvm::formatv("{}->{}",
                                                srcDstReg.first.toString(),
                                                srcDstReg.second.toString())
                                      .str());
        }
        successorAttrs.push_back(getStrArrayAttr(builder, remappingStrs));
      }
      terminatorOp->setAttr("remap_registers",
                            builder.getArrayAttr(successorAttrs));
    }
  }

  return success();
}

// Sorts blocks in dominance order such that the entry block is first and
// all of the following blocks are dominated only by blocks that have come
// before them in the list. This ensures that we always know all registers for
// block live-in values as we walk the blocks.
static SmallVector<Block *, 8>
sortBlocksInDominanceOrder(IREE::VM::FuncOp funcOp) {
  if (funcOp.getBlocks().size() == 1) {
    // Dominance info cannot be computed for regions with one block.
    return {&funcOp.getBlocks().front()};
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
  auto orderedBlocks = markedBlocks.takeVector();
  std::reverse(orderedBlocks.begin(), orderedBlocks.end());
  return orderedBlocks;
}

// Runs linear scan allocation for a single register bank.
// Processes intervals matching isRefBank, allocating from bank into active
// list. Updates map_ and intervalAllocations_ for each allocated interval.
LogicalResult RegisterAllocation::runLinearScan(
    IREE::VM::FuncOp funcOp, const LiveIntervals &liveIntervals,
    IREE::VM::RegisterBank &bank, SmallVectorImpl<const LiveInterval *> &active,
    const llvm::DenseMap<Value, Value> &coalesceSource, bool isRefBank,
    int &maxOrdinal) {

  // Helper to expire old intervals.
  // Use < (not <=) so intervals ending at currentStart are NOT expired yet.
  // This prevents the bug where multiple block args with the same start
  // position get assigned the same register.
  auto expireOldIntervals = [&](uint32_t currentStart) {
    while (!active.empty() && active.front()->end < currentStart) {
      const LiveInterval *expired = active.front();
      size_t byteWidth = isRefBank ? 0 : expired->byteWidth;
      bank.release(intervalAllocations_[expired].assigned, byteWidth);
      active.erase(active.begin());
    }
  };

  // Helper to insert into active list (maintain sorted by end).
  auto insertActive = [&](const LiveInterval *interval) {
    auto it =
        std::lower_bound(active.begin(), active.end(), interval,
                         [](const LiveInterval *a, const LiveInterval *b) {
                           return a->end < b->end;
                         });
    active.insert(it, interval);
  };

  for (const LiveInterval *interval : liveIntervals.getSortedByStart()) {
    // Filter by bank type.
    if (interval->isRef != isRefBank) {
      continue;
    }

    // Skip already-assigned values (entry block args).
    if (map_.count(interval->value)) {
      continue;
    }

    // Expire intervals that ended before this one starts.
    expireOldIntervals(interval->start);

    size_t byteWidth = isRefBank ? 0 : interval->byteWidth;

    // Try coalescing: reuse register from a value whose interval ends exactly
    // where ours begins.
    int ordinal = -1;
    auto coalesceIt = coalesceSource.find(interval->value);
    if (coalesceIt != coalesceSource.end()) {
      auto srcIt = map_.find(coalesceIt->second);
      if (srcIt != map_.end()) {
        int preferredOrdinal = srcIt->second.ordinal();
        if (bank.allocateAt(preferredOrdinal, byteWidth)) {
          ordinal = preferredOrdinal;
        } else {
          // The source register may still be "active" (not expired due to <
          // check). Find and expire it to complete the hand-off.
          for (auto it = active.begin(); it != active.end(); ++it) {
            if (intervalAllocations_[*it].assigned == preferredOrdinal &&
                (*it)->end == interval->start) {
              size_t expiredByteWidth = isRefBank ? 0 : (*it)->byteWidth;
              bank.release(intervalAllocations_[*it].assigned,
                           expiredByteWidth);
              active.erase(it);
              if (bank.allocateAt(preferredOrdinal, byteWidth)) {
                ordinal = preferredOrdinal;
              }
              break;
            }
          }
        }
      }
    }

    // Fall back to first-fit allocation.
    if (ordinal < 0) {
      auto allocated = bank.allocate(byteWidth);
      if (!allocated) {
        return funcOp.emitError()
               << "register allocation failed (" << (isRefBank ? "ref" : "int")
               << " bank overflow)";
      }
      ordinal = *allocated;
    }

    // Store assignment in map.
    intervalAllocations_[interval].assigned = ordinal;
    if (isRefBank) {
      map_[interval->value] = Register::getRef(interval->value.getType(),
                                               ordinal, /*isMove=*/false);
    } else {
      map_[interval->value] =
          Register::getValue(interval->value.getType(), ordinal);
    }

    // Add to active set.
    insertActive(interval);
  }

  maxOrdinal = bank.getMaxUsed();
  return success();
}

// Linear scan allocator implementation.
// Uses LiveIntervals for global register lifetime tracking and enables
// cross-block register reuse for both integer and reference registers.
LogicalResult RegisterAllocation::recalculate(IREE::VM::FuncOp funcOp) {
  map_.clear();
  intervalAllocations_.clear();
  dominanceInfo_.reset();
  maxI32RegisterOrdinal_ = -1;
  maxRefRegisterOrdinal_ = -1;
  scratchI32RegisterCount_ = 0;
  scratchRefRegisterCount_ = 0;

  // Build live intervals for global allocation.
  LiveIntervals liveIntervals;
  if (failed(liveIntervals.build(funcOp))) {
    return funcOp.emitError() << "failed to build live intervals";
  }

  // Also compute liveness for MOVE bit computation (still needed).
  if (failed(liveness_.recalculate(funcOp))) {
    return funcOp.emitError()
           << "failed to calculate required liveness information";
  }

  // Compute dominance info for back-edge detection in MOVE bit logic.
  // Only needed for functions with more than one block.
  if (funcOp.getBlocks().size() > 1) {
    dominanceInfo_.emplace(funcOp);
  }

  // Global register banks (single instance for entire function).
  IREE::VM::RegisterBank intBank(Register::kInt32RegisterCount);
  IREE::VM::RegisterBank refBank(Register::kRefRegisterCount);

  // ===== PHASE A: Pre-allocate entry block args (monotonic, ABI) =====
  // Also set the assigned field in the interval so it can be properly expired.
  Block *entryBlock = &funcOp.getBlocks().front();
  for (auto blockArg : entryBlock->getArguments()) {
    Type type = blockArg.getType();
    if (isa<IREE::VM::RefType>(type)) {
      // Entry block ref args: monotonic allocation for ABI stability.
      auto ordinal = refBank.allocate(0, /*fromEnd=*/true);
      if (!ordinal) {
        return funcOp.emitError()
               << "register allocation failed for entry block arg "
               << blockArg.getArgNumber() << " (ref bank overflow)";
      }
      map_[blockArg] = Register::getRef(type, *ordinal, /*isMove=*/false);
      // Set assigned in interval for expiration tracking.
      if (const LiveInterval *interval = liveIntervals.getInterval(blockArg)) {
        intervalAllocations_[interval].assigned = *ordinal;
      }
    } else if (type.isIntOrFloat()) {
      // Entry block int args: monotonic allocation for ABI stability.
      size_t byteWidth = IREE::Util::getRoundedElementByteWidth(type);
      auto ordinal = intBank.allocate(byteWidth, /*fromEnd=*/true);
      if (!ordinal) {
        return funcOp.emitError()
               << "register allocation failed for entry block arg "
               << blockArg.getArgNumber() << " (int bank overflow)";
      }
      map_[blockArg] = Register::getValue(type, *ordinal);
      // Set assigned in interval for expiration tracking.
      if (const LiveInterval *interval = liveIntervals.getInterval(blockArg)) {
        intervalAllocations_[interval].assigned = *ordinal;
      }
    }
  }

  // ===== PHASE B: Compute coalescing candidates =====
  // Find values that could potentially share registers via hand-off.
  // A value V can inherit register from value U if:
  //   1. U's interval ends exactly where V's begins (hand-off point)
  //   2. U and V have compatible types (same register bank)
  //
  // We store U (the source) for each V (the dest), and resolve to register
  // ordinals during allocation when U has been assigned.
  llvm::DenseMap<Value, Value> coalesceSource;

  auto recordCoalesceCandidate = [&](Value dest, Value src) {
    if (dest.getType() != src.getType()) {
      return;
    }
    auto srcInterval = liveIntervals.getInterval(src);
    auto destInterval = liveIntervals.getInterval(dest);
    if (!srcInterval || !destInterval) {
      return;
    }
    // Only coalesce if intervals meet exactly (hand-off).
    if (srcInterval->end != destInterval->start) {
      return;
    }
    coalesceSource[dest] = src;
  };

  // Scan all operations for coalescing opportunities.
  for (auto *block : liveIntervals.getBlockOrder()) {
    // Block arguments can coalesce with branch operands from predecessors.
    for (auto *pred : block->getPredecessors()) {
      auto branchOp = dyn_cast<BranchOpInterface>(pred->getTerminator());
      if (!branchOp) {
        continue;
      }
      for (unsigned succIdx = 0;
           succIdx < pred->getTerminator()->getNumSuccessors(); ++succIdx) {
        if (pred->getTerminator()->getSuccessor(succIdx) != block) {
          continue;
        }
        OperandRange operands =
            branchOp.getSuccessorOperands(succIdx).getForwardedOperands();
        for (auto [idx, operand] : llvm::enumerate(operands)) {
          if (idx >= block->getNumArguments()) {
            break;
          }
          recordCoalesceCandidate(block->getArgument(idx), operand);
        }
      }
    }

    // Operation results can coalesce with operands whose intervals end here.
    for (auto &op : block->getOperations()) {
      for (Value result : op.getResults()) {
        for (Value operand : op.getOperands()) {
          recordCoalesceCandidate(result, operand);
        }
      }
    }
  }

  // ===== PHASE C+D: Linear scan for both register banks =====
  // Active intervals sorted by end position (one list per bank).
  SmallVector<const LiveInterval *> activeInt;
  SmallVector<const LiveInterval *> activeRef;

  // Helper to seed active list with pre-allocated entry block arg intervals.
  // This allows their registers to be expired and reused.
  auto seedActive = [&](SmallVectorImpl<const LiveInterval *> &active,
                        auto typeCheck) {
    for (auto blockArg : entryBlock->getArguments()) {
      if (typeCheck(blockArg.getType())) {
        if (const LiveInterval *interval =
                liveIntervals.getInterval(blockArg)) {
          auto it = std::lower_bound(
              active.begin(), active.end(), interval,
              [](const LiveInterval *a, const LiveInterval *b) {
                return a->end < b->end;
              });
          active.insert(it, interval);
        }
      }
    }
  };
  seedActive(activeInt, [](Type t) { return t.isIntOrFloat(); });
  seedActive(activeRef, [](Type t) { return isa<IREE::VM::RefType>(t); });

  // Run linear scan for each bank.
  if (failed(runLinearScan(funcOp, liveIntervals, intBank, activeInt,
                           coalesceSource, /*isRefBank=*/false,
                           maxI32RegisterOrdinal_))) {
    return failure();
  }
  if (failed(runLinearScan(funcOp, liveIntervals, refBank, activeRef,
                           coalesceSource, /*isRefBank=*/true,
                           maxRefRegisterOrdinal_))) {
    return failure();
  }

  // Verify bounds.
  if (maxI32RegisterOrdinal_ > Register::kInt32RegisterCount ||
      maxRefRegisterOrdinal_ > Register::kRefRegisterCount) {
    return funcOp.emitError() << "function overflows stack register banks; "
                                 "spilling to memory not yet implemented";
  }

  // Identify discards that can be elided because their operands were released
  // via MOVE on preceding operations.
  computeElidableDiscards(funcOp);

  return success();
}

void RegisterAllocation::computeElidableDiscards(IREE::VM::FuncOp funcOp) {
  discardOperandElidability_.clear();

  // For each discard operand, check if it was released via MOVE on a preceding
  // operation. An operand is elidable when:
  // 1. The value has a "real" use (non-discard) before this discard in the
  //    same block, AND
  // 2. That real use actually got MOVE set (isLastRealValueUse is true and
  //    the op implements RefMoveInterface)
  //
  // Values that only have the discard as their use in a block (e.g., coming
  // from block arguments or definitions in predecessors with no real use
  // before the discard) cannot be elided - the discard is the only thing
  // releasing them.
  for (auto &block : funcOp.getBlocks()) {
    for (auto &op : block.getOperations()) {
      auto discardOp = dyn_cast<IREE::VM::DiscardRefsOp>(&op);
      if (!discardOp) {
        continue;
      }

      SmallVector<bool> operandElidability;
      for (Value ref : discardOp.getRefs()) {
        // Check if there's a preceding real use in this block that got MOVE.
        bool hasPrecedingMoveUse = false;
        for (auto it = Block::iterator(discardOp); it != block.begin();) {
          --it;
          Operation *precedingOp = &*it;
          for (OpOperand &operand : precedingOp->getOpOperands()) {
            if (operand.get() == ref &&
                !isa<IREE::VM::DiscardRefsOp>(precedingOp)) {
              // Found a real use before the discard.
              // Check if MOVE was actually set on this use.
              if (liveness_.isLastRealValueUse(ref, precedingOp,
                                               operand.getOperandNumber())) {
                // Also check if the op implements RefMoveInterface and says
                // this operand is movable.
                if (auto moveOp =
                        dyn_cast<IREE::VM::RefMoveInterface>(precedingOp)) {
                  if (moveOp.isRefOperandMovable(operand.getOperandNumber())) {
                    hasPrecedingMoveUse = true;
                  }
                }
              }
              break;
            }
          }
          if (hasPrecedingMoveUse) {
            break;
          }
        }
        operandElidability.push_back(hasPrecedingMoveUse);
      }
      discardOperandElidability_[discardOp] = std::move(operandElidability);
    }
  }
}

Register RegisterAllocation::mapToRegister(Value value) const {
  auto it = map_.find(value);
  assert(it != map_.end());
  return it->getSecond();
}

Register RegisterAllocation::mapUseToRegister(Value value, Operation *useOp,
                                              int operandIndex) {
  auto reg = mapToRegister(value);
  if (reg.isRef() && liveness_.isLastRealValueUse(value, useOp, operandIndex)) {
    // Set MOVE bit when this is the last real use of the value. Only set MOVE
    // if the op implements RefMoveInterface and says this operand is movable.
    // This prevents setting MOVE on ops that don't honor it in the runtime.
    if (auto moveOp = dyn_cast<IREE::VM::RefMoveInterface>(useOp)) {
      if (moveOp.isRefOperandMovable(operandIndex)) {
        reg.setMove(true);
      }
    }
  }
  return reg;
}

// A feedback arc set containing the minimal list of cycle-causing edges.
// https://en.wikipedia.org/wiki/Feedback_arc_set
struct FeedbackArcSet {
  using NodeID = Register;
  using Edge = std::pair<NodeID, NodeID>;

  // Edges making up a DAG (inputEdges - feedbackEdges).
  SmallVector<Edge, 8> acyclicEdges;

  // Edges of the feedback arc set that, if added to acyclicEdges, would cause
  // cycles.
  SmallVector<Edge, 8> feedbackEdges;

  // Computes the FAS of a given directed graph that may contain cycles.
  static FeedbackArcSet compute(ArrayRef<Edge> inputEdges) {
    FeedbackArcSet result;
    if (inputEdges.empty()) {
      return result;
    } else if (inputEdges.size() == 1) {
      result.acyclicEdges.push_back(inputEdges.front());
      return result;
    }

    struct FASNode {
      NodeID id;
      int indegree = 0;
      int outdegree = 0;
    };
    // This should not be modified after creation in this loop. We take pointers
    // to its entries so do not want to invalidate them with reallocation.
    llvm::SmallDenseMap<NodeID, FASNode> nodes;
    for (auto &edge : inputEdges) {
      NodeID sourceID = edge.first.asBaseRegister();
      NodeID sinkID = edge.second.asBaseRegister();
      assert(sourceID != sinkID && "self-cycles not supported");
      if (nodes.count(sourceID) == 0) {
        nodes.insert({sourceID, {sourceID, 0, 0}});
      }
      if (nodes.count(sinkID) == 0) {
        nodes.insert({sinkID, {sinkID, 0, 0}});
      }
    }

    struct FASEdge {
      FASNode *source;
      FASNode *sink;
    };
    int maxOutdegree = 0;
    int maxIndegree = 0;
    SmallVector<FASEdge, 8> edges;
    for (auto &edge : inputEdges) {
      NodeID sourceID = edge.first.asBaseRegister();
      NodeID sinkID = edge.second.asBaseRegister();
      auto &sourceNode = nodes[sourceID];
      ++sourceNode.outdegree;
      maxOutdegree = std::max(maxOutdegree, sourceNode.outdegree);
      auto &sinkNode = nodes[sinkID];
      ++sinkNode.indegree;
      maxIndegree = std::max(maxIndegree, sinkNode.indegree);
      edges.push_back({&sourceNode, &sinkNode});
    }

    std::vector<SmallVector<FASNode *, 2>> buckets;
    buckets.resize(std::max(maxOutdegree, maxIndegree) + 2);
    auto nodeToBucketIndex = [&](FASNode *node) {
      return node->indegree == 0 || node->outdegree == 0
                 ? buckets.size() - 1
                 : std::abs(node->outdegree - node->indegree);
    };
    auto assignBucket = [&](FASNode *node) {
      buckets[nodeToBucketIndex(node)].push_back(node);
    };
    auto removeBucket = [&](FASNode *node) {
      int index = nodeToBucketIndex(node);
      auto it = std::find(buckets[index].begin(), buckets[index].end(), node);
      if (it != buckets[index].end()) {
        buckets[index].erase(it);
      }
    };
    llvm::SmallPtrSet<FASNode *, 8> remainingNodes;
    for (auto &nodeEntry : nodes) {
      assignBucket(&nodeEntry.getSecond());
      remainingNodes.insert(&nodeEntry.getSecond());
    }

    auto removeNode = [&](FASNode *node) {
      SmallVector<FASEdge> inEdges;
      inEdges.reserve(node->indegree);
      SmallVector<FASEdge> outEdges;
      outEdges.reserve(node->outdegree);
      for (auto &edge : edges) {
        if (edge.sink == node) {
          inEdges.push_back(edge);
        }
        if (edge.source == node) {
          outEdges.push_back(edge);
        }
      }
      bool collectInEdges = node->indegree <= node->outdegree;
      bool collectOutEdges = !collectInEdges;

      SmallVector<Edge> results;
      for (auto &edge : inEdges) {
        if (edge.source == node) {
          continue;
        }
        if (collectInEdges) {
          results.push_back({edge.source->id, edge.sink->id});
        }
        removeBucket(edge.source);
        --edge.source->outdegree;
        assert(edge.source->outdegree >= 0 && "outdegree has become negative");
        assignBucket(edge.source);
      }
      for (auto &edge : outEdges) {
        if (edge.sink == node) {
          continue;
        }
        if (collectOutEdges) {
          results.push_back({edge.source->id, edge.sink->id});
        }
        removeBucket(edge.sink);
        --edge.sink->indegree;
        assert(edge.sink->indegree >= 0 && "indegree has become negative");
        assignBucket(edge.sink);
      }

      remainingNodes.erase(node);
      edges.erase(std::remove_if(edges.begin(), edges.end(),
                                 [&](const FASEdge &edge) {
                                   return edge.source == node ||
                                          edge.sink == node;
                                 }),
                  edges.end());
      return results;
    };
    auto ends = buckets.back();
    while (!remainingNodes.empty()) {
      while (!ends.empty()) {
        auto *node = ends.front();
        ends.erase(ends.begin());
        removeNode(node);
      }
      if (remainingNodes.empty()) {
        break;
      }
      for (ssize_t i = buckets.size() - 1; i >= 0; --i) {
        if (buckets[i].empty()) {
          continue;
        }
        auto *bucket = buckets[i].front();
        buckets[i].erase(buckets[i].begin());
        auto feedbackEdges = removeNode(bucket);
        result.feedbackEdges.append(feedbackEdges.begin(), feedbackEdges.end());
        break;
      }
    }

    // Build the DAG of the remaining edges now that we've isolated the ones
    // that cause cycles.
    llvm::SmallSetVector<NodeID, 8> acyclicNodes;
    SmallVector<Edge, 8> acyclicEdges;
    for (auto &inputEdge : inputEdges) {
      auto it = std::find_if(result.feedbackEdges.begin(),
                             result.feedbackEdges.end(), [&](const Edge &edge) {
                               return edge.first == inputEdge.first &&
                                      edge.second == inputEdge.second;
                             });
      if (it == result.feedbackEdges.end()) {
        acyclicEdges.push_back(inputEdge);
        acyclicNodes.insert(inputEdge.first.asBaseRegister());
        acyclicNodes.insert(inputEdge.second.asBaseRegister());
      }
    }

    // Topologically sort the DAG so that we don't overwrite anything.
    llvm::SmallSetVector<NodeID, 8> unmarkedNodes = acyclicNodes;
    llvm::SmallSetVector<NodeID, 8> markedNodes;
    std::function<void(NodeID)> visit = [&](NodeID node) {
      if (markedNodes.contains(node)) {
        return;
      }
      for (auto &edge : acyclicEdges) {
        if (edge.first != node) {
          continue;
        }
        visit(edge.second);
      }
      markedNodes.insert(node);
    };
    while (!unmarkedNodes.empty()) {
      visit(unmarkedNodes.pop_back_val());
    }
    for (auto node : markedNodes.takeVector()) {
      for (auto &edge : acyclicEdges) {
        if (edge.first != node) {
          continue;
        }
        result.acyclicEdges.push_back({edge.first, edge.second});
      }
    }

    return result;
  }
};

// Computes registers and MOVE bits for all forwarded successor operands,
// including operands coalesced with their destination registers.
//
// For a remap between distinct registers, MOVE transfers ownership and clears
// the source register. Without MOVE, the runtime retains a reference for the
// destination.
//
// A ref operand receives MOVE only when all four conditions hold:
// 1. It is the last occurrence of this value in the successor's operand list.
// 2. It is the last real use across all successors, or the remap is
//    non-coalesced and the original value is not live-in to this target block.
// 3. The branch op supports MOVE on this operand via RefMoveInterface.
// 4. No duplicate occurrence is coalesced with its destination block argument;
//    moving another occurrence would clear that coalesced register.
//
// These conditions also apply to back-edges. When they hold for a remap to a
// loop header, MOVE transfers ownership to the next iteration's block argument.
// The edge being a back-edge does not itself require retaining the source ref.
SmallVector<Register, 8>
RegisterAllocation::getSuccessorOperandRegisters(Operation *branchOp,
                                                 unsigned successorIndex) {
  auto targetOperands = cast<BranchOpInterface>(branchOp)
                            .getSuccessorOperands(successorIndex)
                            .getForwardedOperands();
  if (targetOperands.empty()) {
    return {};
  }
  Block *targetBlock = branchOp->getSuccessor(successorIndex);
  unsigned baseOperandIndex = targetOperands.getBeginOperandIndex();

  struct RefOperandInfo {
    unsigned lastIndex = 0;
    unsigned count = 0;
    bool coalesced = false;
  };
  llvm::DenseMap<Value, RefOperandInfo> refOperands;
  SmallVector<Register, 8> operandRegisters;
  operandRegisters.reserve(targetOperands.size());
  for (auto [idx, operand] : llvm::enumerate(targetOperands)) {
    auto srcReg = mapToRegister(operand);
    operandRegisters.push_back(srcReg);
    if (!srcReg.isRef()) {
      continue;
    }
    auto &info = refOperands[operand];
    info.lastIndex = idx;
    ++info.count;
    info.coalesced |= srcReg == mapToRegister(targetBlock->getArgument(idx));
  }

  auto moveOp = dyn_cast<IREE::VM::RefMoveInterface>(branchOp);
  for (auto [idx, operand] : llvm::enumerate(targetOperands)) {
    auto &srcReg = operandRegisters[idx];
    if (!srcReg.isRef()) {
      continue;
    }
    auto info = refOperands.lookup(operand);
    bool isLastOccurrence = info.lastIndex == idx;
    unsigned globalOperandIndex = baseOperandIndex + idx;
    bool isLastUseAcrossSuccessors =
        liveness_.isLastRealValueUse(operand, branchOp, globalOperandIndex);

    // A ref may remain live on one successor while ownership transfers on
    // another. A non-coalesced remap can MOVE when this target does not use
    // the original value directly, even if another successor does.
    auto dstReg = mapToRegister(targetBlock->getArgument(idx));
    bool transfersOnTargetEdge =
        srcReg != dstReg &&
        !llvm::is_contained(liveness_.getBlockLiveIns(targetBlock), operand);
    bool canTransferOwnership =
        isLastUseAcrossSuccessors || transfersOnTargetEdge;
    bool isMovable = moveOp && moveOp.isRefOperandMovable(globalOperandIndex);

    // Moving another occurrence would clear the coalesced destination register.
    // A single coalesced operand emits no remap pair, so its MOVE bit only
    // affects IR annotation.
    bool hasCoalescedDuplicate = info.count > 1 && info.coalesced;
    if (isLastOccurrence && canTransferOwnership && isMovable &&
        !hasCoalescedDuplicate) {
      srcReg.setMove(true);
    }
  }
  return operandRegisters;
}

SmallVector<std::pair<Register, Register>, 8>
RegisterAllocation::remapSuccessorRegisters(Operation *op, int successorIndex) {
  auto operandRegisters = getSuccessorOperandRegisters(op, successorIndex);
  return remapSuccessorRegisters(op, op->getSuccessor(successorIndex),
                                 operandRegisters);
}

SmallVector<std::pair<Register, Register>, 8>
RegisterAllocation::remapSuccessorRegisters(Location loc, Block *targetBlock,
                                            OperandRange targetOperands) {
  // Without branch op context, leave MOVE bits unset.
  auto operandRegisters = llvm::map_to_vector<8>(
      targetOperands, [&](Value operand) { return mapToRegister(operand); });
  return remapSuccessorRegisters(/*branchOp=*/nullptr, targetBlock,
                                 operandRegisters);
}

SmallVector<std::pair<Register, Register>, 8>
RegisterAllocation::remapSuccessorRegisters(
    Operation *branchOp, Block *targetBlock,
    ArrayRef<Register> operandRegisters) {
  // Compute the initial directed graph of register movements.
  // This may contain cycles ([reg 0->1], [reg 1->0], ...) that would not be
  // possible to evaluate as a direct remapping. Coalesced operands need no
  // remap.
  SmallVector<std::pair<Register, Register>, 8> srcDstRegs;
  for (auto [idx, srcReg] : llvm::enumerate(operandRegisters)) {
    auto dstReg = mapToRegister(targetBlock->getArgument(idx));
    if (srcReg != dstReg) {
      srcDstRegs.push_back({srcReg, dstReg});
    }
  }

  // Compute the feedback arc set to determine which edges are the ones inducing
  // cycles, if any. This also provides us a DAG that we can trivially remap
  // without worrying about cycles.
  auto feedbackArcSet = FeedbackArcSet::compute(srcDstRegs);
  assert(feedbackArcSet.acyclicEdges.size() +
                 feedbackArcSet.feedbackEdges.size() ==
             srcDstRegs.size() &&
         "lost an edge during feedback arc set computation");

  // If there's no cycles we can simply use the sorted DAG produced.
  if (feedbackArcSet.feedbackEdges.empty()) {
    return feedbackArcSet.acyclicEdges;
  }

  // The tail registers in each bank is reserved for swapping, when required.
  int localScratchI32RegCount = 0;
  int localScratchRefRegCount = 0;
  for (auto feedbackEdge : feedbackArcSet.feedbackEdges) {
    Register scratchReg;
    if (feedbackEdge.first.isRef()) {
      localScratchRefRegCount += 1;
      scratchReg = Register::getWithSameType(
          feedbackEdge.first, maxRefRegisterOrdinal_ + localScratchRefRegCount);
    } else {
      localScratchI32RegCount += 1;
      scratchReg = Register::getWithSameType(
          feedbackEdge.first, maxI32RegisterOrdinal_ + localScratchI32RegCount);
      // Integer types that use more than one register slot will be emitted
      // as remapping per 4-byte word, so we have to account for the extra
      // temporaries. See BytecodeEncoder:encodeBranch().
      assert(scratchReg.byteWidth() >= 4 && "expected >= i32");
      localScratchI32RegCount += scratchReg.byteWidth() / 4 - 1;
    }
    feedbackArcSet.acyclicEdges.insert(feedbackArcSet.acyclicEdges.begin(),
                                       {feedbackEdge.first, scratchReg});
    // The scratch register's use as source is its last use, so set MOVE.
    // This ensures the ref in the scratch register is released after the copy,
    // preventing leaks when the branch takes an alternate path (e.g., loop
    // exit).
    Register scratchSrc = scratchReg;
    if (scratchSrc.isRef()) {
      scratchSrc.setMove(true);
    }
    feedbackArcSet.acyclicEdges.push_back({scratchSrc, feedbackEdge.second});
  }
  if (localScratchI32RegCount > 0) {
    scratchI32RegisterCount_ =
        std::max(scratchI32RegisterCount_, localScratchI32RegCount);
    assert(getMaxI32RegisterOrdinal() <= Register::kInt32RegisterCount &&
           "spilling i32 regs");
    if (getMaxI32RegisterOrdinal() > Register::kInt32RegisterCount &&
        branchOp) {
      mlir::emitError(branchOp->getLoc())
          << "spilling entire i32 register address space";
    }
  }
  if (localScratchRefRegCount > 0) {
    scratchRefRegisterCount_ =
        std::max(scratchRefRegisterCount_, localScratchRefRegCount);
    assert(getMaxRefRegisterOrdinal() <= Register::kRefRegisterCount &&
           "spilling ref regs");
    if (getMaxRefRegisterOrdinal() > Register::kRefRegisterCount && branchOp) {
      mlir::emitError(branchOp->getLoc())
          << "spilling entire ref register address space";
    }
  }

  return feedbackArcSet.acyclicEdges;
}

} // namespace mlir::iree_compiler

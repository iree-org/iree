// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/compiler/Dialect/VM/Analysis/RegisterAllocation.h"

#include <algorithm>
#include <map>
#include <utility>

#include "iree/compiler/Dialect/IREE/IR/IREETypes.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"

namespace mlir {
namespace iree_compiler {

static Attribute getStrArrayAttr(Builder &builder,
                                 ArrayRef<std::string> values) {
  return builder.getStrArrayAttr(llvm::to_vector<8>(llvm::map_range(
      values, [](const std::string &value) { return StringRef(value); })));
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
      uint16_t reg = registerAllocation.map_[blockArg];
      blockRegStrs.push_back(std::to_string(reg));
    }
    block.front().setAttr("block_registers",
                          getStrArrayAttr(builder, blockRegStrs));

    for (auto &op : block.getOperations()) {
      if (op.getNumResults() == 0) continue;
      SmallVector<std::string, 8> regStrs;
      regStrs.reserve(op.getNumResults());
      for (auto result : op.getResults()) {
        uint16_t reg = registerAllocation.map_[result];
        regStrs.push_back(std::to_string(reg));
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
          remappingStrs.push_back(
              llvm::formatv("{0}->{1}", srcDstReg.first, srcDstReg.second)
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

// Forms a register reference byte as interpreted by the VM.
// Assumes that the ordinal has been constructed in the valid range.
static uint16_t makeRegisterByte(Type type, int ordinal, bool isMove) {
  if (type.isSignlessIntOrIndexOrFloat()) {
    assert(ordinal < kIntRegisterCount);
    return ordinal;
  } else {
    assert(ordinal < kRefRegisterCount);
    return (ordinal | kRefRegisterTypeBit) | (isMove ? kRefRegisterMoveBit : 0);
  }
}

// Bitmaps set indicating which registers of which banks are in use.
struct RegisterUsage {
  llvm::BitVector intRegisters{kIntRegisterCount};
  llvm::BitVector refRegisters{kRefRegisterCount};
  int maxI32RegisterOrdinal = -1;
  int maxRefRegisterOrdinal = -1;

  void reset() {
    intRegisters.reset();
    refRegisters.reset();
    maxI32RegisterOrdinal = -1;
    maxRefRegisterOrdinal = -1;
  }

  Optional<uint16_t> allocateRegister(Type type) {
    if (type.isSignlessIntOrIndexOrFloat()) {
      int ordinal = intRegisters.find_first_unset();
      if (ordinal >= kIntRegisterCount) {
        return {};
      }
      intRegisters.set(ordinal);
      maxI32RegisterOrdinal = std::max(ordinal, maxI32RegisterOrdinal);
      return makeRegisterByte(type, ordinal, /*isMove=*/false);
    } else {
      int ordinal = refRegisters.find_first_unset();
      if (ordinal >= kRefRegisterCount) {
        return {};
      }
      refRegisters.set(ordinal);
      maxRefRegisterOrdinal = std::max(ordinal, maxRefRegisterOrdinal);
      return makeRegisterByte(type, ordinal, /*isMove=*/false);
    }
  }

  void markRegisterUsed(uint16_t reg) {
    int ordinal = getRegisterOrdinal(reg);
    if (isRefRegister(reg)) {
      refRegisters.set(ordinal);
      maxRefRegisterOrdinal = std::max(ordinal, maxRefRegisterOrdinal);
    } else {
      intRegisters.set(ordinal);
      maxI32RegisterOrdinal = std::max(ordinal, maxI32RegisterOrdinal);
    }
  }

  void releaseRegister(uint16_t reg) {
    if (isRefRegister(reg)) {
      refRegisters.reset(reg & kRefRegisterCount);
    } else {
      intRegisters.reset(reg & kIntRegisterCount);
    }
  }
};

// Sorts blocks in dominance order such that the entry block is first and
// all of the following blocks are dominated only by blocks that have come
// before them in the list. This ensures that we always know all registers for
// block live-in values as we walk the blocks.
static SmallVector<Block *, 8> sortBlocksInDominanceOrder(
    IREE::VM::FuncOp funcOp) {
  DominanceInfo dominanceInfo(funcOp);
  llvm::SmallSetVector<Block *, 8> unmarkedBlocks;
  for (auto &block : funcOp.getBlocks()) {
    unmarkedBlocks.insert(&block);
  }
  llvm::SmallSetVector<Block *, 8> markedBlocks;
  std::function<void(Block *)> visit = [&](Block *block) {
    if (markedBlocks.count(block) > 0) return;
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

// NOTE: this is not a good algorithm, nor is it a good allocator. If you're
// looking at this and have ideas of how to do this for real please feel
// free to rip it all apart :)
//
// Because I'm lazy we really only look at individual blocks at a time. It'd
// be much better to use dominance info to track values across blocks and
// ensure we are avoiding as many moves as possible. The special case we need to
// handle is when values are not defined within the current block (as values in
// dominators are allowed to cross block boundaries outside of arguments).
LogicalResult RegisterAllocation::recalculate(IREE::VM::FuncOp funcOp) {
  map_.clear();

  if (failed(liveness_.recalculate(funcOp))) {
    return funcOp.emitError()
           << "failed to caclculate required liveness information";
  }

  scratchI32RegisterCount_ = 0;
  scratchRefRegisterCount_ = 0;

  // Walk the blocks in dominance order and build their register usage tables.
  // We are accumulating value->register mappings in |map_| as we go and since
  // we are traversing in order know that for each block we will have values in
  // the |map_| for all implicitly captured values.
  auto orderedBlocks = sortBlocksInDominanceOrder(funcOp);
  for (auto *block : orderedBlocks) {
    // Use the block live-in info to populate the register usage info at block
    // entry. This way if the block is dominated by multiple blocks or the
    // live-out of the dominator is a superset of this blocks live-in we are
    // only working with the minimal set.
    RegisterUsage registerUsage;
    for (auto liveInValue : liveness_.getBlockLiveIns(block)) {
      registerUsage.markRegisterUsed(mapToRegister(liveInValue));
    }

    // Allocate arguments first from left-to-right.
    for (auto blockArg : block->getArguments()) {
      auto reg = registerUsage.allocateRegister(blockArg.getType());
      if (!reg.hasValue()) {
        return funcOp.emitError() << "register allocation failed for block arg "
                                  << blockArg.getArgNumber();
      }
      map_[blockArg] = reg.getValue();
    }

    // Cleanup any block arguments that were unused. We do this after the
    // initial allocation above so that block arguments can never alias as that
    // makes things really hard to read. Ideally an optimization pass that
    // removes unused block arguments would prevent this from happening.
    for (auto blockArg : block->getArguments()) {
      if (blockArg.use_empty()) {
        registerUsage.releaseRegister(map_[blockArg]);
      }
    }

    for (auto &op : block->getOperations()) {
      for (auto &operand : op.getOpOperands()) {
        if (liveness_.isLastValueUse(operand.get(), &op)) {
          registerUsage.releaseRegister(map_[operand.get()]);
        }
      }
      for (auto result : op.getResults()) {
        auto reg = registerUsage.allocateRegister(result.getType());
        if (!reg.hasValue()) {
          return op.emitError() << "register allocation failed for result "
                                << result.cast<OpResult>().getResultNumber();
        }
        map_[result] = reg.getValue();
        if (result.use_empty()) {
          registerUsage.releaseRegister(reg.getValue());
        }
      }
    }

    // Track the maximum register of each type used.
    maxI32RegisterOrdinal_ =
        std::max(maxI32RegisterOrdinal_, registerUsage.maxI32RegisterOrdinal);
    maxRefRegisterOrdinal_ =
        std::max(maxRefRegisterOrdinal_, registerUsage.maxRefRegisterOrdinal);
  }

  // We currently don't check during the allocation above. If we implement
  // spilling we could use this max information to reserve space for spilling.
  if (maxI32RegisterOrdinal_ > kIntRegisterCount ||
      maxRefRegisterOrdinal_ > kRefRegisterCount) {
    return funcOp.emitError() << "function overflows stack register banks; "
                                 "spilling to memory not yet implemented";
  }

  return success();
}

uint16_t RegisterAllocation::mapToRegister(Value value) {
  auto it = map_.find(value);
  assert(it != map_.end());
  return it->getSecond();
}

uint16_t RegisterAllocation::mapUseToRegister(Value value, Operation *useOp,
                                              int operandIndex) {
  uint16_t reg = mapToRegister(value);
  if (isRefRegister(reg) &&
      liveness_.isLastValueUse(value, useOp, operandIndex)) {
    reg |= kRefRegisterMoveBit;
  }
  return reg;
}

// A feedback arc set containing the minimal list of cycle-causing edges.
// https://en.wikipedia.org/wiki/Feedback_arc_set
struct FeedbackArcSet {
  using NodeID = uint16_t;
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
    SmallVector<FASNode, 8> nodeStorage;
    llvm::SmallDenseMap<NodeID, FASNode *> nodes;
    for (auto &edge : inputEdges) {
      NodeID sourceID = getBaseRegister(edge.first);
      NodeID sinkID = getBaseRegister(edge.second);
      assert(sourceID != sinkID && "self-cycles not supported");
      if (nodes.count(sourceID) == 0) {
        nodeStorage.push_back({sourceID, 0, 0});
        nodes.insert({sourceID, &nodeStorage.back()});
      }
      if (nodes.count(sinkID) == 0) {
        nodeStorage.push_back({sinkID, 0, 0});
        nodes.insert({sinkID, &nodeStorage.back()});
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
      NodeID sourceID = getBaseRegister(edge.first);
      NodeID sinkID = getBaseRegister(edge.second);
      auto *sourceNode = nodes[sourceID];
      ++sourceNode->outdegree;
      maxOutdegree = std::max(maxOutdegree, sourceNode->outdegree);
      auto *sinkNode = nodes[sinkID];
      ++sinkNode->indegree;
      maxIndegree = std::max(maxIndegree, sinkNode->indegree);
      edges.push_back({sourceNode, sinkNode});
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
    for (auto &nodeEntry : nodes) {
      assignBucket(nodeEntry.second);
    }

    auto removeNode = [&](FASNode *node) {
      SmallVector<FASEdge, 4> inEdges;
      inEdges.reserve(node->indegree);
      SmallVector<FASEdge, 4> outEdges;
      outEdges.reserve(node->outdegree);
      for (auto &edge : edges) {
        if (edge.sink == node) inEdges.push_back(edge);
        if (edge.source == node) outEdges.push_back(edge);
      }
      bool collectInEdges = node->indegree <= node->outdegree;
      bool collectOutEdges = !collectInEdges;

      SmallVector<Edge, 4> results;
      for (auto &edge : inEdges) {
        if (edge.source == node) continue;
        if (collectInEdges) {
          results.push_back({edge.source->id, edge.sink->id});
        }
        removeBucket(edge.source);
        --edge.source->outdegree;
        assignBucket(edge.source);
      }
      for (auto &edge : outEdges) {
        if (edge.sink == node) continue;
        if (collectOutEdges) {
          results.push_back({edge.source->id, edge.sink->id});
        }
        removeBucket(edge.sink);
        --edge.sink->indegree;
        assignBucket(edge.sink);
      }

      nodes.erase(node->id);
      edges.erase(std::remove_if(edges.begin(), edges.end(),
                                 [&](const FASEdge &edge) {
                                   return edge.source == node ||
                                          edge.sink == node;
                                 }),
                  edges.end());
      return results;
    };
    auto ends = buckets.back();
    while (!nodes.empty()) {
      while (!ends.empty()) {
        auto *node = ends.front();
        ends.erase(ends.begin());
        removeNode(node);
      }
      if (nodes.empty()) break;
      for (int i = buckets.size() - 1; i >= 0; --i) {
        if (buckets[i].empty()) continue;
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
      auto it = std::find_if(
          result.feedbackEdges.begin(), result.feedbackEdges.end(),
          [&](const Edge &edge) {
            return compareRegistersEqual(edge.first, inputEdge.first) &&
                   compareRegistersEqual(edge.second, inputEdge.second);
          });
      if (it == result.feedbackEdges.end()) {
        acyclicEdges.push_back(inputEdge);
        acyclicNodes.insert(getBaseRegister(inputEdge.first));
        acyclicNodes.insert(getBaseRegister(inputEdge.second));
      }
    }

    // Topologically sort the DAG so that we don't overwrite anything.
    llvm::SmallSetVector<NodeID, 8> unmarkedNodes = acyclicNodes;
    llvm::SmallSetVector<NodeID, 8> markedNodes;
    std::function<void(NodeID)> visit = [&](NodeID node) {
      if (markedNodes.count(node) > 0) return;
      for (auto &edge : acyclicEdges) {
        if (edge.first != node) continue;
        visit(edge.second);
      }
      markedNodes.insert(node);
    };
    while (!unmarkedNodes.empty()) {
      visit(unmarkedNodes.pop_back_val());
    }
    for (auto node : markedNodes.takeVector()) {
      for (auto &edge : acyclicEdges) {
        if (edge.first != node) continue;
        result.acyclicEdges.push_back({edge.first, edge.second});
      }
    }

    return result;
  }
};

SmallVector<std::pair<uint16_t, uint16_t>, 8>
RegisterAllocation::remapSuccessorRegisters(Operation *op, int successorIndex) {
  // Compute the initial directed graph of register movements.
  // This may contain cycles ([reg 0->1], [reg 1->0], ...) that would not be
  // possible to evaluate as a direct remapping.
  SmallVector<std::pair<uint16_t, uint16_t>, 8> srcDstRegs;
  auto *targetBlock = op->getSuccessor(successorIndex);
  auto operands =
      cast<BranchOpInterface>(op).getSuccessorOperands(successorIndex);
  for (auto it : llvm::enumerate(*operands)) {
    uint16_t srcReg = mapToRegister(it.value());
    BlockArgument targetArg = targetBlock->getArgument(it.index());
    uint16_t dstReg = mapToRegister(targetArg);
    if (!compareRegistersEqual(srcReg, dstReg)) {
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
  int scratchI32Reg = maxI32RegisterOrdinal_;
  int scratchRefReg = maxRefRegisterOrdinal_;
  for (auto feedbackEdge : feedbackArcSet.feedbackEdges) {
    uint16_t scratchReg;
    if (isRefRegister(feedbackEdge.first)) {
      scratchReg = ++scratchRefReg;
      scratchReg |= kRefRegisterTypeBit;
    } else {
      scratchReg = ++scratchI32Reg;
    }
    feedbackArcSet.acyclicEdges.insert(feedbackArcSet.acyclicEdges.begin(),
                                       {feedbackEdge.first, scratchReg});
    feedbackArcSet.acyclicEdges.push_back({scratchReg, feedbackEdge.second});
  }
  if (scratchI32Reg != maxI32RegisterOrdinal_) {
    scratchI32RegisterCount_ = scratchI32Reg - maxI32RegisterOrdinal_;
    assert(getMaxI32RegisterOrdinal() <= kIntRegisterCount &&
           "spilling i32 regs");
    if (getMaxI32RegisterOrdinal() > kIntRegisterCount) {
      op->emitOpError() << "spilling entire i32 register address space";
    }
  }
  if (scratchRefReg != maxRefRegisterOrdinal_) {
    scratchRefRegisterCount_ = scratchRefReg - maxRefRegisterOrdinal_;
    assert(getMaxRefRegisterOrdinal() <= kRefRegisterCount &&
           "spilling ref regs");
    if (getMaxRefRegisterOrdinal() > kRefRegisterCount) {
      op->emitOpError() << "spilling entire ref register address space";
    }
  }

  return feedbackArcSet.acyclicEdges;
}

}  // namespace iree_compiler
}  // namespace mlir

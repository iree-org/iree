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

#include <deque>
#include <memory>

#include "iree/compiler/IR/StructureOps.h"
#include "mlir/Analysis/Dominance.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Block.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Utils.h"

namespace mlir {
namespace iree_compiler {

namespace {

bool IsDead(Operation *op, llvm::DenseSet<Operation *> *deadOpsList) {
  for (auto *result : op->getResults()) {
    for (auto *user : result->getUsers()) {
      if (deadOpsList->count(user) == 0) {
        return false;
      }
    }
  }
  return true;
}

void dceBlock(Block *block, llvm::DenseSet<Operation *> *deadOpsList) {
  // Iterate in reverse so that we can do this in one pass per block.
  std::vector<Operation *> opsToEraseList;
  for (auto &op : llvm::reverse(block->getOperations())) {
    // Ignore terminators.
    if (op.isKnownTerminator()) {
      continue;
    }

    // Ignore ops that have side effects as we can't prove they are dead
    // (yet?).
    // TODO(b/135053584): IREE awareness for out params.
    bool shouldBeSideEffectFree = isa<LoadOp>(op) || isa<AllocOp>(op);
    if (!shouldBeSideEffectFree && !op.hasNoSideEffect()) {
      continue;
    }

    // Ignore stores.
    // TODO(benvanik): IREE awareness of last use/liveness.
    if (isa<StoreOp>(op)) {
      continue;
    }

    if (IsDead(&op, deadOpsList)) {
      deadOpsList->insert(&op);
      opsToEraseList.push_back(&op);
    }
  }

  for (auto *op : opsToEraseList) {
    op->erase();
  }
}

struct CFGStackNode {
  explicit CFGStackNode(DominanceInfoNode *node)
      : node(node), childIterator(node->begin()) {}
  DominanceInfoNode *node;
  DominanceInfoNode::iterator childIterator;
  bool processed = false;
};

void dceRegion(DominanceInfo &domInfo, Region &region) {
  if (region.empty()) {
    return;
  }

  llvm::DenseSet<Operation *> deadOpsList;

  // Depth-first post-order traversal so that we go backwards.
  std::deque<std::unique_ptr<CFGStackNode>> stack;
  stack.emplace_back(
      std::make_unique<CFGStackNode>(domInfo.getRootNode(&region)));
  while (!stack.empty()) {
    auto &currentNode = stack.back();
    if (currentNode->childIterator != currentNode->node->end()) {
      auto *childNode = *(currentNode->childIterator++);
      stack.emplace_back(std::make_unique<CFGStackNode>(childNode));
    } else {
      if (!currentNode->processed) {
        currentNode->processed = true;
        dceBlock(currentNode->node->getBlock(), &deadOpsList);
      }
      stack.pop_back();
    }
  }
}

}  // namespace

class AggressiveOpEliminationPass
    : public FunctionPass<AggressiveOpEliminationPass> {
 public:
  void runOnFunction() override {
    dceRegion(getAnalysis<DominanceInfo>(), getFunction().getBody());
    markAnalysesPreserved<DominanceInfo, PostDominanceInfo>();
  }
};

std::unique_ptr<OpPassBase<FuncOp>> createAggressiveOpEliminationPass() {
  return std::make_unique<AggressiveOpEliminationPass>();
}

static PassRegistration<AggressiveOpEliminationPass> pass(
    "iree-aggressive-op-elimination",
    "Eliminate ops that have no side-effects");

}  // namespace iree_compiler
}  // namespace mlir

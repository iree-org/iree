// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <iterator>

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

#define DEBUG_TYPE "iree-flow-simplify-variable-accesses"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

// Builds symbol ref set for all immutable variables in |moduleOp|.
static DenseSet<StringRef> gatherImmutableVariables(ModuleOp moduleOp) {
  DenseSet<StringRef> set;
  for (auto variableOp : moduleOp.getOps<IREE::Flow::VariableOp>()) {
    if (!variableOp.is_mutable()) {
      set.insert(variableOp.sym_name());
    }
  }
  return set;
}

// Hoists all loads of immutable variables in |funcOp| to the entry block.
// |immutableVariables| is used for lookups of which variables are immutable.
static void hoistImmutableLoads(FuncOp funcOp,
                                DenseSet<StringRef> &immutableVariables) {
  // Since CSE of loads isn't a thing yet we perform a basic deduping here by
  // folding all subsequent loads into the first one found. This works only for
  // immutable variables as otherwise we'd have to ensure stores and
  // side-effects were properly observed.
  DenseMap<Attribute, Operation *> loadOps;
  auto *entryBlock = &funcOp.getBlocks().front();
  Operation *lastEntryOp = nullptr;
  SmallVector<std::pair<Operation *, Operation *>> opReplacements;
  for (auto &block : funcOp) {
    auto ops = llvm::to_vector<8>(block.getOps<IREE::Flow::VariableLoadOp>());
    for (auto &op : ops) {
      if (!immutableVariables.contains(op.variable())) continue;
      auto variableRef = op.variableAttr().cast<Attribute>();
      auto it = loadOps.find(variableRef);
      if (it == loadOps.end()) {
        // Move to entry block; even if it's already there (so loads are
        // hoisted at the same time).
        LLVM_DEBUG(llvm::dbgs() << "moving immutable variable " << op.variable()
                                << " load to the entry block\n");
        if (lastEntryOp) {
          op->moveAfter(lastEntryOp);
        } else {
          op->moveBefore(entryBlock, entryBlock->begin());
        }
        loadOps[variableRef] = op;
        lastEntryOp = op;
      } else {
        LLVM_DEBUG(llvm::dbgs()
                   << "CSE'ing immutable variable " << op.variable() << "\n");
        opReplacements.push_back({op, it->getSecond()});
      }
    }
  }
  for (auto &replacement : opReplacements) {
    replacement.first->replaceAllUsesWith(replacement.second);
    replacement.first->erase();
  }
}

static bool doesOpBlockMotion(Operation *op) {
  return isa<mlir::CallOpInterface>(op) ||
         op->hasTrait<OpTrait::IREE::Util::YieldPoint>();
}

static void moveOpUpInBlock(Block &block, Operation *op) {
  while (op->getPrevNode()) {
    if (doesOpBlockMotion(op->getPrevNode())) break;
    op->moveBefore(op->getPrevNode());
  }
}

static void moveOpDownInBlock(Block &block, Operation *op) {
  while (op->getNextNode() != block.getTerminator()) {
    if (doesOpBlockMotion(op->getNextNode())) break;
    op->moveAfter(op->getNextNode());
  }
}

// Optimizes the load/store ops for each given bucket.
// Returns true if any op was removed.
static bool optimizeBuckets(
    Block &block, std::map<StringRef, SmallVector<Operation *>> &buckets) {
  bool didRemoveAny = false;
  for (auto &bucket : buckets) {
    // First perform basic load-store forwarding and such.
    auto &ops = bucket.second;
    for (int i = ops.size() - 1; i >= 1; --i) {
      auto previous = ops[i - 1];
      auto current = ops[i];
      if (isa<IREE::Flow::VariableStoreOp>(previous) &&
          isa<IREE::Flow::VariableLoadOp>(current)) {
        // RAW - forward the stored variable to the following use.
        auto storedValue = previous->getOperand(0);
        LLVM_DEBUG({
          llvm::dbgs() << "RAW: replacing load with previous store value:\n";
          current->dump();
          llvm::dbgs() << "->\n";
          storedValue.dump();
        });
        current->replaceAllUsesWith(ValueRange{storedValue});
        ops.erase(ops.begin() + i);
        current->erase();
        didRemoveAny = true;
      } else if (isa<IREE::Flow::VariableLoadOp>(previous) &&
                 isa<IREE::Flow::VariableLoadOp>(current)) {
        // RAR - forward the loaded variable to the following use.
        LLVM_DEBUG({
          llvm::dbgs() << "RAR: replacing subsequent load with op:\n";
          current->dump();
          llvm::dbgs() << "->\n";
          previous->dump();
        });
        current->replaceAllUsesWith(previous);
        ops.erase(ops.begin() + i);
        current->erase();
        didRemoveAny = true;
      } else if (isa<IREE::Flow::VariableStoreOp>(previous) &&
                 isa<IREE::Flow::VariableStoreOp>(current)) {
        // WAW - remove the first store.
        LLVM_DEBUG({
          llvm::dbgs() << "WAW: erasing source op:\n";
          previous->dump();
          llvm::dbgs() << "\nand keeping subsequent op:\n";
          current->dump();
        });
        ops.erase(ops.begin() + i - 1);
        previous->erase();
        didRemoveAny = true;
      }
    }
    if (ops.empty()) continue;

    if (auto loadOp = dyn_cast<IREE::Flow::VariableLoadOp>(ops.front())) {
      // If the head op is a load we can move that to the top of the block.
      LLVM_DEBUG(llvm::dbgs() << "moving mutable variable " << loadOp.variable()
                              << " load upward\n");
      moveOpUpInBlock(block, ops.front());
    }
    if (auto storeOp = dyn_cast<IREE::Flow::VariableStoreOp>(ops.back())) {
      // If the tail op is a store we can move that to the bottom of the block.
      LLVM_DEBUG(llvm::dbgs() << "moving mutable variable "
                              << storeOp.variable() << " store downward\n");
      moveOpDownInBlock(block, ops.back());
    }
  }
  return didRemoveAny;
}

// Hoists loads and sinks stores to the boundary of |block| when safe.
// |immutableVariables| is used for lookups of which variables are immutable.
//
// Basic algorithm (repeat until no op removals):
//   for each op:
//     if immutable: skip
//     add to load/store buckets (sorted vector)
//   for each bucket (symbol):
//     walk ops in reverse:
//       if (prev == store && this == load)  // RAW
//         replace load with store source
//       if (prev == load && this == load)  // RAR
//         replace with first load
//       if (prev == store && this == store) // WAW
//         remove first store
//     if (head == load) move load to front
//     if (tail == store) move store to back
//
// Returns true if there were any removals and the block should be reprocessed.
static bool rearrangeBlockVariableAccesses(
    Block &block, DenseSet<StringRef> &immutableVariables) {
  // Gather sequences of operations that are safe to reorder.
  // Certain ops - like calls/do_not_optimize/etc - prevent us from moving any
  // variable operations across them.
  //
  // From each sequence we produce [symbol_name, [op, op, op, ...]] buckets.
  // NOTE: we use a map here so that we are deterministically ordered. This may
  // not be needed but the variable count is low and it's nice to not care about
  // op order issues.
  SmallVector<std::map<StringRef, SmallVector<Operation *>>> sequencedBuckets;
  sequencedBuckets.push_back({});  // Start in a sequence.
  block.walk([&](Operation *op) {
    auto &buckets = sequencedBuckets.back();
    if (auto loadOp = dyn_cast<IREE::Flow::VariableLoadOp>(op)) {
      if (!immutableVariables.contains(loadOp.variable())) {
        buckets[loadOp.variable()].push_back(op);
      }
    } else if (auto storeOp = dyn_cast<IREE::Flow::VariableStoreOp>(op)) {
      buckets[storeOp.variable()].push_back(op);
    } else if (doesOpBlockMotion(op)) {
      // Split point - all accesses after this point must not assume anything
      // about accesses before it.
      if (!buckets.empty()) {
        sequencedBuckets.push_back({});
      }
    }
  });
  bool didRemoveAny = false;
  for (auto &buckets : sequencedBuckets) {
    didRemoveAny = optimizeBuckets(block, buckets) || didRemoveAny;
  }
  return didRemoveAny;
}

namespace {

class SimplifyVariableAccessesPass
    : public SimplifyVariableAccessesBase<SimplifyVariableAccessesPass> {
 public:
  void runOnOperation() override {
    auto funcOp = getOperation();
    if (funcOp.empty()) return;

    auto moduleOp = funcOp->getParentOfType<mlir::ModuleOp>();
    assert(moduleOp && "func not in a module");

    // Build a set of all immutable variables for fast lookup.
    auto immutableVariables = gatherImmutableVariables(moduleOp);

    // Hoist immutable variables first. These have no hazards and don't care
    // about control flow - like `constant` - so getting them handled first
    // avoids the need for us to do the full analysis.
    hoistImmutableLoads(funcOp, immutableVariables);

    // We can't optimize the function if there are indirect loads/stores.
    // Note that constant loads are still ok above.
    for (auto &block : funcOp) {
      for (auto &op : block) {
        if (isa<IREE::Flow::VariableLoadIndirectOp>(op) ||
            isa<IREE::Flow::VariableStoreIndirectOp>(op)) {
          LLVM_DEBUG(llvm::dbgs()
                     << "bailing on variable access simplification: indirect "
                        "accesses present in function\n");
          return;
        }
      }
    }

    // For each block in the function hoist loads and sink stores.
    // This does no cross-block movement, though it really should. Maybe when a
    // real compiler engineer sees this they'll be inspired to do this properly.
    for (auto &block : funcOp) {
      LLVM_DEBUG(llvm::dbgs() << "==== REARRANGING BLOCK ACCESSES ====\n");
      while (rearrangeBlockVariableAccesses(block, immutableVariables)) {
        // NOTE: block is processed until no more ops are removed. Will always
        // end in a fixed amount of time as ops are only removed from the block.
      }
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createSimplifyVariableAccessesPass() {
  return std::make_unique<SimplifyVariableAccessesPass>();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

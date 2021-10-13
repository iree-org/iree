// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SetVector.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/RegionGraphTraits.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

namespace {

template <typename Range, typename Pred>
bool compare_ranges(Range &&lhs, Range &&rhs, Pred pred) {
  auto lhsIt = lhs.begin();
  auto rhsIt = rhs.begin();
  while (lhsIt != lhs.end() && rhsIt != rhs.end()) {
    if (!pred(*lhsIt++, *rhsIt++)) return false;
  }
  if ((lhsIt == lhs.end()) != (rhsIt == rhs.end())) {
    // Block count mismatch. We do this here so that we avoid the O(n) scan
    // that would have been required to calculate the size above.
    return false;
  }
  return true;
}

static bool isStructurallyEquivalentTo(Region &lhs, Region &rhs,
                                       BlockAndValueMapping &parentMapping);
static bool isStructurallyEquivalentTo(Operation &lhs, Operation &rhs,
                                       BlockAndValueMapping &parentMapping);

// Recursively compares two regions for structural equivalence.
// Structural equivalence ensures that operations on both the |lhs| and |rhs|
// have the same attributes and same use-def structure.
//
// Example:
//   func @lhs(%arg0 : index) -> index {
//     %c1 = arith.constant 1 : index
//     %0 = add %arg0, %c1 : index
//     return %0 : index
//   }
//   func @rhs(%arg0 : index) -> index {
//     %c1 = arith.constant 1 : index
//     %0 = add %arg0, %c1 : index
//     return %0 : index
//   }
//
//   assert(isStructurallyEquivalentTo(lhs.getBody(), rhs.getBody()));
//
// TODO(#3996): upstream into mlir::OperationEquivalence if this works.
// TODO(#3996): add symbol ref comparison (add to BlockAndValueMapping).
static bool isStructurallyEquivalentTo(Region &lhs, Region &rhs) {
  BlockAndValueMapping mapping;
  return isStructurallyEquivalentTo(lhs, rhs, mapping);
}

static bool isStructurallyEquivalentTo(Region &lhs, Region &rhs,
                                       BlockAndValueMapping &mapping) {
  // Use compare_ranges to walk the block list in parallel and get a boolean in
  // the case of size mismatch without an O(N) linked-list size query.
  if (!compare_ranges(
          lhs.getBlocks(), rhs.getBlocks(),
          [&](Block &lhsBlock, Block &rhsBlock) {
            if (lhsBlock.getNumArguments() != rhsBlock.getNumArguments()) {
              return false;
            }
            for (auto argPair :
                 llvm::zip(lhsBlock.getArguments(), rhsBlock.getArguments())) {
              auto &lhsArg = std::get<0>(argPair);
              auto &rhsArg = std::get<1>(argPair);
              if (lhsArg.getType() != rhsArg.getType()) return false;
              mapping.map(lhsArg, rhsArg);
            }
            mapping.map(&lhsBlock, &rhsBlock);
            return true;
          })) {
    return false;  // block mismatch
  }

  // Walk the blocks again now that we have a populated mapping.
  // We do this in topological order so that we have all values required by a
  // block mapped by the time we reach it observing transitive block dominance.
  llvm::SetVector<Block *> lhsBlocks;
  for (Block &b : lhs.getBlocks()) {
    llvm::ReversePostOrderTraversal<Block *> traversal(&b);
    lhsBlocks.insert(traversal.begin(), traversal.end());
  }
  llvm::SetVector<Block *> rhsBlocks;
  for (Block &b : rhs.getBlocks()) {
    llvm::ReversePostOrderTraversal<Block *> traversal(&b);
    rhsBlocks.insert(traversal.begin(), traversal.end());
  }
  for (auto blockPair : llvm::zip(lhsBlocks, rhsBlocks)) {
    auto &lhsBlock = std::get<0>(blockPair);
    auto &rhsBlock = std::get<1>(blockPair);
    for (auto opPair :
         llvm::zip(lhsBlock->getOperations(), rhsBlock->getOperations())) {
      auto &lhsOp = std::get<0>(opPair);
      auto &rhsOp = std::get<1>(opPair);
      if (!isStructurallyEquivalentTo(lhsOp, rhsOp, mapping)) {
        return false;
      }
    }
  }

  // Equivalent!
  return true;
}
static bool isStructurallyEquivalentTo(Operation &lhs, Operation &rhs,
                                       BlockAndValueMapping &parentMapping) {
  // Check operation metadata for early-exit opportunities.
  if (lhs.getName() != rhs.getName()) return false;
  if (lhs.getNumOperands() != rhs.getNumOperands()) return false;
  if (lhs.getNumResults() != rhs.getNumResults()) return false;
  if (lhs.getNumRegions() != rhs.getNumRegions()) return false;
  if (lhs.getNumSuccessors() != rhs.getNumSuccessors()) return false;

  // TODO(#3996): symbol mapping; for now allow them to differ unconditionally.
  if (!compare_ranges(
          lhs.getAttrs(), rhs.getAttrs(),
          [&](const NamedAttribute &lhs, const NamedAttribute &rhs) {
            if (lhs.first == "function_ref" ||
                lhs.first == SymbolTable::getSymbolAttrName()) {
              return true;
            }
            return lhs == rhs;
          })) {
    return false;
  }

  // If the op references blocks (such as a branch) then we expect to have them
  // in the mapping already from the parent region to do the lhs->rhs mapping.
  for (auto successorPair :
       llvm::zip(lhs.getSuccessors(), rhs.getSuccessors())) {
    auto *lhsSuccessor = std::get<0>(successorPair);
    auto *rhsSuccessor = std::get<1>(successorPair);
    if (rhsSuccessor != parentMapping.lookup(lhsSuccessor)) return false;
  }

  // Ensure result types match first and add to the block and value mapping.
  // For many ops if the result types don't match it's a good (cheap) indicator
  // that the operands won't match either so this still allows a somewhat-early
  // exit prior to the full traversal.
  for (auto resultPair : llvm::zip(lhs.getResults(), rhs.getResults())) {
    auto &lhsValue = std::get<0>(resultPair);
    auto &rhsValue = std::get<1>(resultPair);
    if (lhsValue.getType() != rhsValue.getType()) return false;
    parentMapping.map(lhsValue, rhsValue);
  }

  // Check operands using the lhs->rhs mapping; since this op is only consuming
  // these values they should already be defined in the mapping.
  for (auto operandPair : llvm::zip(lhs.getOperands(), rhs.getOperands())) {
    auto &lhsValue = std::get<0>(operandPair);
    auto &rhsValue = std::get<1>(operandPair);
    if (lhsValue.getType() != rhsValue.getType()) return false;
    if (rhsValue != parentMapping.lookup(lhsValue)) return false;
  }

  // Recurse into regions.
  for (auto regionPair : llvm::zip(lhs.getRegions(), rhs.getRegions())) {
    auto &lhsRegion = std::get<0>(regionPair);
    auto &rhsRegion = std::get<1>(regionPair);

    // If the region is isolated we don't want to reuse any parent mapping or
    // pollute it with our mappings.
    BlockAndValueMapping scopedRegionMapping;
    BlockAndValueMapping regionMapping =
        lhs.hasTrait<OpTrait::IsIsolatedFromAbove>() ? scopedRegionMapping
                                                     : parentMapping;

    if (!isStructurallyEquivalentTo(lhsRegion, rhsRegion, regionMapping)) {
      return false;
    }
  }

  // Equivalent!
  return true;
}

bool areExecutablesEquivalent(ExecutableOp lhs, ExecutableOp rhs) {
  auto lhsModule = lhs.getInnerModule();
  auto rhsModule = rhs.getInnerModule();

  // Must have the same number of entry point ops, with the same attributes.
  // Entry point op symbol names are expected to differ, that won't affect
  // equivalence.
  if (!compare_ranges(lhsModule.getOps<DispatchEntryOp>(),
                      rhsModule.getOps<DispatchEntryOp>(),
                      [](DispatchEntryOp lhs, DispatchEntryOp rhs) {
                        return lhs->getAttrs() == rhs->getAttrs();
                      })) {
    return false;  // dispatch entry mismatch
  }

  // Walk all functions and ensure equivalent.
  if (!compare_ranges(
          lhsModule.getOps<mlir::FuncOp>(), rhsModule.getOps<mlir::FuncOp>(),
          [](mlir::FuncOp lhs, mlir::FuncOp rhs) {
            if (lhs.getType() != rhs.getType()) return false;
            if (lhs->getAttrs() != rhs->getAttrs()) return false;
            return isStructurallyEquivalentTo(lhs.getRegion(), rhs.getRegion());
          })) {
    return false;  // dispatch entry mismatch
  }

  return true;
}

// Replaces each usage of an entry point with its original symbol name with a
// new symbol name.
void replaceEntryPointUses(
    mlir::ModuleOp moduleOp,
    const DenseMap<Attribute, SymbolRefAttr> &replacements) {
  for (auto funcOp : moduleOp.getOps<mlir::FuncOp>()) {
    funcOp.walk([&](DispatchOp dispatchOp) {
      auto it = replacements.find(dispatchOp.entry_point());
      if (it != replacements.end()) {
        dispatchOp.entry_pointAttr(it->second.cast<SymbolRefAttr>());
      }
    });
  }
}

}  // namespace

class DeduplicateExecutablesPass
    : public DeduplicateExecutablesBase<DeduplicateExecutablesPass> {
 public:
  explicit DeduplicateExecutablesPass() {}
  DeduplicateExecutablesPass(const DeduplicateExecutablesPass &pass) {}

  void runOnOperation() override {
    auto moduleOp = getOperation();
    auto executableOps = llvm::to_vector<8>(moduleOp.getOps<ExecutableOp>());
    auto builder = OpBuilder::atBlockBegin(moduleOp.getBody());

    SmallVector<ExecutableOp, 3> duplicateExecutableOps;
    DenseMap<Attribute, SymbolRefAttr> entryPointRefReplacements;

    // For each executable, find the first executable which it is equivalent to.
    for (int i = executableOps.size() - 1; i >= 0; --i) {
      auto duplicateExecutableOp = executableOps[i];

      for (int j = 0; j < i; ++j) {
        auto referenceExecutableOp = executableOps[j];
        if (!isStructurallyEquivalentTo(
                duplicateExecutableOp.getBodyRegion(),
                referenceExecutableOp.getBodyRegion())) {
          continue;
        }

        // Found an equivalent executable! Record it and move on to the next.
        duplicateExecutableOps.push_back(duplicateExecutableOp);

        // Record entry point reference replacements.
        for (auto entryOpPair : llvm::zip(
                 duplicateExecutableOp.getBlock().getOps<DispatchEntryOp>(),
                 referenceExecutableOp.getBlock().getOps<DispatchEntryOp>())) {
          auto oldSymbolRefAttr = SymbolRefAttr::get(
              builder.getContext(), duplicateExecutableOp.getName(),
              {SymbolRefAttr::get(builder.getContext(),
                                  std::get<0>(entryOpPair).sym_name())});
          auto newSymbolRefAttr = SymbolRefAttr::get(
              builder.getContext(), referenceExecutableOp.getName(),
              {SymbolRefAttr::get(builder.getContext(),
                                  std::get<1>(entryOpPair).sym_name())});
          entryPointRefReplacements[oldSymbolRefAttr] = newSymbolRefAttr;
        }

        break;
      }
    }

    totalExecutables = executableOps.size();
    executablesDeduplicated = duplicateExecutableOps.size();
    remainingExecutables = totalExecutables - executablesDeduplicated;

    replaceEntryPointUses(moduleOp, entryPointRefReplacements);

    // Remove the duplicate executables now that they are no longer referenced.
    //
    // Note: removing executables can leave gaps in numbering if they were
    // originally numbered. While we could renumber them, we choose to keep
    // original names (numbers and all) to make it easier to track executables
    // through this pass.
    for (auto executableOp : duplicateExecutableOps) {
      executableOp.erase();
    }
  }

 private:
  Statistic totalExecutables{
      this, "total executable(s)",
      "Number of flow.executable ops before deduplication"};
  Statistic executablesDeduplicated{
      this, "duplicate executable(s)",
      "Number of flow.executable ops removed as duplicates"};
  Statistic remainingExecutables{
      this, "unique executable(s)",
      "Number of flow.executable ops remaining after deduplication"};
};

std::unique_ptr<OperationPass<mlir::ModuleOp>>
createDeduplicateExecutablesPass() {
  return std::make_unique<DeduplicateExecutablesPass>();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

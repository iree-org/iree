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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
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
    if (!pred(*lhsIt++, *rhsIt++))
      return false;
  }
  if ((lhsIt == lhs.end()) != (rhsIt == rhs.end())) {
    // Block count mismatch. We do this here so that we avoid the O(n) scan
    // that would have been required to calculate the size above.
    return false;
  }
  return true;
}

static bool isStructurallyEquivalentTo(Region &lhs, Region &rhs,
                                       IRMapping &parentMapping);
static bool isStructurallyEquivalentTo(Operation &lhs, Operation &rhs,
                                       IRMapping &parentMapping);

// Recursively compares two regions for structural equivalence.
// Structural equivalence ensures that operations on both the |lhs| and |rhs|
// have the same attributes and same use-def structure.
//
// Example:
//   func.func @lhs(%arg0 : index) -> index {
//     %c1 = arith.constant 1 : index
//     %0 = add %arg0, %c1 : index
//     return %0 : index
//   }
//   func.func @rhs(%arg0 : index) -> index {
//     %c1 = arith.constant 1 : index
//     %0 = add %arg0, %c1 : index
//     return %0 : index
//   }
//
//   assert(isStructurallyEquivalentTo(lhs.getBody(), rhs.getBody()));
//
// TODO(#3996): upstream into mlir::OperationEquivalence if this works.
// TODO(#3996): add symbol ref comparison (add to IRMapping).
static bool isStructurallyEquivalentTo(Region &lhs, Region &rhs) {
  IRMapping mapping;
  return isStructurallyEquivalentTo(lhs, rhs, mapping);
}

static bool isStructurallyEquivalentTo(Region &lhs, Region &rhs,
                                       IRMapping &mapping) {
  // Use compare_ranges to walk the block list in parallel and get a boolean in
  // the case of size mismatch without an O(N) linked-list size query.
  if (!compare_ranges(
          lhs.getBlocks(), rhs.getBlocks(),
          [&](Block &lhsBlock, Block &rhsBlock) {
            if (lhsBlock.getNumArguments() != rhsBlock.getNumArguments()) {
              return false;
            }
            for (auto [lhsArg, rhsArg] : llvm::zip_equal(
                     lhsBlock.getArguments(), rhsBlock.getArguments())) {
              if (lhsArg.getType() != rhsArg.getType())
                return false;
              mapping.map(lhsArg, rhsArg);
            }
            mapping.map(&lhsBlock, &rhsBlock);
            return true;
          })) {
    return false; // block mismatch
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
  if (lhsBlocks.size() != rhsBlocks.size())
    return false;
  for (auto [lhsBlock, rhsBlock] : llvm::zip_equal(lhsBlocks, rhsBlocks)) {
    auto &lhsOperations = lhsBlock->getOperations();
    auto &rhsOperations = rhsBlock->getOperations();
    if (lhsOperations.size() != rhsOperations.size())
      return false;
    for (auto [lhsOp, rhsOp] : llvm::zip_equal(lhsOperations, rhsOperations)) {
      if (!isStructurallyEquivalentTo(lhsOp, rhsOp, mapping)) {
        return false;
      }
    }
  }

  // Equivalent!
  return true;
}

static bool isStructurallyEquivalentTo(Operation &lhs, Operation &rhs,
                                       IRMapping &parentMapping) {
  // Check operation metadata for early-exit opportunities.
  if (lhs.getName() != rhs.getName())
    return false;
  if (lhs.getNumOperands() != rhs.getNumOperands())
    return false;
  if (lhs.getNumResults() != rhs.getNumResults())
    return false;
  if (lhs.getNumRegions() != rhs.getNumRegions())
    return false;
  if (lhs.getNumSuccessors() != rhs.getNumSuccessors())
    return false;

  // TODO(#3996): symbol mapping; for now allow them to differ unconditionally.
  if (!compare_ranges(
          lhs.getAttrs(), rhs.getAttrs(),
          [&](const NamedAttribute &lhs, const NamedAttribute &rhs) {
            if (lhs.getName() == "function_ref" ||
                lhs.getName() == SymbolTable::getSymbolAttrName()) {
              return true;
            }
            return lhs == rhs;
          })) {
    return false;
  }

  // If the op references blocks (such as a branch) then we expect to have them
  // in the mapping already from the parent region to do the lhs->rhs mapping.
  for (auto [lhsSuccessor, rhsSuccessor] :
       llvm::zip_equal(lhs.getSuccessors(), rhs.getSuccessors())) {
    if (rhsSuccessor != parentMapping.lookup(lhsSuccessor))
      return false;
  }

  // Ensure result types match first and add to the block and value mapping.
  // For many ops if the result types don't match it's a good (cheap) indicator
  // that the operands won't match either so this still allows a somewhat-early
  // exit prior to the full traversal.
  for (auto [lhsValue, rhsValue] :
       llvm::zip_equal(lhs.getResults(), rhs.getResults())) {
    if (lhsValue.getType() != rhsValue.getType())
      return false;
    parentMapping.map(lhsValue, rhsValue);
  }

  // Check operands using the lhs->rhs mapping; since this op is only consuming
  // these values they should already be defined in the mapping.
  for (auto [lhsValue, rhsValue] :
       llvm::zip_equal(lhs.getOperands(), rhs.getOperands())) {
    if (lhsValue.getType() != rhsValue.getType())
      return false;
    if (rhsValue != parentMapping.lookup(lhsValue))
      return false;
  }

  // Recurse into regions.
  for (auto [lhsRegion, rhsRegion] :
       llvm::zip_equal(lhs.getRegions(), rhs.getRegions())) {
    // If the region is isolated we don't want to reuse any parent mapping or
    // pollute it with our mappings.
    IRMapping scopedRegionMapping;
    IRMapping regionMapping = lhs.hasTrait<OpTrait::IsIsolatedFromAbove>()
                                  ? scopedRegionMapping
                                  : parentMapping;
    if (!isStructurallyEquivalentTo(lhsRegion, rhsRegion, regionMapping)) {
      return false;
    }
  }

  // Equivalent!
  return true;
}

// Replaces each usage of an entry point with its original symbol name with a
// new symbol name.
void replaceEntryPointUses(
    mlir::ModuleOp moduleOp,
    const DenseMap<Attribute, SymbolRefAttr> &replacements) {
  for (auto funcLikeOp : moduleOp.getOps<FunctionOpInterface>()) {
    funcLikeOp->walk([&](DispatchOp dispatchOp) {
      bool didChange = false;
      SmallVector<Attribute> newAttrs;
      for (auto oldAttr : dispatchOp.getEntryPoints()) {
        auto it = replacements.find(oldAttr);
        if (it != replacements.end()) {
          didChange = true;
          newAttrs.push_back(it->second);
        } else {
          newAttrs.push_back(oldAttr);
        }
      }
      if (didChange) {
        dispatchOp.setEntryPointsAttr(
            ArrayAttr::get(moduleOp.getContext(), newAttrs));
      }
    });
  }
}

} // namespace

class DeduplicateExecutablesPass
    : public DeduplicateExecutablesBase<DeduplicateExecutablesPass> {
public:
  explicit DeduplicateExecutablesPass() {}
  DeduplicateExecutablesPass(const DeduplicateExecutablesPass &pass) {}

  void runOnOperation() override {
    auto moduleOp = getOperation();

    // Bucket based on the hash of the names of at most the first 5 ops.
    // 5 was randomly chosen to be small enough to not increase overhead much,
    // but giving at least enough of a sample that there is some bucketing. This
    // was not empiraclly deetermined.
    llvm::MapVector<uint32_t, SmallVector<ExecutableOp, 3>> executableOpsMap;
    totalExecutables = 0;
    for (auto op : moduleOp.getOps<ExecutableOp>()) {
      int count = 0;
      llvm::hash_code hash(1);
      op.walk([&](Operation *it) {
        hash = llvm::hash_combine(hash, it->getName());
        return (++count >= 5) ? WalkResult::interrupt() : WalkResult::advance();
      });
      executableOpsMap[hash_value(hash)].push_back(op);
      ++totalExecutables;
    }

    auto builder = OpBuilder::atBlockBegin(moduleOp.getBody());
    SmallVector<ExecutableOp, 3> duplicateExecutableOps;
    DenseMap<Attribute, SymbolRefAttr> entryPointRefReplacements;

    // For each executable, find the first executable which it is equivalent to.
    for (auto &[key, executableOps] : executableOpsMap) {
      (void)key;
      for (int i = executableOps.size() - 1; i >= 0; --i) {
        auto duplicateExecutableOp = executableOps[i];

        for (int j = 0; j < i; ++j) {
          auto referenceExecutableOp = executableOps[j];
          if (!isStructurallyEquivalentTo(duplicateExecutableOp.getBody(),
                                          referenceExecutableOp.getBody())) {
            continue;
          }

          // Found an equivalent executable! Record it and move on to the next.
          duplicateExecutableOps.push_back(duplicateExecutableOp);

          // Record entry point reference replacements.
          for (auto [oldExportOp, newExportOp] :
               llvm::zip_equal(duplicateExecutableOp.getBlock()
                                   .getOps<ExecutableExportOp>(),
                               referenceExecutableOp.getBlock()
                                   .getOps<ExecutableExportOp>())) {
            auto oldSymbolRefAttr = SymbolRefAttr::get(
                builder.getContext(), duplicateExecutableOp.getName(),
                {SymbolRefAttr::get(builder.getContext(),
                                    oldExportOp.getSymName())});
            auto newSymbolRefAttr = SymbolRefAttr::get(
                builder.getContext(), referenceExecutableOp.getName(),
                {SymbolRefAttr::get(builder.getContext(),
                                    newExportOp.getSymName())});
            entryPointRefReplacements[oldSymbolRefAttr] = newSymbolRefAttr;
          }

          break;
        }
      }
    }

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

} // namespace Flow
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir

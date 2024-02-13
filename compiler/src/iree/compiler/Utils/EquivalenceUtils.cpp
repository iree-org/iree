// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Utils/EquivalenceUtils.h"

#include "llvm/ADT/PostOrderIterator.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/RegionGraphTraits.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir::iree_compiler {

OperationEquivalenceCache::OperationEquivalenceCache(MLIRContext *context)
    : functionRefName(StringAttr::get(context, "function_ref")),
      symbolAttrName(
          StringAttr::get(context, SymbolTable::getSymbolAttrName())) {}

OperationEquivalenceCache::~OperationEquivalenceCache() {
  for (auto *mapping : mappingFreeList)
    delete mapping;
  for (auto region : regions)
    delete region.second;
  for (auto block : blocks)
    delete block.second;
  for (auto op : ops)
    delete op.second;
}

bool OperationEquivalenceCache::isSymbolAttrName(StringAttr name) const {
  return name == functionRefName || name == symbolAttrName;
}

OperationEquivalenceCache::IRMappingPtr
OperationEquivalenceCache::acquireMapping() {
  IRMapping *mapping = nullptr;
  if (!mappingFreeList.empty()) {
    mapping = mappingFreeList.pop_back_val();
  } else {
    mapping = new IRMapping();
  }
  return IRMappingPtr(mapping, [this](IRMapping *mapping) {
    mapping->clear();
    mappingFreeList.push_back(mapping);
  });
}

OperationEquivalenceCache::RegionEntry &
OperationEquivalenceCache::getRegion(Region *region) {
  auto it = regions.find(region);
  if (it != regions.end())
    return *it->second;
  RegionEntry *entry = new RegionEntry();
  for (Block &block : region->getBlocks()) {
    llvm::ReversePostOrderTraversal<Block *> traversal(&block);
    entry->blocks.insert(traversal.begin(), traversal.end());
  }
  regions[region] = entry;
  return *entry;
}

OperationEquivalenceCache::BlockEntry &
OperationEquivalenceCache::getBlock(Block *block) {
  auto it = blocks.find(block);
  if (it != blocks.end())
    return *it->second;
  BlockEntry *entry = new BlockEntry();
  entry->count = block->getOperations().size();
  blocks[block] = entry;
  return *entry;
}

OperationEquivalenceCache::OperationEntry &
OperationEquivalenceCache::getOp(Operation *op) {
  auto it = ops.find(op);
  if (it != ops.end())
    return *it->second;
  OperationEntry *entry = new OperationEntry();
  entry->attrs.append(op->getRawDictionaryAttrs().getValue());
  if (op->getPropertiesStorageSize()) {
    op->getName().populateInherentAttrs(op, entry->attrs);
  }
  ops[op] = entry;
  return *entry;
}

template <typename Range, typename Pred>
bool compare_ranges(Range &&lhs, Range &&rhs, Pred pred) {
  auto lhsIt = lhs.begin();
  auto rhsIt = rhs.begin();
  auto lhsEnd = lhs.end();
  auto rhsEnd = rhs.end();
  while (lhsIt != lhsEnd && rhsIt != rhsEnd) {
    if (!pred(*lhsIt++, *rhsIt++))
      return false;
  }
  if ((lhsIt == lhsEnd) != (rhsIt == rhsEnd)) {
    // Block count mismatch. We do this here so that we avoid the O(n) scan
    // that would have been required to calculate the size above.
    return false;
  }
  return true;
}

static bool isStructurallyEquivalentTo(OperationEquivalenceCache &cache,
                                       Operation &lhs, Operation &rhs,
                                       IRMapping &parentMapping);

bool isStructurallyEquivalentTo(OperationEquivalenceCache &cache, Region &lhs,
                                Region &rhs) {
  auto mapping = cache.acquireMapping();
  return isStructurallyEquivalentTo(cache, lhs, rhs, *mapping);
}

bool isStructurallyEquivalentTo(Region &lhs, Region &rhs) {
  OperationEquivalenceCache cache(lhs.getContext());
  return isStructurallyEquivalentTo(cache, lhs, rhs);
}

bool isStructurallyEquivalentTo(Operation &lhs, Operation &rhs) {
  OperationEquivalenceCache cache(lhs.getContext());
  auto mapping = cache.acquireMapping();
  return isStructurallyEquivalentTo(cache, lhs, rhs, *mapping);
}

bool isStructurallyEquivalentTo(OperationEquivalenceCache &cache,
                                Operation &lhs, Operation &rhs) {
  auto mapping = cache.acquireMapping();
  return isStructurallyEquivalentTo(cache, lhs, rhs, *mapping);
}

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
bool isStructurallyEquivalentTo(OperationEquivalenceCache &cache, Region &lhs,
                                Region &rhs, IRMapping &mapping) {
  auto &lhsRegionEntry = cache.getRegion(&lhs);
  auto &rhsRegionEntry = cache.getRegion(&rhs);
  if (lhsRegionEntry.blocks.size() != rhsRegionEntry.blocks.size())
    return false;

  // Map blocks and their arguments so that we can compare their use by ops.
  for (auto [lhsBlock, rhsBlock] :
       llvm::zip_equal(lhsRegionEntry.blocks, rhsRegionEntry.blocks)) {
    if (lhsBlock->getNumArguments() != rhsBlock->getNumArguments())
      return false;
    for (auto [lhsArg, rhsArg] :
         llvm::zip_equal(lhsBlock->getArguments(), rhsBlock->getArguments())) {
      if (lhsArg.getType() != rhsArg.getType())
        return false;
      mapping.map(lhsArg, rhsArg);
    }
    mapping.map(lhsBlock, rhsBlock);
  }

  // Walk the blocks and populate a mapping. The blocks are stored in reverse
  // dominance order so that we always have the mappings available.
  for (auto [lhsBlock, rhsBlock] :
       llvm::zip_equal(lhsRegionEntry.blocks, rhsRegionEntry.blocks)) {
    const auto &lhsBlockEntry = cache.getBlock(lhsBlock);
    const auto &rhsBlockEntry = cache.getBlock(rhsBlock);
    if (lhsBlockEntry.count != rhsBlockEntry.count)
      return false;

    for (auto [lhsOp, rhsOp] : llvm::zip_equal(lhsBlock->getOperations(),
                                               rhsBlock->getOperations())) {
      if (!isStructurallyEquivalentTo(cache, lhsOp, rhsOp, mapping))
        return false;
    }
  }

  // Equivalent!
  return true;
}

static bool isStructurallyEquivalentTo(OperationEquivalenceCache &cache,
                                       Operation &lhs, Operation &rhs,
                                       IRMapping &parentMapping) {
  // Check operation metadata for early-exit opportunities.
  if (lhs.getName() != rhs.getName() ||
      lhs.getNumOperands() != rhs.getNumOperands() ||
      lhs.getNumResults() != rhs.getNumResults() ||
      lhs.getNumRegions() != rhs.getNumRegions() ||
      lhs.getNumSuccessors() != rhs.getNumSuccessors()) {
    return false;
  }

  auto &lhsEntry = cache.getOp(&lhs);
  auto &rhsEntry = cache.getOp(&rhs);

  // TODO(#3996): symbol mapping; for now allow them to differ unconditionally.
  if (lhsEntry.attrs.getAttrs().size() != rhsEntry.attrs.getAttrs().size())
    return false;
  for (auto [lhsAttr, rhsAttr] :
       llvm::zip_equal(lhsEntry.attrs, rhsEntry.attrs)) {
    if (!cache.isSymbolAttrName(lhsAttr.getName())) {
      if (lhsAttr != rhsAttr)
        return false;
    }
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
    if (lhs.hasTrait<OpTrait::IsIsolatedFromAbove>()) {
      auto scopedRegionMapping = cache.acquireMapping();
      if (!isStructurallyEquivalentTo(cache, lhsRegion, rhsRegion,
                                      *scopedRegionMapping)) {
        return false;
      }
    } else {
      IRMapping clonedParentMapping = parentMapping;
      if (!isStructurallyEquivalentTo(cache, lhsRegion, rhsRegion,
                                      clonedParentMapping)) {
        return false;
      }
    }
  }

  // Equivalent!
  return true;
}

} // namespace mlir::iree_compiler

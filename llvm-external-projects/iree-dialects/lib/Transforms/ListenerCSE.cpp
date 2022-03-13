//===-- ListenerCSE.cpp - Common subexpr elimination with a listener ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Transforms/ListenerCSE.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/RecyclingAllocator.h"
#include <deque>

using namespace mlir;

//===----------------------------------------------------------------------===//
// BEGIN copied from mlir/lib/Transforms/CSE.cpp
//===----------------------------------------------------------------------===//
namespace {
struct SimpleOperationInfo : public llvm::DenseMapInfo<Operation *> {
  static unsigned getHashValue(const Operation *opC) {
    return OperationEquivalence::computeHash(
        const_cast<Operation *>(opC),
        /*hashOperands=*/OperationEquivalence::directHashValue,
        /*hashResults=*/OperationEquivalence::ignoreHashValue,
        OperationEquivalence::IgnoreLocations);
  }
  static bool isEqual(const Operation *lhsC, const Operation *rhsC) {
    auto *lhs = const_cast<Operation *>(lhsC);
    auto *rhs = const_cast<Operation *>(rhsC);
    if (lhs == rhs)
      return true;
    if (lhs == getTombstoneKey() || lhs == getEmptyKey() ||
        rhs == getTombstoneKey() || rhs == getEmptyKey())
      return false;
    return OperationEquivalence::isEquivalentTo(
        const_cast<Operation *>(lhsC), const_cast<Operation *>(rhsC),
        /*mapOperands=*/OperationEquivalence::exactValueMatch,
        /*mapResults=*/OperationEquivalence::ignoreValueEquivalence,
        OperationEquivalence::IgnoreLocations);
  }
};
} // namespace

namespace {
/// Simple common sub-expression elimination.
struct CSE {
  /// Shared implementation of operation elimination and scoped map definitions.
  using AllocatorTy = llvm::RecyclingAllocator<
      llvm::BumpPtrAllocator,
      llvm::ScopedHashTableVal<Operation *, Operation *>>;
  using ScopedMapTy = llvm::ScopedHashTable<Operation *, Operation *,
                                            SimpleOperationInfo, AllocatorTy>;

//===----------------------------------------------------------------------===//
// END copied from mlir/lib/Transforms/CSE.cpp
//===----------------------------------------------------------------------===//
  CSE(DominanceInfo *domInfo, RewriteListener *listener)
      : domInfo(domInfo), listener(listener) {}
//===----------------------------------------------------------------------===//
// BEGIN copied from mlir/lib/Transforms/CSE.cpp
//===----------------------------------------------------------------------===//

  /// Represents a single entry in the depth first traversal of a CFG.
  struct CFGStackNode {
    CFGStackNode(ScopedMapTy &knownValues, DominanceInfoNode *node)
        : scope(knownValues), node(node), childIterator(node->begin()),
          processed(false) {}

    /// Scope for the known values.
    ScopedMapTy::ScopeTy scope;

    DominanceInfoNode *node;
    DominanceInfoNode::const_iterator childIterator;

    /// If this node has been fully processed yet or not.
    bool processed;
  };

  /// Attempt to eliminate a redundant operation. Returns success if the
  /// operation was marked for removal, failure otherwise.
  LogicalResult simplifyOperation(ScopedMapTy &knownValues, Operation *op,
                                  bool hasSSADominance);
  void simplifyBlock(ScopedMapTy &knownValues, Block *bb, bool hasSSADominance);
  void simplifyRegion(ScopedMapTy &knownValues, Region &region);
  /// Return the number of erased operations.
  unsigned simplify(Operation *rootOp);

private:
  /// Operations marked as dead and to be erased.
  std::vector<Operation *> opsToErase;

  /// The dominance info to use.
  DominanceInfo *domInfo;
//===----------------------------------------------------------------------===//
// END copied from mlir/lib/Transforms/CSE.cpp
//===----------------------------------------------------------------------===//
  /// An optional listener to notify of replaced or erased operations.
  RewriteListener *listener;
//===----------------------------------------------------------------------===//
// BEGIN copied from mlir/lib/Transforms/CSE.cpp
//===----------------------------------------------------------------------===//
};

} // namespace

/// Attempt to eliminate a redundant operation.
LogicalResult CSE::simplifyOperation(ScopedMapTy &knownValues, Operation *op,
                                     bool hasSSADominance) {
  // Don't simplify terminator operations.
  if (op->hasTrait<OpTrait::IsTerminator>())
    return failure();

  // If the operation is already trivially dead just add it to the erase list.
  if (isOpTriviallyDead(op)) {
    opsToErase.push_back(op);
    return success();
  }

  // Don't simplify operations with nested blocks. We don't currently model
  // equality comparisons correctly among other things. It is also unclear
  // whether we would want to CSE such operations.
  if (op->getNumRegions() != 0)
    return failure();

  // TODO: We currently only eliminate non side-effecting
  // operations.
  if (!MemoryEffectOpInterface::hasNoEffect(op))
    return failure();

  // Look for an existing definition for the operation.
  if (auto *existing = knownValues.lookup(op)) {

    // If we find one then replace all uses of the current operation with the
    // existing one and mark it for deletion. We can only replace an operand in
    // an operation if it has not been visited yet.
    if (hasSSADominance) {
      // If the region has SSA dominance, then we are guaranteed to have not
      // visited any use of the current operation.
//===----------------------------------------------------------------------===//
// END copied from mlir/lib/Transforms/CSE.cpp
//===----------------------------------------------------------------------===//
      if (listener)
        listener->notifyOperationReplaced(op, existing->getResults());
//===----------------------------------------------------------------------===//
// BEGIN copied from mlir/lib/Transforms/CSE.cpp
//===----------------------------------------------------------------------===//
      op->replaceAllUsesWith(existing);
      opsToErase.push_back(op);
    } else {
      // When the region does not have SSA dominance, we need to check if we
      // have visited a use before replacing any use.
      for (auto it : llvm::zip(op->getResults(), existing->getResults())) {
        std::get<0>(it).replaceUsesWithIf(
            std::get<1>(it), [&](OpOperand &operand) {
              return !knownValues.count(operand.getOwner());
            });
      }

      // There may be some remaining uses of the operation.
      if (op->use_empty())
        opsToErase.push_back(op);
    }

    // If the existing operation has an unknown location and the current
    // operation doesn't, then set the existing op's location to that of the
    // current op.
    if (existing->getLoc().isa<UnknownLoc>() &&
        !op->getLoc().isa<UnknownLoc>()) {
      existing->setLoc(op->getLoc());
    }

    return success();
  }

  // Otherwise, we add this operation to the known values map.
  knownValues.insert(op, op);
  return failure();
}

void CSE::simplifyBlock(ScopedMapTy &knownValues, Block *bb,
                        bool hasSSADominance) {
  for (auto &op : *bb) {
    // If the operation is simplified, we don't process any held regions.
    if (succeeded(simplifyOperation(knownValues, &op, hasSSADominance)))
      continue;

    // Most operations don't have regions, so fast path that case.
    if (op.getNumRegions() == 0)
      continue;

    // If this operation is isolated above, we can't process nested regions with
    // the given 'knownValues' map. This would cause the insertion of implicit
    // captures in explicit capture only regions.
    if (op.mightHaveTrait<OpTrait::IsIsolatedFromAbove>()) {
      ScopedMapTy nestedKnownValues;
      for (auto &region : op.getRegions())
        simplifyRegion(nestedKnownValues, region);
      continue;
    }

    // Otherwise, process nested regions normally.
    for (auto &region : op.getRegions())
      simplifyRegion(knownValues, region);
  }
}

void CSE::simplifyRegion(ScopedMapTy &knownValues, Region &region) {
  // If the region is empty there is nothing to do.
  if (region.empty())
    return;

  bool hasSSADominance = domInfo->hasSSADominance(&region);

  // If the region only contains one block, then simplify it directly.
  if (region.hasOneBlock()) {
    ScopedMapTy::ScopeTy scope(knownValues);
    simplifyBlock(knownValues, &region.front(), hasSSADominance);
    return;
  }

  // If the region does not have dominanceInfo, then skip it.
  // TODO: Regions without SSA dominance should define a different
  // traversal order which is appropriate and can be used here.
  if (!hasSSADominance)
    return;

  // Note, deque is being used here because there was significant performance
  // gains over vector when the container becomes very large due to the
  // specific access patterns. If/when these performance issues are no
  // longer a problem we can change this to vector. For more information see
  // the llvm mailing list discussion on this:
  // http://lists.llvm.org/pipermail/llvm-commits/Week-of-Mon-20120116/135228.html
  std::deque<std::unique_ptr<CFGStackNode>> stack;

  // Process the nodes of the dom tree for this region.
  stack.emplace_back(std::make_unique<CFGStackNode>(
      knownValues, domInfo->getRootNode(&region)));

  while (!stack.empty()) {
    auto &currentNode = stack.back();

    // Check to see if we need to process this node.
    if (!currentNode->processed) {
      currentNode->processed = true;
      simplifyBlock(knownValues, currentNode->node->getBlock(),
                    hasSSADominance);
    }

    // Otherwise, check to see if we need to process a child node.
    if (currentNode->childIterator != currentNode->node->end()) {
      auto *childNode = *(currentNode->childIterator++);
      stack.emplace_back(
          std::make_unique<CFGStackNode>(knownValues, childNode));
    } else {
      // Finally, if the node and all of its children have been processed
      // then we delete the node.
      stack.pop_back();
    }
  }
}

unsigned CSE::simplify(Operation *rootOp) {
  /// A scoped hash table of defining operations within a region.
  ScopedMapTy knownValues;

  for (auto &region : rootOp->getRegions())
    simplifyRegion(knownValues, region);

  /// Erase any operations that were marked as dead during simplification.
  for (auto *op : opsToErase) {
//===----------------------------------------------------------------------===//
// END copied from mlir/lib/Transforms/CSE.cpp
//===----------------------------------------------------------------------===//
    if (listener)
      listener->notifyOperationRemoved(op);
//===----------------------------------------------------------------------===//
// BEGIN copied from mlir/lib/Transforms/CSE.cpp
//===----------------------------------------------------------------------===//
    op->erase();
  }

  return opsToErase.size();
}

//===----------------------------------------------------------------------===//
// END copied from mlir/lib/Transforms/CSE.cpp
//===----------------------------------------------------------------------===//

/// Run CSE on the provided operation
LogicalResult mlir::eliminateCommonSubexpressions(Operation *op,
                                                  DominanceInfo *domInfo,
                                                  RewriteListener *listener) {
  assert(op->hasTrait<OpTrait::IsIsolatedFromAbove>() &&
         "can only do CSE on isolated-from-above ops");

  Optional<DominanceInfo> defaultDomInfo;
  if (domInfo == nullptr) {
    defaultDomInfo.emplace(op);
    domInfo = &*defaultDomInfo;
  }

  CSE cse(domInfo, listener);
  cse.simplify(op);
  return success();
}

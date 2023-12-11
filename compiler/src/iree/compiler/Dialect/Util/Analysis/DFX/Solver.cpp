// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/Analysis/DFX/Solver.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "iree-util-dfx"

namespace mlir::iree_compiler::DFX {

Solver::~Solver() {
  // Cleanup all elements; since we allocated them from the bump ptr allocator
  // they won't have their destructors called otherwise. Some elements may have
  // their own out-of-band allocations (like DenseMap) that would get leaked.
  for (auto it : elementMap) {
    it.second->~AbstractElement();
  }
}

LogicalResult Solver::run() {
  LLVM_DEBUG(llvm::dbgs() << "[Solver] identified and initialized "
                          << depGraph.syntheticRoot.deps.size()
                          << " abstract elements\n");

  // Now that all abstract elements are collected and initialized we start
  // the abstract analysis.
  phase = Phase::UPDATE;
  auto result = runTillFixpoint();
  phase = Phase::DONE;

  if (failed(result)) {
    LLVM_DEBUG(llvm::dbgs()
               << "[Solver] fixed point iteration failed to converge\n");
  }
  return result;
}

LogicalResult Solver::runTillFixpoint() {
  unsigned iterationCounter = 1;
  unsigned maxIterations = maxFixpointIterations.value_or(32);

  SmallVector<AbstractElement *, 32> changedElements;
  SetVector<AbstractElement *> worklist, invalidElements;
  worklist.insert(depGraph.syntheticRoot.begin(), depGraph.syntheticRoot.end());

  do {
    // Remember the size to determine newly added elements in this iteration.
    size_t elementCount = depGraph.syntheticRoot.deps.size();
    LLVM_DEBUG(llvm::dbgs() << "\n\n[Solver] iteration#: " << iterationCounter
                            << ", worklist size: " << worklist.size() << "\n");

    // For invalid elements we can fix dependent elements that have a required
    // dependence thereby folding long dependence chains in a single step
    // without the need to run updates.
    for (size_t i = 0; i < invalidElements.size(); ++i) {
      auto *invalidElement = invalidElements[i];

      // Check the dependences to fast track invalidation.
      LLVM_DEBUG(llvm::dbgs() << "[Solver] invalidElement: " << *invalidElement
                              << " has " << invalidElement->deps.size()
                              << " required & optional dependences\n");
      while (!invalidElement->deps.empty()) {
        const auto &dep = invalidElement->deps.back();
        invalidElement->deps.pop_back();
        auto *dependentElement = cast<AbstractElement>(dep.getPointer());
        if (dep.getInt() == static_cast<unsigned>(Resolution::OPTIONAL)) {
          worklist.insert(dependentElement);
          continue;
        }
        dependentElement->getState().indicatePessimisticFixpoint();
        assert(dependentElement->getState().isAtFixpoint() &&
               "expected fixpoint state");
        if (!dependentElement->getState().isValidState()) {
          invalidElements.insert(dependentElement);
        } else {
          changedElements.push_back(dependentElement);
        }
      }
    }

    // Add all abstract elements that are potentially dependent on one that
    // changed to the work list.
    for (auto *changedElement : changedElements) {
      while (!changedElement->deps.empty()) {
        worklist.insert(
            cast<AbstractElement>(changedElement->deps.back().getPointer()));
        changedElement->deps.pop_back();
      }
    }

    LLVM_DEBUG(llvm::dbgs()
               << "[Solver] iteration#: " << iterationCounter
               << ", worklist + dependent size: " << worklist.size() << "\n");

    // Reset the changed and invalid set.
    changedElements.clear();
    invalidElements.clear();

    // Update all abstract elements in the work list and record the ones that
    // changed.
    for (auto *element : worklist) {
      const auto &elementState = element->getState();
      if (!elementState.isAtFixpoint()) {
        if (updateElement(*element) == ChangeStatus::CHANGED) {
          changedElements.push_back(element);
        }
        LLVM_DEBUG(llvm::dbgs() << "\n");
      }

      // Use the invalidElements vector to propagate invalid states fast
      // transitively without requiring updates.
      if (!elementState.isValidState())
        invalidElements.insert(element);
    }

    // Add elements to the changed set if they have been created in the last
    // iteration.
    changedElements.append(depGraph.syntheticRoot.begin() + elementCount,
                           depGraph.syntheticRoot.end());

    // Reset the work list and repopulate with the changed abstract elements.
    // Note that dependent ones have already been added above.
    worklist.clear();
    worklist.insert(changedElements.begin(), changedElements.end());
  } while (!worklist.empty() && (iterationCounter++ < maxIterations));

  LLVM_DEBUG(llvm::dbgs() << "\n[Solver] fixpoint iteration done after: "
                          << iterationCounter << "/" << maxIterations
                          << " iterations\n");

  // Reset abstract elements not settled in a sound fixpoint by now. This
  // happens when we stopped the fixpoint iteration early. Note that only the
  // ones marked as "changed" *and* the ones transitively depending on them
  // need to be reverted to a pessimistic state. Others might not be in a
  // fixpoint state but we can use the optimistic results for them anyway.
  SmallPtrSet<AbstractElement *, 32> visitedElements;
  for (size_t i = 0; i < changedElements.size(); i++) {
    auto *changedElement = changedElements[i];
    if (!visitedElements.insert(changedElement).second)
      continue;

    auto &elementState = changedElement->getState();
    if (!elementState.isAtFixpoint()) {
      elementState.indicatePessimisticFixpoint();
    }

    while (!changedElement->deps.empty()) {
      changedElements.push_back(
          cast<AbstractElement>(changedElement->deps.back().getPointer()));
      changedElement->deps.pop_back();
    }
  }

  LLVM_DEBUG({
    if (!visitedElements.empty()) {
      llvm::dbgs() << "\n[Solver] finalized " << visitedElements.size()
                   << " abstract elements\n";
    }
  });

  return iterationCounter < maxIterations ? success() : failure();
}

ChangeStatus Solver::updateElement(AbstractElement &element) {
  assert(phase == Phase::UPDATE &&
         "can update element only in the update stage");

  // Use a new dependence vector for this update so we can possibly drop them
  // all if we reach a fixpoint.
  DependenceVector dependencies;
  dependenceStack.push_back(&dependencies);

  // Perform the abstract element update.
  auto &elementState = element.getState();
  ChangeStatus changeStatus = element.update(*this);

  if (dependencies.empty()) {
    // If the element did not query any non-fix information the state
    // will not change and we can indicate that right away.
    elementState.indicateOptimisticFixpoint();
  }
  if (!elementState.isAtFixpoint())
    rememberDependences();

  // Verify the stack is balanced by ensuring we pop the vector we pushed above.
  auto *poppedDependencies = dependenceStack.pop_back_val();
  (void)poppedDependencies;
  assert(poppedDependencies == &dependencies &&
         "inconsistent usage of the dependence stack");

  return changeStatus;
}

void Solver::recordDependence(const AbstractElement &fromElement,
                              const AbstractElement &toElement,
                              Resolution resolution) {
  if (resolution == Resolution::NONE)
    return;
  // If we are outside of an update, thus before the actual fixpoint iteration
  // started (= when we create elements), we do not track dependences because we
  // will put all elements into the initial worklist anyway.
  if (dependenceStack.empty())
    return;
  if (fromElement.getState().isAtFixpoint())
    return;
  dependenceStack.back()->push_back({&fromElement, &toElement, resolution});
}

void Solver::rememberDependences() {
  assert(!dependenceStack.empty() && "no dependences to remember");
  for (auto &depInfo : *dependenceStack.back()) {
    assert((depInfo.resolution == Resolution::REQUIRED ||
            depInfo.resolution == Resolution::OPTIONAL) &&
           "expected required or optional dependence (1 bit)");
    auto &dependentElements =
        const_cast<AbstractElement &>(*depInfo.fromElement).deps;
    dependentElements.push_back(
        AbstractElement::DepTy(const_cast<AbstractElement *>(depInfo.toElement),
                               static_cast<unsigned>(depInfo.resolution)));
  }
}

void Solver::print(llvm::raw_ostream &os) { depGraph.print(os); }

void Solver::dumpGraph() { depGraph.dumpGraph(); }

} // namespace mlir::iree_compiler::DFX

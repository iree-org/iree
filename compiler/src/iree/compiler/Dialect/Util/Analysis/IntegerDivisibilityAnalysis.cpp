// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/Analysis/IntegerDivisibilityAnalysis.h"

#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "iree-util-int-divisibility-analysis"

using llvm::dbgs;

namespace mlir::iree_compiler::IREE::Util {

void IntegerDivisibilityAnalysis::setToEntryState(
    IntegerDivisibilityLattice *lattice) {
  propagateIfChanged(lattice,
                     lattice->join(IntegerDivisibility::getMinDivisibility()));
}

LogicalResult IntegerDivisibilityAnalysis::visitOperation(
    Operation *op, ArrayRef<const IntegerDivisibilityLattice *> operands,
    ArrayRef<IntegerDivisibilityLattice *> results) {
  auto inferrable = dyn_cast<InferIntDivisibilityOpInterface>(op);
  if (!inferrable) {
    setAllToEntryStates(results);
    return success();
  }

  LLVM_DEBUG(dbgs() << "Inferring divisibility for " << *op << "\n");
  auto argDivs = llvm::map_to_vector(
      operands, [](const IntegerDivisibilityLattice *lattice) {
        return lattice->getValue();
      });
  auto joinCallback = [&](Value v, const IntegerDivisibility &newDiv) {
    auto result = dyn_cast<OpResult>(v);
    if (!result)
      return;
    assert(llvm::is_contained(op->getResults(), result));

    LLVM_DEBUG(dbgs() << "Inferred divisibility " << newDiv << "\n");
    IntegerDivisibilityLattice *lattice = results[result.getResultNumber()];
    IntegerDivisibility oldDiv = lattice->getValue();

    ChangeResult changed = lattice->join(newDiv);

    // Catch loop results with loop variant bounds and conservatively make
    // them [-inf, inf] so we don't circle around infinitely often (because
    // the dataflow analysis in MLIR doesn't attempt to work out trip counts
    // and often can't).
    bool isYieldedResult = llvm::any_of(v.getUsers(), [](Operation *op) {
      return op->hasTrait<OpTrait::IsTerminator>();
    });
    if (isYieldedResult && !oldDiv.isUninitialized() &&
        !(lattice->getValue() == oldDiv)) {
      LLVM_DEBUG(llvm::dbgs() << "Loop variant loop result detected\n");
      changed |= lattice->join(IntegerDivisibility::getMinDivisibility());
    }
    propagateIfChanged(lattice, changed);
  };

  inferrable.inferResultDivisibility(argDivs, joinCallback);
  return success();
}

} // namespace mlir::iree_compiler::IREE::Util

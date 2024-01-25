// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/SynchronizationAnalysis.h"
#include "iree/compiler/Codegen/Common/PassDetail.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "llvm/ADT/SetOperations.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Transforms.h"

using namespace mlir::dataflow;

namespace mlir::iree_compiler {

ChangeResult SetLattice::join(const AbstractDenseLattice &rhs) {
  auto &rhsSet = static_cast<const SetLattice &>(rhs);
  return join(rhsSet.getSet());
}

ChangeResult SetLattice::join(const DenseSet<Value> &rhs) {
  bool changed = llvm::set_union(set, rhs);
  return changed ? ChangeResult::Change : ChangeResult::NoChange;
}

void SetLattice::clear() { set.clear(); }

void SetLattice::print(raw_ostream &os) const {
  os << "[";
  llvm::interleaveComma(set, os,
                        [&](Value value) { os << value.getAsOpaquePointer(); });
  os << "]";
}

/// RAW Analysis:
/// after = (before - read(op)) U write(op)
///
/// WAR Analysis:
/// after = (before - write(op)) U read(op)
void SynchronizationAnalysis::visitOperation(Operation *op,
                                             const SetLattice &before,
                                             SetLattice *after) {
  DenseSet<Value> subtractSet;
  DenseSet<Value> unionSet;

  auto bufferizableOp = dyn_cast<bufferization::BufferizableOpInterface>(op);

  auto getReadWriteSet = [&](Operation *op, DenseSet<Value> &readSet,
                             DenseSet<Value> &writeSet) {
    if (!bufferizableOp) {
      return;
    }

    // TODO: Should we keep track of aliases? It is possible, and will give
    // better analysis.

    for (OpOperand &operand : op->getOpOperands()) {
      // TODO: Check encoding here.
      if (!isa<TensorType>(operand.get().getType())) {
        continue;
      }

      if (oneShotState.bufferizesToMemoryRead(operand)) {
        readSet.insert(operand.get());
      }

      if (oneShotState.bufferizesToMemoryWrite(operand)) {
        writeSet.insert(operand.get());
      }
    }
  };

  if (kind == SynchronizationKind::ReadAfterWrite) {
    getReadWriteSet(op, subtractSet, unionSet);
  } else {
    getReadWriteSet(op, unionSet, subtractSet);
  }

  op->dump();
  llvm::errs() << "subtractSet: ";
  for (auto value : subtractSet) {
    llvm::errs() << value.getAsOpaquePointer() << " ";
  }
  llvm::errs() << "\n";
  llvm::errs() << "unionSet: ";
  for (auto value : unionSet) {
    llvm::errs() << value.getAsOpaquePointer() << " ";
  }
  llvm::errs() << "\n";

  // after = (before - subtractSet) U unionSet
  DenseSet<Value> result = before.getSet();
  llvm::set_subtract(result, subtractSet);
  llvm::set_union(result, unionSet);
  // Propagate the change.
  propagateIfChanged(after, after->join(result));
}

static ArrayRef<BlockArgument> getEntryBlockArguements(Region &region) {
  return region.front().getArguments();
}

/// When transfering region, we simply add new values corressponding to the old
/// values, if the old value was in set.
void SynchronizationAnalysis::visitRegionBranchControlFlowTransfer(
    RegionBranchOpInterface branch, std::optional<unsigned> regionFrom,
    std::optional<unsigned> regionTo, const SetLattice &before,
    SetLattice *after) {

  DenseSet<Value> result = before.getSet();

  if (!regionFrom.has_value() && !regionTo.has_value()) {
    // after = before.
  } else if (!regionFrom.has_value()) {
    // Handle Entry from parent op to region.
    //
    // BeforeLattice
    // scf.for {
    //   AfterLattice
    //   ...
    //   scf.yield
    // }
    //
    // If any init_args value is in before, add the corresponding iter_args in
    // after.
    unsigned regionIn = regionTo.value();
    Region &region = branch->getRegion(regionIn);

    OperandRange initArgs =
        branch.getEntrySuccessorOperands(RegionBranchPoint::parent());
    ArrayRef<BlockArgument> iterArgs = getEntryBlockArguements(region);

    for (auto [initArg, iterArg] : llvm::zip(initArgs, iterArgs)) {
      if (result.contains(initArg)) {
        result.insert(iterArg);
      }
    }
  } else if (!regionTo.has_value()) {
    // Handle exit from region to parent op.
    //
    // scf.for {
    //   ...
    //   scf.yield
    //   BefortLattce
    // }
    // AfterLattice
    //
    // If any yielded value is in before, add the corresponding result in after.
    unsigned regionOut = regionFrom.value();
    Region &region = branch->getRegion(regionOut);

    OperandRange yieldedValues = branch.getEntrySuccessorOperands(region);
    ResultRange results = branch->getResults();

    // TODO: Maybe we can replace the values here.
    for (auto [yieldedValue, resultValue] : llvm::zip(yieldedValues, results)) {
      if (result.contains(yieldedValue)) {
        result.insert(resultValue);
      }
    }
  } else {
    // Handle transfer from one region to another region.
    Region &beforeRegion = branch->getRegion(regionFrom.value());
    Region &afterRegion = branch->getRegion(regionTo.value());

    // If any yielded value is in before, add the corresponding result in after.
    OperandRange yieldedValues = branch.getEntrySuccessorOperands(beforeRegion);
    ArrayRef<BlockArgument> iterArgs = getEntryBlockArguements(afterRegion);

    // TODO: Maybe we can replace the values here.
    for (auto [yieldedValue, iterArg] : llvm::zip(yieldedValues, iterArgs)) {
      if (result.contains(yieldedValue)) {
        result.insert(iterArg);
      }
    }
  }

  propagateIfChanged(after, after->join(result));
}

void SynchronizationAnalysis::setToEntryState(SetLattice *lattice) {
  lattice->clear();
}

} // namespace mlir::iree_compiler

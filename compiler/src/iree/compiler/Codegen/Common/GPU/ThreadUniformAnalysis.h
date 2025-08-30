// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_COMMON_GPU_THREADUNIFORMANALYSIS_H_
#define IREE_COMPILER_CODEGEN_COMMON_GPU_THREADUNIFORMANALYSIS_H_

#include "mlir/Analysis/DataFlow/SparseAnalysis.h"

namespace mlir::iree_compiler::dataflow {
//===----------------------------------------------------------------------===//
// ThreadUniform
//===----------------------------------------------------------------------===//
/// Class holding the state of the sparse lattice. There are three possible
/// states for the lattice:
/// `uninitialized`: the state is unknown.
/// `uniform`: the value was determined to be uniform across threads.
/// `dependent`: the value is thread dependent.
struct ThreadUniform {
  ThreadUniform() = default;
  /// Creates a dependent state.
  static ThreadUniform getDependent() { return ThreadUniform(dependent); }
  /// Creates a uniform state.
  static ThreadUniform getUniform() { return ThreadUniform(uniform); }
  /// Returns whether this state is uniform.
  bool isUniform() const { return state == uniform; }
  /// Compares two states.
  bool operator==(const ThreadUniform &other) const {
    return state == other.state;
  }
  /// Prints the state to a stream.
  void print(llvm::raw_ostream &s) const;
  /// Joins two states, where `dependent` is the top state of dataflow.
  static ThreadUniform join(const ThreadUniform &lhs, const ThreadUniform &rhs);

private:
  typedef enum { uninitialized, uniform, dependent } State;
  ThreadUniform(State state) : state(state) {}
  State state = uninitialized;
};

//===----------------------------------------------------------------------===//
// ThreadUniformLattice
//===----------------------------------------------------------------------===//
/// Class holding a lattice for the sparse analysis.
class ThreadUniformLattice : public mlir::dataflow::Lattice<ThreadUniform> {
public:
  using Lattice::Lattice;
};

//===----------------------------------------------------------------------===//
// ThreadUniformAnalysis
//===----------------------------------------------------------------------===//
/// The dataflow analysis computing whether a value is thread uniform or not.
class ThreadUniformAnalysis
    : public mlir::dataflow::SparseForwardDataFlowAnalysis<
          ThreadUniformLattice> {
public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

  /// Sets the lattice to a pessimistic state.
  void setToEntryState(ThreadUniformLattice *lattice) override {
    propagateIfChanged(lattice, lattice->join(ThreadUniform::getDependent()));
  }

  /// Visits an operation and determines whether it's uniform or not.
  LogicalResult
  visitOperation(Operation *op, ArrayRef<const ThreadUniformLattice *> operands,
                 ArrayRef<ThreadUniformLattice *> results) override;

  /// Handles the uniformity of control-flow arguments and results.
  void
  visitNonControlFlowArguments(Operation *op, const RegionSuccessor &successor,
                               ArrayRef<ThreadUniformLattice *> argLattices,
                               unsigned firstIndex) override;

  /// Override the default handling, this is necessary as control-flow with
  /// `scf.forall` / `scf.in_parallel` is broken, and the fix is large. TODO:
  /// Remove this once the ops have been fixed.
  void visitRegionSuccessors(
      ProgramPoint *point, RegionBranchOpInterface branch,
      RegionBranchPoint successor,
      ArrayRef<mlir::dataflow::AbstractSparseLattice *> lattices) override;
};
} // namespace mlir::iree_compiler::dataflow

MLIR_DECLARE_EXPLICIT_TYPE_ID(
    mlir::iree_compiler::dataflow::ThreadUniformLattice)

#endif // IREE_COMPILER_CODEGEN_COMMON_GPU_THREADUNIFORMANALYSIS_H_

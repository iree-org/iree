// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// This file a defines a pre-bufferization Read-After-Write
// and Write-After-Read analysis.
//
//===----------------------------------------------------------------------===//

#ifndef IREE_COMPILER_CODEGEN_COMMON_SYNCHRONIZATIONANALYSIS_H_
#define IREE_COMPILER_CODEGEN_COMMON_SYNCHRONIZATIONANALYSIS_H_

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/DenseAnalysis.h"

namespace mlir::iree_compiler {

class SetLattice final : public dataflow::AbstractDenseLattice {
public:
  using AbstractDenseLattice::AbstractDenseLattice;

  virtual ~SetLattice() = default;

  ChangeResult join(const AbstractDenseLattice &rhs) override;

  ChangeResult join(const DenseSet<Attribute> &rhsReads,
                    const DenseSet<Attribute> &rhsWrites);

  void clear();

  void print(raw_ostream &os) const override;

  const DenseSet<Attribute> getPendingReads() const { return pendingReads; }
  const DenseSet<Attribute> getPendingWrites() const { return pendingWrites; }

private:
  DenseSet<Attribute> pendingReads;
  DenseSet<Attribute> pendingWrites;
};

class SynchronizationAnalysis final
    : public dataflow::DenseForwardDataFlowAnalysis<SetLattice> {
public:
  SynchronizationAnalysis(DataFlowSolver &solver)
      : DenseForwardDataFlowAnalysis(solver) {}

  virtual ~SynchronizationAnalysis() = default;

  void visitOperation(Operation *op, const SetLattice &before,
                      SetLattice *after) override;

private:
  std::tuple<DenseSet<Attribute>, DenseSet<Attribute>>
  getReadAndWriteSet(Operation *op);

  void setToEntryState(SetLattice *lattice) override;
};

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_COMMON_SYNCHRONIZATIONANALYSIS_H_

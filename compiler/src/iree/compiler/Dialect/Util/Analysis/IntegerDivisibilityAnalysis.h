// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_UTIL_INTEGER_DIVISIBILITY_ANALYSIS_H_
#define IREE_COMPILER_DIALECT_UTIL_INTEGER_DIVISIBILITY_ANALYSIS_H_

#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"

#include <optional>

namespace mlir::iree_compiler::IREE::Util {

class IntegerDivisibilityLattice
    : public dataflow::Lattice<IntegerDivisibility> {
public:
  using Lattice::Lattice;
};

class IntegerDivisibilityAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<
          IntegerDivisibilityLattice> {
public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

  // At an entry point, set the lattice to the most pessimistic state,
  // indicating that no further reasoning can be done.
  void setToEntryState(IntegerDivisibilityLattice *lattice) override;

  // Visit an operation, invoking the transfer function.
  LogicalResult
  visitOperation(Operation *op,
                 ArrayRef<const IntegerDivisibilityLattice *> operands,
                 ArrayRef<IntegerDivisibilityLattice *> results) override;
};

} // namespace mlir::iree_compiler::IREE::Util

#endif // IREE_COMPILER_DIALECT_UTIL_INTEGER_DIVISIBILITY_ANALYSIS_H_

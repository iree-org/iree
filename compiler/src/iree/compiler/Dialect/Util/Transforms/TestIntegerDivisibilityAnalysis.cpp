// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/Analysis/IntegerDivisibilityAnalysis.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"

namespace mlir::iree_compiler::IREE::Util {

#define GEN_PASS_DEF_TESTINTEGERDIVISIBILITYANALYSISPASS
#include "iree/compiler/Dialect/Util/Transforms/Passes.h.inc"

namespace {

class TestIntegerDivisibilityAnalysisPass
    : public impl::TestIntegerDivisibilityAnalysisPassBase<
          TestIntegerDivisibilityAnalysisPass> {
public:
  void runOnOperation() override {
    Operation *rootOp = getOperation();
    MLIRContext *context = &getContext();

    // The pass is rooted on `iree_unregistered.test_int_divisibility` ops,
    // which are expected to have a single operand for which to annotate
    // divisibility information.
    SmallVector<std::pair<Operation *, Value>> queryOps;
    rootOp->walk([&](Operation *op) {
      if (op->getName().getStringRef() ==
              "iree_unregistered.test_int_divisibility" &&
          op->getNumOperands() == 1) {
        queryOps.emplace_back(op, op->getOperand(0));
      }
    });

    DataFlowSolver solver;
    // DeadCodeAnalysis is the base analysis that allows the solver to traverse
    // control flow. We include it to make the divisibility analysis more
    // powerful.
    solver.load<dataflow::DeadCodeAnalysis>();
    // SparseConstantPropagation is needed because DeadCodeAnalysis is too
    // conservative. It allows the analysis to call visitNonControlFlowArguments
    // and analyze arguments like loop induction variables.
    solver.load<dataflow::SparseConstantPropagation>();
    solver.load<IntegerDivisibilityAnalysis>();
    if (failed(solver.initializeAndRun(rootOp))) {
      return signalPassFailure();
    }

    for (auto &[op, value] : queryOps) {
      auto *lattice = solver.lookupState<IntegerDivisibilityLattice>(value);
      if (!lattice || lattice->getValue().isUninitialized()) {
        op->setAttr("divisibility", StringAttr::get(context, "uninitialized"));
        continue;
      }

      // Format for the divisibility information is "udiv = X, sdiv = Y".
      const auto &div = lattice->getValue().getValue();
      std::string result;
      llvm::raw_string_ostream os(result);
      os << "udiv = " << div.udiv() << ", sdiv = " << div.sdiv();
      op->setAttr("divisibility", StringAttr::get(context, os.str()));
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Util

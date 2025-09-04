// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Common/GPU/ThreadUniformAnalysis.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_TESTTHREADUNIFORMANALYSISPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {
struct TestThreadUniformAnalysisPass final
    : impl::TestThreadUniformAnalysisPassBase<TestThreadUniformAnalysisPass> {
  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();

    // Configure and run the dataflow analysis.
    DataFlowSolver solver;
    solver.load<mlir::dataflow::SparseConstantPropagation>();
    solver.load<mlir::dataflow::DeadCodeAnalysis>();
    solver.load<iree_compiler::dataflow::ThreadUniformAnalysis>();

    if (failed(solver.initializeAndRun(funcOp))) {
      return signalPassFailure();
    }

    Builder b(&getContext());
    constexpr llvm::StringLiteral kAttrName =
        "thread_uniform_analysis.is_uniform";

    funcOp.walk([&](Operation *op) {
      if (op->getNumResults() == 0) {
        return;
      }
      bool isUniform = llvm::all_of(op->getResults(), [&](Value v) {
        auto lattice =
            solver.lookupState<iree_compiler::dataflow::ThreadUniformLattice>(
                v);
        return lattice && lattice->getValue().isUniform();
      });
      op->setDiscardableAttr(kAttrName, b.getBoolAttr(isUniform));
    });
  }
};
} // namespace
} // namespace mlir::iree_compiler

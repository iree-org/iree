// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Indexing/Analysis/IndexRange.h"
#include "iree/compiler/Dialect/Indexing/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Indexing/Transforms/Passes.h"
#include "iree/compiler/Dialect/Indexing/Transforms/Utils.h"
#include "iree/compiler/Dialect/Util/Analysis/DFX/Solver.h"
#include "iree/compiler/Dialect/Util/Analysis/DFX/State.h"
#include "iree/compiler/Dialect/Util/Analysis/Explorer.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::Indexing {

namespace {

class TestIndexRangeAnalysisPass
    : public TestIndexRangeAnalysisBase<TestIndexRangeAnalysisPass> {
public:
  void runOnOperation() override {
    Explorer explorer(getOperation(), TraversalAction::SHALLOW);
    llvm::BumpPtrAllocator allocator;
    DFX::Solver solver(explorer, allocator);

    // Collect all probe points.
    SmallVector<std::pair<Operation *, const IndexRangeValueElement *>>
        indexQueryOps;
    SmallVector<std::pair<Operation *, const ShapedDimsRangeValueElement *>>
        shapedQueryOps;
    getOperation()->walk([&](Operation *op) {
      if (op->getName().getStringRef() == "iree_unregistered.test_intrange" &&
          op->getNumOperands() == 1) {
        Value operand = op->getOperands().front();
        const IndexRangeValueElement &element =
            solver.getOrCreateElementFor<IndexRangeValueElement>(
                Position::forValue(operand));
        indexQueryOps.emplace_back(op, &element);
      }
      if (op->getName().getStringRef() == "iree_unregistered.test_dimranges" &&
          op->getNumOperands() == 1) {
        Value operand = op->getOperands().front();
        const ShapedDimsRangeValueElement &element =
            solver.getOrCreateElementFor<ShapedDimsRangeValueElement>(
                Position::forValue(operand));
        shapedQueryOps.emplace_back(op, &element);
      }
    });

    // Solve.
    if (failed(solver.run())) {
      return signalPassFailure();
    }

    // Update.
    for (auto &it : indexQueryOps) {
      it.first->setAttr("analysis", StringAttr::get(&getContext(),
                                                    it.second->getAsStr(
                                                        solver.getAsmState())));
    }
    for (auto &it : shapedQueryOps) {
      it.first->setAttr("analysis", StringAttr::get(&getContext(),
                                                    it.second->getAsStr(
                                                        solver.getAsmState())));
    }

    // Drop assertion ops.
    RewritePatternSet patterns(&getContext());
    populateStripIndexingAssertionPatterns(&getContext(), patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<void>> createTestIndexRangeAnalysisPass() {
  return std::make_unique<TestIndexRangeAnalysisPass>();
}

} // namespace mlir::iree_compiler::IREE::Indexing

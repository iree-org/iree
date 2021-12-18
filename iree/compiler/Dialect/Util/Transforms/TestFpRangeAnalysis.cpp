// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/Analysis/Attributes/Range.h"
#include "iree/compiler/Dialect/Util/Analysis/DFX/Solver.h"
#include "iree/compiler/Dialect/Util/Analysis/DFX/State.h"
#include "iree/compiler/Dialect/Util/Analysis/Explorer.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Util {

namespace {

class TestFpRangeAnalysisPass
    : public PassWrapper<TestFpRangeAnalysisPass, OperationPass<void>> {
 public:
  StringRef getArgument() const override {
    return "iree-util-test-fp-range-analysis";
  }

  StringRef getDescription() const override {
    return "Tests fp range analysis by evaluating any "
           "'iree_unregistered.test_fprange' op and setting the results on an "
           "attribute";
  }

  void runOnOperation() override {
    Explorer explorer(getOperation(), TraversalAction::SHALLOW);
    llvm::BumpPtrAllocator allocator;
    DFX::Solver solver(explorer, allocator);

    // Collect all probe points.
    SmallVector<std::pair<Operation *, const FpRangeValueElement *>> queryOps;
    getOperation()->walk([&](Operation *op) {
      if (op->getName().getStringRef() == "iree_unregistered.test_fprange" &&
          op->getNumOperands() == 1) {
        Value operand = op->getOperands().front();
        const FpRangeValueElement &element =
            solver.getOrCreateElementFor<FpRangeValueElement>(
                Position::forValue(operand));
        queryOps.emplace_back(op, &element);
      }
    });

    // Solve.
    if (failed(solver.run())) {
      return signalPassFailure();
    }

    // Update.
    for (auto &it : queryOps) {
      it.first->setAttr("analysis",
                        StringAttr::get(it.second->getAsStr(), &getContext()));
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<void>> createTestFpRangeAnalsysis() {
  return std::make_unique<TestFpRangeAnalysisPass>();
}

static PassRegistration<TestFpRangeAnalysisPass> pass;

}  // namespace Util
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

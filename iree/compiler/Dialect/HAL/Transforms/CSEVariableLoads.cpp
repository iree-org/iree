// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

class CSEVariableLoadsPass
    : public PassWrapper<CSEVariableLoadsPass, OperationPass<FuncOp>> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<HALDialect>();
  }

  void runOnOperation() override {
    auto funcOp = getOperation();

    // Note: we assume that no IsolatedFromAbove regions are used with HAL
    // variables here. If there are any, they would need to be treated
    // independently of other regions.
    DenseMap<StringRef, SmallVector<VariableLoadOp, 4>>
        loadOpsGroupedByVariable;
    DenseSet<StringRef> storeOps;
    funcOp.walk([&](Operation *op) {
      if (auto loadOp = dyn_cast<VariableLoadOp>(op)) {
        loadOpsGroupedByVariable[loadOp.variable()].push_back(loadOp);
      } else if (auto storeOp = dyn_cast<VariableStoreOp>(op)) {
        storeOps.insert(storeOp.variable());
      } else if (auto storeIndirectOp = dyn_cast<VariableStoreIndirectOp>(op)) {
        // Not sure (without more analysis) which variable is being stored,
        // so give up.
        // TODO(scotttodd): handle indirect stores (trace to variable names)
        return;
      } else if (auto callOp = dyn_cast<CallOp>(op)) {
        // Not sure (without more analysis) which variables may be written to
        // as a result of the call, so give up.
        // TODO(scotttodd): handle calls (aggregate set of variables written
        //   by the function or any function that may be reached under it)
        return;
      }
    });

    Block *entryBlock = &funcOp.body().front();
    auto entryBlockBuilder = OpBuilder::atBlockBegin(entryBlock);

    for (auto loadOpsGroup : loadOpsGroupedByVariable) {
      StringRef variableName = loadOpsGroup.first;
      auto loadOps = loadOpsGroup.second;

      // Bail if the variable was written to at all.
      // We could do more complex analysis here but that's better handled
      // upstream with real CSE and side effect modeling :)
      if (storeOps.count(variableName)) continue;

      if (loadOps.size() == 1) continue;

      // Replace all load ops with a new load op in the entry block.
      auto clonedLoadOp = entryBlockBuilder.clone(*loadOps[0]);
      for (int i = 0; i < loadOps.size(); ++i) {
        loadOps[i].replaceAllUsesWith(clonedLoadOp);
        loadOps[i].erase();
      }
    }
  }
};

std::unique_ptr<OperationPass<FuncOp>> createCSEVariableLoadsPass() {
  return std::make_unique<CSEVariableLoadsPass>();
}

static PassRegistration<CSEVariableLoadsPass> pass(
    "iree-hal-cse-variable-loads",
    "Eliminates redundant 'hal.variable.load' ops within functions with no "
    "'hal.variable.store' ops");

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

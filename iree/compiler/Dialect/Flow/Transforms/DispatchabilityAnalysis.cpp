// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <utility>

#include "iree/compiler/Dialect/Flow/Analysis/Dispatchability.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

class DispatchabilityAnalysisPass
    : public PassWrapper<DispatchabilityAnalysisPass, OperationPass<ModuleOp>> {
 public:
  DispatchabilityAnalysisPass() = default;

  void runOnOperation() override {
    // Force creation (or caching) of dispatchability information.
    auto &dispatchability = getAnalysis<Dispatchability>();
    markAllAnalysesPreserved();

    // Build the dispatchable func table.
    if (dispatchableFuncOps_) {
      dispatchability.walkDispatchableOps([&](FuncOp funcOp) {
        (*dispatchableFuncOps_)[funcOp.getName()] = funcOp;
      });
    }
  }

  std::shared_ptr<llvm::StringMap<FuncOp>> dispatchableFuncOps_;
};

std::unique_ptr<OperationPass<ModuleOp>> createDispatchabilityAnalysisPass() {
  return std::make_unique<DispatchabilityAnalysisPass>();
}

static PassRegistration<DispatchabilityAnalysisPass> pass(
    "iree-flow-dispatchability-analysis",
    "Analyzes functions to determine their dispatchability");

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

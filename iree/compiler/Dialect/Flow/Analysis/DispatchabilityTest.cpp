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

#include "iree/compiler/Dialect/Flow/Analysis/Dispatchability.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir {
namespace iree_compiler {

class DispatchabilityTestPass
    : public PassWrapper<DispatchabilityTestPass, OperationPass<ModuleOp>> {
 public:
  void runOnOperation() override {
    if (failed(Dispatchability::annotateIR(getOperation()))) {
      signalPassFailure();
    }
  }
};

namespace IREE {
namespace Flow {
std::unique_ptr<OperationPass<ModuleOp>> createDispatchabilityTestPass() {
  return std::make_unique<DispatchabilityTestPass>();
}
}  // namespace Flow
}  // namespace IREE

static PassRegistration<DispatchabilityTestPass> pass(
    "test-iree-flow-dispatchability",
    "Test pass used for dispatchability analysis");

}  // namespace iree_compiler
}  // namespace mlir

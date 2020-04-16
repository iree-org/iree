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

#include "iree/compiler/Dialect/VM/Analysis/RegisterAllocation.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir {
namespace iree_compiler {

class RegisterAllocationTestPass
    : public PassWrapper<RegisterAllocationTestPass,
                         OperationPass<IREE::VM::FuncOp>> {
 public:
  void runOnOperation() override {
    if (failed(RegisterAllocation::annotateIR(getOperation()))) {
      signalPassFailure();
    }
  }
};

namespace IREE {
namespace VM {
std::unique_ptr<OperationPass<IREE::VM::FuncOp>>
createRegisterAllocationTestPass() {
  return std::make_unique<RegisterAllocationTestPass>();
}
}  // namespace VM
}  // namespace IREE

static PassRegistration<RegisterAllocationTestPass> pass(
    "test-iree-vm-register-allocation",
    "Test pass used for register allocation");

}  // namespace iree_compiler
}  // namespace mlir

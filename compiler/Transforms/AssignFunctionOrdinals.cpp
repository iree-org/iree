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

#include "mlir/IR/Attributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Utils.h"

namespace mlir {
namespace iree_compiler {

class AssignFunctionOrdinalsPass
    : public ModulePass<AssignFunctionOrdinalsPass> {
 public:
  void runOnModule() override {
    Builder builder(getModule());
    int nextFunctionOrdinal = 0;
    for (auto funcOp : getModule().getOps<FuncOp>()) {
      funcOp.setAttr("iree.ordinal",
                     builder.getI32IntegerAttr(nextFunctionOrdinal++));
    }
  }
};

std::unique_ptr<OpPassBase<ModuleOp>> createAssignFunctionOrdinalsPass() {
  return std::make_unique<AssignFunctionOrdinalsPass>();
}

static PassRegistration<AssignFunctionOrdinalsPass> pass(
    "iree-assign-function-ordinals", "Assigns all functions ordinals");

}  // namespace iree_compiler
}  // namespace mlir

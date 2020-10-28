// Copyright 2020 Google LLC
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

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

class FindDuplicateExecutablesPass
    : public PassWrapper<FindDuplicateExecutablesPass,
                         OperationPass<ExecutableOp>> {
 public:
  void runOnOperation() override {
    auto executableOp = getOperation();
    auto parentModuleOp = dyn_cast<ModuleOp>(executableOp.getParentOp());
    auto siblingExecutableOps =
        llvm::to_vector<8>(parentModuleOp.getOps<ExecutableOp>());

    auto hashAttr = executableOp.getAttrOfType<IntegerAttr>("func_hash");

    // Iterate in order, stopping when this op itself is reached, or the
    // first duplicate sibling is encountered.
    //
    // For example: if i==3, i==4, and i==5 are duplicates we want to *keep*
    // i==3 and replace uses of i==4 and i==5 with i==3. There's no need to
    // check past an op itself since those comparisons will happen in other
    // pass iterations.
    for (auto siblingExecutableOp : siblingExecutableOps) {
      if (executableOp == siblingExecutableOp) {
        break;
      }

      auto siblingHashAttr =
          siblingExecutableOp.getAttrOfType<IntegerAttr>("func_hash");
      if (hashAttr.getValue() == siblingHashAttr.getValue()) {
        // Assume that hash collisions never happen.
        // TODO(scotttodd): More reliable, but still efficient, comparison?
        //   Or a second check after the hash test? With some caching?
        Builder builder(executableOp.getContext());
        executableOp.setAttr(
            "duplicate_of",
            builder.getSymbolRefAttr(siblingExecutableOp.sym_name()));
        break;
      }
    }
  }
};

std::unique_ptr<OperationPass<ExecutableOp>>
createFindDuplicateExecutablesPass() {
  return std::make_unique<FindDuplicateExecutablesPass>();
}

static PassRegistration<FindDuplicateExecutablesPass> pass(
    "iree-flow-find-duplicate-executables",
    "Finds which executables are duplicates of earlier executables");

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

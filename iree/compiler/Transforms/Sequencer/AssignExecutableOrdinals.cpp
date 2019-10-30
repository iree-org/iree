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

#include "iree/compiler/IR/StructureOps.h"
#include "iree/compiler/Utils/OpUtils.h"
#include "third_party/llvm/llvm/include/llvm/ADT/DenseMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Utils.h"

namespace mlir {
namespace iree_compiler {

class AssignExecutableOrdinalsPass
    : public ModulePass<AssignExecutableOrdinalsPass> {
 public:
  void runOnModule() override {
    Builder builder(getModule());
    int nextExecutableOrdinal = 0;
    for (auto multiArchExecutableOp :
         getModule().getOps<IREE::MultiArchExecutableOp>()) {
      multiArchExecutableOp.setAttr(
          "iree.ordinal", builder.getI32IntegerAttr(nextExecutableOrdinal++));

      // We'll scan for all entry points in the first executable. Then on all
      // other executables we can reuse the ordinals (ensuring that iteration
      // order does not matter).
      llvm::DenseMap<StringRef, FuncOp> entryPointMap;
      for (auto executableOp :
           multiArchExecutableOp.getBlock().getOps<IREE::ExecutableOp>()) {
        executableOp.setAttr("iree.ordinal",
                             multiArchExecutableOp.getAttr("iree.ordinal"));
        int nextEntryPointOrdinal = 0;
        for (auto funcOp : executableOp.getInnerModule().getOps<FuncOp>()) {
          if (!funcOp.getAttr("iree.executable.export")) continue;
          auto it = entryPointMap.find(funcOp.getName());
          if (it == entryPointMap.end()) {
            funcOp.setAttr("iree.ordinal",
                           builder.getI32IntegerAttr(nextEntryPointOrdinal++));
            entryPointMap.insert({funcOp.getName(), funcOp});
          } else {
            funcOp.setAttr("iree.ordinal", it->second.getAttr("iree.ordinal"));
          }
        }
      }
    }
  }
};

std::unique_ptr<OpPassBase<ModuleOp>> createAssignExecutableOrdinalsPass() {
  return std::make_unique<AssignExecutableOrdinalsPass>();
}

static PassRegistration<AssignExecutableOrdinalsPass> pass(
    "iree-assign-executable-ordinals",
    "Assigns executable and entry point ordinals");

}  // namespace iree_compiler
}  // namespace mlir

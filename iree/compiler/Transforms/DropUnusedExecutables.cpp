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

#include "iree/compiler/IR/Sequencer/HLOps.h"
#include "iree/compiler/IR/StructureOps.h"
#include "third_party/llvm/llvm/include/llvm/ADT/SetVector.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Utils.h"

namespace mlir {
namespace iree_compiler {

// Drops all executables in a module that are not used by any dispatch
// sequencer op.
class DropUnusedExecutablesPass : public ModulePass<DropUnusedExecutablesPass> {
 public:
  void runOnModule() override {
    DenseSet<StringRef> usedExecutableNames;
    for (auto funcOp : getModule().getOps<FuncOp>()) {
      funcOp.walk([&](IREESeq::HL::DispatchOp op) {
        usedExecutableNames.insert(op.getExecutable());
      });
    }
    DenseSet<Operation *> deadExecutables;
    for (auto executableOp :
         getModule().getOps<IREE::MultiArchExecutableOp>()) {
      if (usedExecutableNames.count(executableOp.getName()) == 0) {
        deadExecutables.insert(executableOp);
      }
    }
    for (auto executableOp : deadExecutables) {
      executableOp->erase();
    }
  }
};

std::unique_ptr<OpPassBase<ModuleOp>> createDropUnusedExecutablesPass() {
  return std::make_unique<DropUnusedExecutablesPass>();  // NOLINT
}

static PassRegistration<DropUnusedExecutablesPass> executableFunctionsPass(
    "iree-drop-unused-executables",
    "Drop all executables not reachable from a dispatch/reduce op.");

}  // namespace iree_compiler
}  // namespace mlir

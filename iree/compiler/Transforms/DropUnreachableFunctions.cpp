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

#include "iree/compiler/Utils/ModuleUtils.h"
#include "third_party/llvm/llvm/include/llvm/ADT/SetVector.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Utils.h"

namespace mlir {
namespace iree_compiler {

// Drops all functions in a module that are not reachable by functions with the
// "iree.module.export" attribute.
class DropUnreachableModuleFunctionsPass
    : public ModulePass<DropUnreachableModuleFunctionsPass> {
 public:
  void runOnModule() override {
    dropUnusedFunctions(getModule(), {"iree.module.export"});
  }
};

// Drops all functions in a module that are not reachable by functions with the
// "iree.executable.export" attribute.
class DropUnreachableExecutableFunctionsPass
    : public ModulePass<DropUnreachableExecutableFunctionsPass> {
 public:
  void runOnModule() override {
    dropUnusedFunctions(getModule(), {"iree.executable.export"});
  }
};

std::unique_ptr<OpPassBase<ModuleOp>>
createDropUnreachableModuleFunctionsPass() {
  return std::make_unique<DropUnreachableModuleFunctionsPass>();
}

std::unique_ptr<OpPassBase<ModuleOp>>
createDropUnreachableExecutableFunctionsPass() {
  return std::make_unique<DropUnreachableExecutableFunctionsPass>();
}

static PassRegistration<DropUnreachableModuleFunctionsPass> moduleFunctionsPass(
    "iree-drop-unreachable-module-functions",
    "Drop all functions not reachable from an exported function");

static PassRegistration<DropUnreachableExecutableFunctionsPass>
    executableFunctionsPass(
        "iree-drop-unreachable-executable-functions",
        "Drop all functions not reachable from an exported function");

}  // namespace iree_compiler
}  // namespace mlir

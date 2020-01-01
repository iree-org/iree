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

#include "llvm/ADT/SetVector.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Function.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Utils.h"

namespace mlir {
namespace iree_compiler {

namespace {

// Finds a list of functions with the given |attrName| and adds them to |funcs|.
void findFunctionsWithAttr(ModuleOp module, const char *attrName,
                           llvm::SetVector<FuncOp> &funcs) {
  for (auto func : module.getOps<FuncOp>()) {
    if (func.getAttr(attrName)) {
      funcs.insert(func);
    }
  }
}

// Inserts functions reachable directly from |func| to |usedFuncs|.
void insertUsedFunctions(ModuleOp module, FuncOp func,
                         DenseSet<FuncOp> *usedFuncs,
                         std::vector<FuncOp> *toSearch) {
  auto onCalledFunction = [&](StringRef calleeName) {
    auto calleeFunc = module.lookupSymbol<FuncOp>(calleeName);
    if (usedFuncs->insert(calleeFunc).second) {
      // New function found! Add to queue for searching.
      toSearch->push_back(calleeFunc);
    }
  };
  for (auto &block : func) {
    for (auto &op : block) {
      // TODO(benvanik): replace with iree_hl.call check.
      if (auto calleeAttr = op.getAttr("callee")) {
        onCalledFunction(calleeAttr.cast<FlatSymbolRefAttr>().getValue());
      }
    }
  }
}

// Returns a set containing the names of all functions used by the given
// |rootFuncs| list.
DenseSet<FuncOp> findUsedFunctions(ModuleOp module,
                                   ArrayRef<FuncOp> rootFuncs) {
  // Breadth-first search.
  DenseSet<FuncOp> usedFuncs;
  usedFuncs.insert(rootFuncs.begin(), rootFuncs.end());
  std::vector<FuncOp> toSearch = {rootFuncs.begin(), rootFuncs.end()};
  while (!toSearch.empty()) {
    auto func = toSearch.back();
    toSearch.pop_back();
    insertUsedFunctions(module, func, &usedFuncs, &toSearch);
  }
  return usedFuncs;
}

// Drops all functions unreachable from functions with at least one of the
// specified |keepAttrs| on them.
void dropUnusedFunctions(ModuleOp module, ArrayRef<const char *> keepAttrs) {
  // Find all of the exported functions we'll treat as roots.
  llvm::SetVector<FuncOp> rootFuncs;
  for (auto keepAttr : keepAttrs) {
    findFunctionsWithAttr(module, keepAttr, rootFuncs);
  }

  // Find the full set of all used functions reachable from the given rootFuncs.
  // This set will contain the rootFuncs.
  auto usedFuncs = findUsedFunctions(module, rootFuncs.getArrayRef());

  // Drop all unused functions.
  std::vector<FuncOp> deadFuncs;
  for (auto func : module.getOps<FuncOp>()) {
    if (!llvm::is_contained(usedFuncs, func)) {
      deadFuncs.push_back(func);
    }
  }
  for (auto func : deadFuncs) {
    func.erase();
  }
}

}  // namespace

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
createDropUnreachableExecutableFunctionsPass() {
  return std::make_unique<DropUnreachableExecutableFunctionsPass>();
}

static PassRegistration<DropUnreachableExecutableFunctionsPass>
    executableFunctionsPass(
        "iree-drop-unreachable-executable-functions",
        "Drop all functions not reachable from an exported function");

}  // namespace iree_compiler
}  // namespace mlir

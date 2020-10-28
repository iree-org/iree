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

namespace {

// Replaces each usage of an entry point with its original symbol name with a
// new symbol name.
void replaceEntryPointUses(mlir::ModuleOp moduleOp,
                           const DenseMap<Attribute, Attribute> &replacements) {
  for (auto funcOp : moduleOp.getOps<mlir::FuncOp>()) {
    funcOp.walk([&](DispatchOp dispatchOp) {
      auto it = replacements.find(dispatchOp.entry_point());
      if (it != replacements.end()) {
        dispatchOp.entry_pointAttr(it->second.cast<SymbolRefAttr>());
      }
    });
  }
}

}  // namespace

class DeduplicateExecutablesPass
    : public PassWrapper<DeduplicateExecutablesPass, OperationPass<ModuleOp>> {
 public:
  explicit DeduplicateExecutablesPass() {}
  DeduplicateExecutablesPass(const DeduplicateExecutablesPass &pass) {}

  void runOnOperation() override {
    auto moduleOp = getOperation();
    auto executableOps = llvm::to_vector<8>(moduleOp.getOps<ExecutableOp>());
    auto builder = OpBuilder::atBlockBegin(moduleOp.getBody());

    SmallVector<ExecutableOp, 3> duplicateExecutableOps;
    DenseMap<Attribute, Attribute> entryPointRefReplacements;

    for (auto executableOp : executableOps) {
      auto duplicateOpSym =
          executableOp.getAttrOfType<SymbolRefAttr>("duplicate_of");
      if (!duplicateOpSym) {
        continue;
      }
      auto duplicateOp =
          dyn_cast<ExecutableOp>(SymbolTable::lookupNearestSymbolFrom(
              moduleOp, duplicateOpSym.getLeafReference()));

      auto oldSymbolRefAttr = builder.getSymbolRefAttr(
          executableOp.getName(),
          {builder.getSymbolRefAttr(
              executableOp.getDispatchEntryOp().sym_name())});
      auto newSymbolRefAttr = builder.getSymbolRefAttr(
          duplicateOp.getName(),
          {builder.getSymbolRefAttr(
              duplicateOp.getDispatchEntryOp().sym_name())});
      entryPointRefReplacements[oldSymbolRefAttr] = newSymbolRefAttr;
      duplicateExecutableOps.push_back(executableOp);
    }

    totalExecutables = executableOps.size();
    executablesDeduplicated = duplicateExecutableOps.size();
    remainingExecutables = totalExecutables - executablesDeduplicated;

    replaceEntryPointUses(moduleOp, entryPointRefReplacements);
    for (auto executableOp : duplicateExecutableOps) {
      executableOp.erase();
    }

    // Clean up our working attributes.
    for (auto executableOp : moduleOp.getOps<ExecutableOp>()) {
      executableOp.removeAttr("func_hash");
      executableOp.removeAttr("duplicate_of");
    }

    // TODO(scotttodd): rewrite executable indices, filling in gaps?
  }

 private:
  Statistic totalExecutables{
      this, "total executable(s)",
      "Number of flow.executable ops before deduplication"};
  Statistic executablesDeduplicated{
      this, "duplicate executable(s)",
      "Number of flow.executable ops removed as duplicates"};
  Statistic remainingExecutables{
      this, "unique executable(s)",
      "Number of flow.executable ops remaining after deduplication"};
};

std::unique_ptr<OperationPass<ModuleOp>> createDeduplicateExecutablesPass() {
  return std::make_unique<DeduplicateExecutablesPass>();
}

static PassRegistration<DeduplicateExecutablesPass> pass(
    "iree-flow-dedupliclate-executables",
    "Deduplicates executables that are identical");

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

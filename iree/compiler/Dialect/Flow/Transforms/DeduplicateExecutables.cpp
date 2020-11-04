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

#include "iree/compiler/Dialect/Flow/Analysis/ExecutableHashAnalysis.h"
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

bool areRegionsEquivalent(Region *lhs, Region *rhs) {
  if (lhs->getBlocks().size() != rhs->getBlocks().size()) {
    return false;
  }

  auto blockPairs = llvm::zip(lhs->getBlocks(), rhs->getBlocks());
  for (auto blockPair : blockPairs) {
    auto &lhsBlock = std::get<0>(blockPair);
    auto &rhsBlock = std::get<1>(blockPair);
    if (lhsBlock.getOperations().size() != rhsBlock.getOperations().size()) {
      return false;
    }

    auto opPairs =
        llvm::zip(lhsBlock.getOperations(), rhsBlock.getOperations());
    for (auto opPair : opPairs) {
      auto &lhsOp = std::get<0>(opPair);
      auto &rhsOp = std::get<1>(opPair);
      if (!OperationEquivalence::isEquivalentTo(
              &lhsOp, &rhsOp, OperationEquivalence::IgnoreOperands)) {
        return false;
      }

      if (lhsOp.getNumRegions() != rhsOp.getNumRegions()) {
        return false;
      }
      auto lhsRegions = lhsOp.getRegions();
      auto rhsRegions = rhsOp.getRegions();
      for (int i = 0; i < lhsRegions.size(); ++i) {
        if (!areRegionsEquivalent(&lhsRegions[i], &rhsRegions[i])) {
          return false;
        }
      }
    }
  }

  return true;
}

bool areExecutablesEquivalent(ExecutableOp lhs, ExecutableOp rhs) {
  auto lhsModule = lhs.getInnerModule();
  auto rhsModule = rhs.getInnerModule();
  auto lhsFuncOps = llvm::to_vector<1>(lhsModule.getOps<FuncOp>());
  auto rhsFuncOps = llvm::to_vector<1>(rhsModule.getOps<FuncOp>());
  if (lhsFuncOps.size() != rhsFuncOps.size()) {
    return false;
  }

  for (int i = 0; i < lhsFuncOps.size(); ++i) {
    auto lhsRegion = lhsFuncOps[i].getCallableRegion();
    auto rhsRegion = rhsFuncOps[i].getCallableRegion();
    return areRegionsEquivalent(lhsRegion, rhsRegion);
  }

  return true;
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

    // For each executable, find the first executable which it is equivalent to.
    for (int i = executableOps.size() - 1; i >= 0; --i) {
      auto executableOp = executableOps[i];
      // auto hashAnalysis =
      //     getChildAnalysis<ExecutableHashAnalysis>(executableOp);

      for (int j = 0; j < i; ++j) {
        auto comparisonExecutableOp = executableOps[j];
        // auto comparisonHashAnalysis =
        //     getChildAnalysis<ExecutableHashAnalysis>(comparisonExecutableOp);

        // Fast hash comparison first, then full equivalence check.
        if (/* hashAnalysis.hashCode != comparisonHashAnalysis.hashCode || */
            !areExecutablesEquivalent(executableOp, comparisonExecutableOp)) {
          continue;
        }

        // Record entry point reference replacements.
        auto dispatchEntryOps = llvm::to_vector<1>(
            executableOp.getBlock().getOps<DispatchEntryOp>());
        auto comparisonDispatchEntryOps = llvm::to_vector<1>(
            comparisonExecutableOp.getBlock().getOps<DispatchEntryOp>());
        assert(dispatchEntryOps.size() == comparisonDispatchEntryOps.size());
        for (int k = 0; k < dispatchEntryOps.size(); ++k) {
          auto oldSymbolRefAttr = builder.getSymbolRefAttr(
              executableOp.getName(),
              {builder.getSymbolRefAttr(dispatchEntryOps[k].sym_name())});
          auto newSymbolRefAttr = builder.getSymbolRefAttr(
              comparisonExecutableOp.getName(),
              {builder.getSymbolRefAttr(
                  comparisonDispatchEntryOps[k].sym_name())});
          entryPointRefReplacements[oldSymbolRefAttr] = newSymbolRefAttr;
        }

        duplicateExecutableOps.push_back(executableOp);
        break;
      }
    }

    totalExecutables = executableOps.size();
    executablesDeduplicated = duplicateExecutableOps.size();
    remainingExecutables = totalExecutables - executablesDeduplicated;

    replaceEntryPointUses(moduleOp, entryPointRefReplacements);
    for (auto executableOp : duplicateExecutableOps) {
      executableOp.erase();
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

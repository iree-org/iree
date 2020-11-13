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
void replaceEntryPointUses(
    mlir::ModuleOp moduleOp,
    const DenseMap<Attribute, SymbolRefAttr> &replacements) {
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

  for (auto blockPair : llvm::zip(lhs->getBlocks(), rhs->getBlocks())) {
    auto &lhsBlock = std::get<0>(blockPair);
    auto &rhsBlock = std::get<1>(blockPair);
    // Warning: .size() is linear time.
    // We could instead iterate through both lists of operations explicitly,
    // stopping when operations are not equivalent, OR either list runs out of
    // operations early.
    if (lhsBlock.getOperations().size() != rhsBlock.getOperations().size()) {
      return false;
    }

    for (auto opPair :
         llvm::zip(lhsBlock.getOperations(), rhsBlock.getOperations())) {
      auto &lhsOp = std::get<0>(opPair);
      auto &rhsOp = std::get<1>(opPair);
      if (!OperationEquivalence::isEquivalentTo(
              &lhsOp, &rhsOp, OperationEquivalence::IgnoreOperands)) {
        return false;
      }

      // We want to check the operand _types_, but don't care if the actual
      // operand references differ (as they live in separate modules anyway).
      if (!std::equal(lhsOp.operand_type_begin(), lhsOp.operand_type_end(),
                      rhsOp.operand_type_begin())) {
        return false;
      }

      // If the operations have regions, recurse into them (depth-first).
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

  // TODO(scotttodd): Generalize: replace special cases with just calling
  //   areRegionsEquivalent() on module.getBodyRegion(). We want to ignore
  //   operation names and sym_name attrs, which
  //   OperationEquivalence::isEquivalentTo() does not support [yet].

  // Must have the same number of entry point ops, with the same attributes.
  // Entry point op symbol names are expected to differ, that won't affect
  // equivalence.
  auto lhsEntryOps = llvm::to_vector<1>(lhsModule.getOps<DispatchEntryOp>());
  auto rhsEntryOps = llvm::to_vector<1>(rhsModule.getOps<DispatchEntryOp>());
  if (lhsEntryOps.size() != rhsEntryOps.size()) {
    return false;
  }
  for (int i = 0; i < lhsEntryOps.size(); ++i) {
    if (lhsEntryOps[i].getAttrs() != rhsEntryOps[i].getAttrs()) {
      return false;
    }
  }

  // Must have the same number of functions, with each listed in the same order
  // and with equivalent regions inside.
  auto lhsFuncOps = llvm::to_vector<1>(lhsModule.getOps<FuncOp>());
  auto rhsFuncOps = llvm::to_vector<1>(rhsModule.getOps<FuncOp>());
  if (lhsFuncOps.size() != rhsFuncOps.size()) {
    return false;
  }
  for (int i = 0; i < lhsFuncOps.size(); ++i) {
    auto lhsRegion = lhsFuncOps[i].getCallableRegion();
    auto rhsRegion = rhsFuncOps[i].getCallableRegion();
    if (!areRegionsEquivalent(lhsRegion, rhsRegion)) {
      return false;
    }
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
    DenseMap<Attribute, SymbolRefAttr> entryPointRefReplacements;

    // For each executable, find the first executable which it is equivalent to.
    for (int i = executableOps.size() - 1; i >= 0; --i) {
      auto duplicateExecutableOp = executableOps[i];

      for (int j = 0; j < i; ++j) {
        auto referenceExecutableOp = executableOps[j];

        if (!areExecutablesEquivalent(duplicateExecutableOp,
                                      referenceExecutableOp)) {
          continue;
        }

        // Found an equivalent executable! Record it and move on to the next.
        duplicateExecutableOps.push_back(duplicateExecutableOp);

        // Record entry point reference replacements.
        for (auto entryOpPair : llvm::zip(
                 duplicateExecutableOp.getBlock().getOps<DispatchEntryOp>(),
                 referenceExecutableOp.getBlock().getOps<DispatchEntryOp>())) {
          auto oldSymbolRefAttr = builder.getSymbolRefAttr(
              duplicateExecutableOp.getName(),
              {builder.getSymbolRefAttr(std::get<0>(entryOpPair).sym_name())});
          auto newSymbolRefAttr = builder.getSymbolRefAttr(
              referenceExecutableOp.getName(),
              {builder.getSymbolRefAttr(std::get<1>(entryOpPair).sym_name())});
          entryPointRefReplacements[oldSymbolRefAttr] = newSymbolRefAttr;
        }

        break;
      }
    }

    totalExecutables = executableOps.size();
    executablesDeduplicated = duplicateExecutableOps.size();
    remainingExecutables = totalExecutables - executablesDeduplicated;

    replaceEntryPointUses(moduleOp, entryPointRefReplacements);

    // Remove the duplicate executables now that they are no longer referenced.
    //
    // Note: removing executables can leave gaps in numbering if they were
    // originally numbered. While we could renumber them, we choose to keep
    // original names (numbers and all) to make it easier to track executables
    // through this pass.
    for (auto executableOp : duplicateExecutableOps) {
      executableOp.erase();
    }
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
    "iree-flow-deduplicate-executables",
    "Deduplicates executables that are identical");

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

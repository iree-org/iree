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

#include "integrations/tensorflow/compiler/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Utils.h"

namespace mlir {
namespace iree_compiler {

namespace {}  // namespace

// Clones FuncOp's until they have a single use only (or no users).
//
// The tf-shape-inference pass doesn't support functions that have more than
// a single use. But some real code from frontends does end up creating code
// like that. For example, the same LSTM cell function or loop body function
// will be reused.
//
// This pass clones functions as needed to establish the invariant that all
// functions have a single use. This can in principle cause exponential code
// size bloat, and should in general be guided by a proper cost model.
//
// There are two factors which should be considered by a principled replacement
// to this pass:
//
// 1. IREE currently relies on "sufficiently good shape inference" for
// correctness so for now the cost of doing this seems acceptable since
// pathological cases haven't hit us yet.
//
// 2. Cloning functions can help by allowing code to be specialized (much as
// inlining does). In fact, tf-shape-inference attempts to do specialization
// of callees which is difficult if callees have multiple uses.
class GuaranteeAllFuncsOneUse
    : public PassWrapper<GuaranteeAllFuncsOneUse, OperationPass<ModuleOp>> {
 public:
  void runOnOperation() override {
    if (failed(run())) {
      signalPassFailure();
    }
  }

  LogicalResult run() {
    auto module = getOperation();

    // Overall strategy:
    // Fixed point iteration, iteratively applying a rule that clones
    // any FuncOp with more than one use to eliminate its uses.

    SymbolTable symbolTable(module);
    bool madeChanges = false;
    // This value needs to be low enough to actually stop compilation in a
    // reasonable time, but not too low that it blocks real programs.
    // This number was chosen semi-randomly.
    const int kMaxClones = 1000;
    int numClones = 0;
    do {
      madeChanges = false;
      for (auto func : llvm::make_early_inc_range(module.getOps<FuncOp>())) {
        auto usesOptional = symbolTable.getSymbolUses(func, module);
        if (!usesOptional.hasValue()) {
          return func.emitError() << "could not walk uses of func";
        }
        auto &uses = *usesOptional;
        if (llvm::size(uses) <= 1) {
          continue;
        }
        // At this point, we know we are going to change the module.
        madeChanges = true;
        for (const SymbolTable::SymbolUse &use : llvm::drop_begin(uses, 1)) {
          auto newFunc = func.clone();
          if (numClones++ > kMaxClones) {
            return func.emitError()
                   << "reached cloning limit (likely recursive call graph or "
                      "repeated diamond-like call structure "
                      "or just very large program)";
          }
          symbolTable.insert(newFunc);
          newFunc.setVisibility(SymbolTable::Visibility::Private);
          if (failed(symbolTable.replaceAllSymbolUses(func, newFunc.getName(),
                                                      use.getUser()))) {
            return func.emitError() << "could not replace symbol use";
          }
        }
      }
    } while (madeChanges);

    return success();
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createGuaranteeAllFuncsOneUse() {
  return std::make_unique<GuaranteeAllFuncsOneUse>();
}

static PassRegistration<GuaranteeAllFuncsOneUse> pass(
    "iree-guarantee-all-funcs-one-use",
    "Guarantee all func's have only a single use.");

}  // namespace iree_compiler
}  // namespace mlir

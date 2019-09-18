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

#include <cstdio>
#include <memory>

#include "third_party/llvm/llvm/projects/google_mlir/include/mlir/Dialect/StandardOps/Ops.h"
#include "third_party/llvm/llvm/projects/google_mlir/include/mlir/IR/BlockAndValueMapping.h"
#include "third_party/llvm/llvm/projects/google_mlir/include/mlir/IR/OperationSupport.h"
#include "third_party/llvm/llvm/projects/google_mlir/include/mlir/IR/PatternMatch.h"
#include "third_party/llvm/llvm/projects/google_mlir/include/mlir/Pass/Pass.h"
#include "third_party/llvm/llvm/projects/google_mlir/include/mlir/Pass/PassRegistry.h"
#include "third_party/mlir_edge/iree/compiler/IR/Interpreter/HLOps.h"
#include "third_party/mlir_edge/iree/compiler/IR/Ops.h"

namespace mlir {
namespace iree_compiler {

namespace optimize_patterns {
#include "third_party/mlir_edge/iree/compiler/Transforms/Interpreter/LegalizeInterpreterOps.inc"
}  // namespace optimize_patterns

void tryElideClone(IREEInterp::HL::CloneOp *cloneOp,
                   std::vector<Operation *> *deadOperations) {
  cloneOp->replaceAllUsesWith(cloneOp->src());
  deadOperations->push_back(cloneOp->getOperation());
}

void removeIdentityClones(FuncOp function) {
  std::vector<Operation *> deadOperations;
  for (auto &block : function.getBlocks()) {
    for (auto &op : block.getOperations()) {
      if (auto cloneOp = dyn_cast<IREEInterp::HL::CloneOp>(op)) {
        tryElideClone(&cloneOp, &deadOperations);
      }
    }
  }
  for (auto *op : deadOperations) {
    op->erase();
  }
}

class LegalizeInterpreterOpsPass
    : public FunctionPass<LegalizeInterpreterOpsPass> {
 public:
  void runOnFunction() override {
    OwningRewritePatternList patterns;
    auto *context = getFunction().getContext();
    optimize_patterns::populateWithGenerated(context, &patterns);
    applyPatternsGreedily(getFunction(), patterns);

    removeIdentityClones(getFunction());
  }
};

std::unique_ptr<OpPassBase<FuncOp>> createLegalizeInterpreterOpsPass() {
  return std::make_unique<LegalizeInterpreterOpsPass>();
}

static PassRegistration<LegalizeInterpreterOpsPass> pass(
    "iree-legalize-interpreter-ops", "Optimizes common IREE op patterns.");

}  // namespace iree_compiler
}  // namespace mlir

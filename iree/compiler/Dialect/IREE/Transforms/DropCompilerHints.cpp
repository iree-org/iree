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

#include <utility>

#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "iree/compiler/Dialect/IREE/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {

class DropCompilerHintsPass
    : public PassWrapper<DropCompilerHintsPass, OperationPass<void>> {
 public:
  void runOnOperation() override {
    // We can't use patterns and applyPatternsAndFoldGreedily because that
    // automatically does canonicalization.
    getOperation()->walk([&](DoNotOptimizeOp op) {
      op.replaceAllUsesWith(op.getOperands());
      op.erase();
    });
  }
};

std::unique_ptr<OperationPass<void>> createDropCompilerHintsPass() {
  return std::make_unique<DropCompilerHintsPass>();
}

static PassRegistration<DropCompilerHintsPass> pass(
    "iree-drop-compiler-hints",
    "Deletes operations that have no runtime equivalent and are only used in "
    "the compiler. This should be performed after all other compiler passes.");

}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

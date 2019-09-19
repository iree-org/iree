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

#include "iree/compiler/IR/Ops.h"
#include "iree/compiler/IR/StructureOps.h"
#include "third_party/llvm/llvm/projects/google_mlir/include/mlir/IR/Builders.h"
#include "third_party/llvm/llvm/projects/google_mlir/include/mlir/IR/PatternMatch.h"
#include "third_party/llvm/llvm/projects/google_mlir/include/mlir/Pass/Pass.h"
#include "third_party/llvm/llvm/projects/google_mlir/include/mlir/Pass/PassRegistry.h"

namespace mlir {
namespace iree_compiler {
namespace {

struct LegalizeTensorMemRefPass
    : public FunctionPass<LegalizeTensorMemRefPass> {
  void runOnFunction() override {
    for (auto& block : getFunction().getBlocks()) {
      for (auto& op : block.getOperations()) {
        if (isa<IREE::TensorToMemRefOp>(op) &&
            isa_and_nonnull<IREE::MemRefToTensorOp>(
                op.getOperand(0)->getDefiningOp())) {
          op.getResult(0)->replaceAllUsesWith(
              op.getOperand(0)->getDefiningOp()->getOperand(0));
        }
      }

      // Performs cleanup of ops removed above.
      OwningRewritePatternList patterns;
      applyPatternsGreedily(getFunction(), patterns);
    }
  }
};

}  // namespace

std::unique_ptr<OpPassBase<FuncOp>> createLegalizeTensorMemRefPass() {
  return std::make_unique<LegalizeTensorMemRefPass>();
}

static PassRegistration<LegalizeTensorMemRefPass> legalize_tensor_memref_pass(
    "iree-legalize-tensor-memref", "Remove extra tensor/memref operations.");

}  // namespace iree_compiler
}  // namespace mlir

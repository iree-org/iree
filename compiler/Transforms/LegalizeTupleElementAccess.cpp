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

#include "third_party/llvm/llvm/include/llvm/ADT/ArrayRef.h"
#include "third_party/llvm/llvm/include/llvm/ADT/SmallVector.h"
#include "third_party/llvm/llvm/include/llvm/ADT/iterator_range.h"
#include "third_party/llvm/llvm/projects/google_mlir/include/mlir/IR/BlockAndValueMapping.h"
#include "third_party/llvm/llvm/projects/google_mlir/include/mlir/IR/Builders.h"
#include "third_party/llvm/llvm/projects/google_mlir/include/mlir/IR/PatternMatch.h"
#include "third_party/llvm/llvm/projects/google_mlir/include/mlir/IR/StandardTypes.h"
#include "third_party/llvm/llvm/projects/google_mlir/include/mlir/Pass/Pass.h"
#include "third_party/llvm/llvm/projects/google_mlir/include/mlir/Pass/PassRegistry.h"
#include "third_party/llvm/llvm/projects/google_mlir/include/mlir/Transforms/Utils.h"
#include "third_party/tensorflow/compiler/mlir/xla/ir/hlo_ops.h"

using mlir::PassRegistration;

namespace mlir {
namespace iree_compiler {
namespace {

struct LegalizeTupleElementAccessPass
    : public FunctionPass<LegalizeTupleElementAccessPass> {
  void runOnFunction() override;
};

/// Legalizes XLA Tuple element access, mostly be removing them entirely.
void LegalizeTupleElementAccessPass::runOnFunction() {
  auto func = getFunction();
  Builder builder(func.getContext());

  llvm::SmallVector<Operation*, 10> ops_to_remove;
  func.walk([&](xla_hlo::TupleOp op) {
    bool can_remove_tuple = true;
    for (auto* user : op.getResult()->getUsers()) {
      if (auto get_op = dyn_cast<xla_hlo::GetTupleElementOp>(user)) {
        get_op.getResult()->replaceAllUsesWith(
            op.getOperand(get_op.index().getLimitedValue()));
        ops_to_remove.push_back(get_op.getOperation());
      } else {
        can_remove_tuple = false;
      }
    }

    if (can_remove_tuple) {
      ops_to_remove.push_back(op.getOperation());
    }
  });

  // Cleanup ops that have been bypassed.
  for (auto op : ops_to_remove) {
    op->erase();
  }
}

}  // namespace

std::unique_ptr<OpPassBase<FuncOp>> createLegalizeTupleElementAccessPass() {
  return std::make_unique<LegalizeTupleElementAccessPass>();
}

static PassRegistration<LegalizeTupleElementAccessPass> legalize_pass(
    "legalize-tuple-element-access",
    "Remove xla_hlo::GetTupleElementOp commands wherever possible.");

}  // namespace iree_compiler
}  // namespace mlir

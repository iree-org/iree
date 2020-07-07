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

#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace VMLA {

namespace {

// Unrolls a multi-dimensional mhlo.reduce op into one mhlo.reduce op per
// dimension. The XLA operation semantics state that this is a valid
// transformation.
void unrollReduceOp(mhlo::ReduceOp reduceOp) {
  // Create one op per dimension being reduced.
  // We'll do this by chaining the original input through with the temporary
  // reduction results. The results we end up with will be the originally
  // requested shape and we can just substitute them.
  SmallVector<int64_t, 4> sortedDimensions{
      reduceOp.dimensions().getValues<int64_t>()};
  llvm::sort(sortedDimensions,
             [](int64_t a, int64_t b) { return (a - b) > 0; });

  // Insert at the same place as the original op.
  OpBuilder builder(reduceOp);
  SmallVector<Value, 4> temps{reduceOp.operands()};
  for (int64_t dimension : sortedDimensions) {
    // Create the new reduction using the results of the previous operation.
    auto singleAttrType =
        RankedTensorType::get({1}, builder.getIntegerType(64));
    auto singleReduceOp = builder.create<mhlo::ReduceOp>(
        reduceOp.getLoc(), temps, reduceOp.init_values(),
        DenseIntElementsAttr::get(singleAttrType, {dimension}));
    BlockAndValueMapping mapping;
    reduceOp.body().cloneInto(&singleReduceOp.body(), mapping);
    temps = singleReduceOp.getResults();
  }

  // Replace uses of the existing results with the new results.
  reduceOp.replaceAllUsesWith(temps);

  // Erase original op.
  reduceOp.erase();
}

}  // namespace

class UnrollReductionsPass
    : public PassWrapper<UnrollReductionsPass, FunctionPass> {
 public:
  void runOnFunction() override {
    for (auto &block : getFunction()) {
      auto reduceOps = llvm::to_vector<4>(block.getOps<mhlo::ReduceOp>());
      for (auto reduceOp : reduceOps) {
        if (reduceOp.dimensions().getNumElements() > 1) {
          unrollReduceOp(reduceOp);
        }
      }
    }
  }
};

std::unique_ptr<OperationPass<FuncOp>> createUnrollReductionsPass() {
  return std::make_unique<UnrollReductionsPass>();
}

static PassRegistration<UnrollReductionsPass> pass(
    "iree-vmla-unroll-reductions",
    "Unrolls multi-dimensional reductions to one reduction per dimension.");

}  // namespace VMLA
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

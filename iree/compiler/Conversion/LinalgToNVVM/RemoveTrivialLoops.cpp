// Copyright 2021 Google LLC
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

#include "iree/compiler/Conversion/Common/Transforms.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {

/// If the value is a threadID return the range [0, workgroupSize-1].
static Optional<std::pair<AffineExpr, AffineExpr>> threadIdMinMax(
    Value value, SmallVectorImpl<Value> &dims, SmallVectorImpl<Value> &symbols,
    ArrayRef<int32_t> workgroupSize) {
  if (auto idOp = value.getDefiningOp<gpu::ThreadIdOp>()) {
    unsigned index = StringSwitch<unsigned>(idOp.dimension())
                         .Case("x", 0)
                         .Case("y", 1)
                         .Case("z", 2);
    OpBuilder b(value.getContext());
    AffineExpr zero = b.getAffineConstantExpr(0);
    AffineExpr ubExpr = b.getAffineConstantExpr(workgroupSize[index]);
    return std::make_pair(zero, ubExpr - 1);
  }
  return {};
}

namespace {

class RemoveSingleIterationLoopPass
    : public PassWrapper<RemoveSingleIterationLoopPass, OperationPass<FuncOp>> {
  void runOnOperation() override {
    FuncOp funcOp = getOperation();
    std::array<int32_t, 3> workgroupSize;
    for (auto it : llvm::enumerate(funcOp->getAttr("cuda_workgroup_size")
                                       .cast<DenseIntElementsAttr>()
                                       .getIntValues())) {
      workgroupSize[it.index()] = it.value().getZExtValue();
    }
    auto getThreadIdMinMax = [&workgroupSize](Value value,
                                              SmallVectorImpl<Value> &dims,
                                              SmallVectorImpl<Value> &symbols) {
      return threadIdMinMax(value, dims, symbols, workgroupSize);
    };
    MLIRContext *context = funcOp->getContext();
    OwningRewritePatternList patterns(context);
    populateRemoveSingleIterationLoopPattern(patterns, getThreadIdMinMax);
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
  }
};

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createRemoveSingleIterationLoopPass() {
  return std::make_unique<RemoveSingleIterationLoopPass>();
}

static PassRegistration<RemoveSingleIterationLoopPass> pass(
    "iree-cuda-remove-single-iteration-loop",
    "Remove distributed loop with single iteration.");

}  // namespace iree_compiler
}  // namespace mlir

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

//===-RemoveDeadMemAllocsPass.cpp - Pass to remove dead alloc-like ops ----===//
//
// Pass to remove operations with Allocate MemoryEffects when the allocations
// are dead.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Conversion/Common/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {

namespace {
struct RemoveDeadMemAllocs : RewritePattern {
  RemoveDeadMemAllocs(PatternBenefit benefit = 1)
      : RewritePattern(benefit, MatchAnyOpTypeTag()) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto memEffect = dyn_cast<MemoryEffectOpInterface>(op);
    if (!memEffect || !memEffect.hasEffect<MemoryEffects::Allocate>()) {
      return failure();
    }
    if (!op->use_empty()) return failure();
    rewriter.eraseOp(op);
    return success();
  }
};

struct RemoveDeadMemAllocsPass
    : public PassWrapper<RemoveDeadMemAllocsPass, OperationPass<>> {
  void runOnOperation() override {
    OwningRewritePatternList patterns;
    patterns.insert<RemoveDeadMemAllocs>();
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};
}  // namespace

std::unique_ptr<OperationPass<>> createRemoveDeadMemAllocsPass() {
  return std::make_unique<RemoveDeadMemAllocsPass>();
}

static PassRegistration<RemoveDeadMemAllocsPass> pass(
    "iree-codegen-remove-dead-mem-allocs",
    "Remove operations with Allocate semantics that have no uses",
    [] { return std::make_unique<RemoveDeadMemAllocsPass>(); });

}  // namespace iree_compiler
}  // namespace mlir

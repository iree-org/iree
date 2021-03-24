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

#include "iree/compiler/Conversion/LinalgToLLVM/Passes.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {

namespace {

struct PlanConvLoopOrderPass
    : PassWrapper<PlanConvLoopOrderPass, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }
  void runOnFunction() override;
};

}  // namespace

void PlanConvLoopOrderPass::runOnFunction() {
  auto funcOp = getOperation();
  auto context = funcOp.getContext();

  auto marker = Identifier::get("generalized_from_conv", context);
  linalg::LinalgTransformationFilter firstStepMarker(
      /*matchDisjunction=*/ArrayRef<Identifier>(),
      /*replacement=*/marker);
  linalg::LinalgTransformationFilter secondStepMarker(
      /*matchDisjunction=*/marker,
      /*replacement=*/llvm::None);

  SmallVector<unsigned, 8> loopOrder = {
      /*batch=*/0,
      /*output_height=*/1,
      /*output_width=*/2,
      /*filter_height=*/5,
      /*filter_width=*/6,
      /*input_channel=*/4,
      /*output_channel=*/3,
  };

  OwningRewritePatternList patterns(&getContext());
  linalg::populateLinalgConvGeneralizationPatterns(patterns, firstStepMarker);
  patterns.insert<linalg::LinalgInterchangePattern<linalg::GenericOp>>(
      context, loopOrder, secondStepMarker);

  (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}

std::unique_ptr<FunctionPass> createPlanConvLoopOrderPass() {
  return std::make_unique<PlanConvLoopOrderPass>();
}

static PassRegistration<PlanConvLoopOrderPass> pass(
    "iree-codegen-linalg-to-llvm-plan-conv-loop-order",
    "Convert linalg.conv to linalg.generic with a CPU-friendly iterator order",
    [] { return std::make_unique<PlanConvLoopOrderPass>(); });

}  // namespace iree_compiler
}  // namespace mlir

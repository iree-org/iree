// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Conversion/PassDetail.h"
#include "iree/compiler/Conversion/Passes.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {

namespace {

struct LinalgToLLVMPlanConvLoopOrderPass
    : LinalgToLLVMPlanConvLoopOrderBase<LinalgToLLVMPlanConvLoopOrderPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }
  void runOnOperation() override;
};

}  // namespace

void LinalgToLLVMPlanConvLoopOrderPass::runOnOperation() {
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
  patterns.insert<linalg::GenericOpInterchangePattern>(context, loopOrder,
                                                       secondStepMarker);

  (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}

std::unique_ptr<OperationPass<FuncOp>>
createLinalgToLLVMPlanConvLoopOrderPass() {
  return std::make_unique<LinalgToLLVMPlanConvLoopOrderPass>();
}

}  // namespace iree_compiler
}  // namespace mlir

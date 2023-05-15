// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/CommonPasses.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-rematerialize-parallel-ops"

namespace mlir {
namespace iree_compiler {

namespace {

/// Merge elementwise operations into their consumers.
struct MergeElementwiseOps : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter& rewriter) const override {
    // Find the first operand that is defined by another generic op on tensors.
    for (OpOperand& opOperand : genericOp->getOpOperands()) {
      if (!linalg::areElementwiseOpsFusable(&opOperand)) continue;

      FailureOr<linalg::ElementwiseOpFusionResult> fusionResult =
          linalg::fuseElementwiseOps(rewriter, &opOperand);
      if (succeeded(fusionResult)) {
        // Forward lowering config.
        if (auto loweringAttr = getLoweringConfig(genericOp)) {
          setLoweringConfig(fusionResult->fusedOp, loweringAttr);
        }
        auto replacements = fusionResult->fusedOp->getResults().take_back(
            genericOp.getNumResults());
        rewriter.replaceOp(genericOp, replacements);
        return success();
      }
    }
    return failure();
  }
};

struct RematerializeParallelOpsPass
    : public RematerializeParallelOpsBase<RematerializeParallelOpsPass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    RewritePatternSet fusionPatterns(funcOp.getContext());
    fusionPatterns.insert<MergeElementwiseOps>(funcOp.getContext());
    linalg::populateEraseUnusedOperandsAndResultsPatterns(fusionPatterns);
    if (failed(
            applyPatternsAndFoldGreedily(funcOp, std::move(fusionPatterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createRematerializeParallelOpsPass() {
  return std::make_unique<RematerializeParallelOpsPass>();
}

}  // namespace iree_compiler
}  // namespace mlir

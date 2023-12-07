// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/PassDetail.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/IREECodegenAttrs.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-rematerialize-parallel-ops"

namespace mlir::iree_compiler {

namespace {

static bool isScalarOrTensorOfSizeOne(Type t) {
  if (auto tensorType = dyn_cast<RankedTensorType>(t)) {
    return tensorType.hasStaticShape() && tensorType.getNumElements() == 1;
  }
  return t.isIntOrIndexOrFloat();
}

/// Rematerialize all parallel elementwise operations into its users within a
/// `flow.dispatch.region`.
struct RematerializeParallelOpsPattern
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    // Avoid doing this for scalar operations.
    auto isScalarValue = [](Value v) {
      return isScalarOrTensorOfSizeOne(v.getType());
    };
    if (llvm::all_of(genericOp.getOperands(), isScalarValue) &&
        llvm::all_of(genericOp.getResults(), isScalarValue)) {
      return failure();
    }

    // Find the first operand that is defined by another generic op on tensors.
    for (OpOperand &opOperand : genericOp->getOpOperands()) {
      if (!linalg::areElementwiseOpsFusable(&opOperand))
        continue;

      FailureOr<linalg::ElementwiseOpFusionResult> fusionResult =
          linalg::fuseElementwiseOps(rewriter, &opOperand);
      if (succeeded(fusionResult)) {
        auto replacements = fusionResult->fusedOp->getResults().take_back(
            genericOp.getNumResults());
        // Copy over any non native attributes for the operation.
        auto prunedAttributeList = linalg::getPrunedAttributeList(genericOp);
        fusionResult->fusedOp->setAttrs(prunedAttributeList);
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
    fusionPatterns.insert<RematerializeParallelOpsPattern>(funcOp.getContext());
    linalg::populateEraseUnusedOperandsAndResultsPatterns(fusionPatterns);
    if (failed(
            applyPatternsAndFoldGreedily(funcOp, std::move(fusionPatterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createRematerializeParallelOpsPass() {
  return std::make_unique<RematerializeParallelOpsPass>();
}

} // namespace mlir::iree_compiler

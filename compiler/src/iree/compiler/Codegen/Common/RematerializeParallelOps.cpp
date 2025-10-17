// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-rematerialize-parallel-ops"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_REMATERIALIZEPARALLELOPSPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

static bool isScalarOrTensorOfSizeOne(Type t) {
  if (auto tensorType = dyn_cast<RankedTensorType>(t)) {
    return tensorType.hasStaticShape() && tensorType.getNumElements() == 1;
  }
  return t.isIntOrIndexOrFloat();
}

static bool hasDirectWriteResult(Operation *op) {
  return llvm::any_of(op->getUsers(),
                      llvm::IsaPred<IREE::TensorExt::DispatchTensorStoreOp>);
}

/// Rematerialize all parallel elementwise operations into its users within a
/// `flow.dispatch.region`.
struct RematerializeParallelOpsPattern
    : public OpRewritePattern<linalg::GenericOp> {
  using Base::Base;

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
      if (!linalg::areElementwiseOpsFusable(&opOperand)) {
        continue;
      }
      auto producer = opOperand.get().getDefiningOp<linalg::GenericOp>();
      if (producer && hasExternalCapture(producer)) {
        continue;
      }
      if (producer && hasDirectWriteResult(producer)) {
        continue;
      }
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

struct RematerializeParallelOpsPass final
    : impl::RematerializeParallelOpsPassBase<RematerializeParallelOpsPass> {
  void runOnOperation() override {
    mlir::FunctionOpInterface funcOp = getOperation();
    RewritePatternSet fusionPatterns(funcOp.getContext());
    fusionPatterns.insert<RematerializeParallelOpsPattern>(funcOp.getContext());
    linalg::populateEraseUnusedOperandsAndResultsPatterns(fusionPatterns);
    if (failed(applyPatternsGreedily(funcOp, std::move(fusionPatterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler

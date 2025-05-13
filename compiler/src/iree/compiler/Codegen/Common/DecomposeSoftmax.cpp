// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_DECOMPOSESOFTMAXPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

static bool isScalarOrTensorOfSizeOne(Type t) {
  if (auto tensorType = dyn_cast<RankedTensorType>(t)) {
    return tensorType.hasStaticShape() && tensorType.getNumElements() == 1;
  }
  return t.isIntOrIndexOrFloat();
}

struct FuseElementWiseGenericOps : public OpRewritePattern<linalg::GenericOp> {
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
        rewriter.replaceOp(genericOp, replacements);
        return success();
      }
    }
    return failure();
  }
};

/// Given an N-dimensional tensor x, this op converts
/// softmax(x) to the following sequence of operations:
///
/// 1. Compute the max of x along dimension d. This results
///    in a N-1 dimensional tensor m.
///    m = max(x, dim = d)
///
/// 2. Subtract m from x and exponentiate. This results in
///    a N dimensional tensor z.
///    z = exp(x - m)
///
/// 3. Compute the sum of z along dimension d. This results in
///    a N-1 dimensional tensor l.
///    l = sum(z, dim = d)
///
/// 4. Divide z and l. This gives the N-dimensional softmax.
///    softmax = z / l
///
static LogicalResult convertSoftmaxToGenerics(mlir::FunctionOpInterface funcOp,
                                              RewritePatternSet &patterns,
                                              bool useFusion) {
  IRRewriter rewriter(funcOp.getContext());
  SmallVector<Operation *> toDelete;
  SmallVector<Operation *> softmaxOpsToDecompose;
  funcOp.walk([&](linalg::SoftmaxOp softmaxOp) {
    softmaxOpsToDecompose.push_back(softmaxOp);
  });
  OpBuilder::InsertionGuard guard(rewriter);
  for (Operation *softmaxOp : softmaxOpsToDecompose) {
    // Cast linalg::softmax to AggregatedOpInterface since this where
    // `decomposeOperation` is implemented.
    auto decomposableSoftmaxOp = cast<linalg::AggregatedOpInterface>(softmaxOp);

    // Decompose linalg::softmax.
    FailureOr<SmallVector<Value>> result =
        decomposableSoftmaxOp.decomposeOperation(rewriter);
    if (failed(result)) {
      failed(rewriter.notifyMatchFailure(
          softmaxOp, "linalg::SoftmaxOp could not be decomposed"));
      return failure();
    }
    // Replace the result of linalg::softmax with the `result` generated via
    // the decomposition above.
    rewriter.replaceOp(decomposableSoftmaxOp, *result);
  }
  if (!softmaxOpsToDecompose.empty() && useFusion) {
    patterns.insert<FuseElementWiseGenericOps>(funcOp.getContext());
  }
  return success();
}

struct DecomposeSoftmaxPass
    : impl::DecomposeSoftmaxPassBase<DecomposeSoftmaxPass> {
  using impl::DecomposeSoftmaxPassBase<
      DecomposeSoftmaxPass>::DecomposeSoftmaxPassBase;
  explicit DecomposeSoftmaxPass(bool useFusion) { this->useFusion = useFusion; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    IRRewriter rewriter(context);
    RewritePatternSet patterns(context);
    if (failed(convertSoftmaxToGenerics(getOperation(), patterns, useFusion))) {
      return signalPassFailure();
    }
    linalg::populateEraseUnusedOperandsAndResultsPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createDecomposeSoftmaxPass(bool useFusion) {
  return std::make_unique<DecomposeSoftmaxPass>(useFusion);
}

} // namespace mlir::iree_compiler

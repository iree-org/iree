// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/PassDetail.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler {

namespace {
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
static LogicalResult convertSoftmaxToGenerics(func::FuncOp funcOp) {
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
  for (Operation *op : toDelete) {
    rewriter.eraseOp(op);
  }

  return success();
}

struct DecomposeSoftmaxPass : DecomposeSoftmaxBase<DecomposeSoftmaxPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    IRRewriter rewriter(context);
    if (failed(convertSoftmaxToGenerics(getOperation())))
      return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createDecomposeSoftmaxPass() {
  return std::make_unique<DecomposeSoftmaxPass>();
}

} // namespace mlir::iree_compiler

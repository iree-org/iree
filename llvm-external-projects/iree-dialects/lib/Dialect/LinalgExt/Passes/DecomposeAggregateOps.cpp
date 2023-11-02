// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgExt/Passes/PassDetail.h"
#include "iree-dialects/Dialect/LinalgExt/Passes/Passes.h"
#include "iree-dialects/Dialect/LinalgExt/Utils/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace LinalgExt {

namespace {

/// Following utility traverses through a given `funcOp`, gets hold of those ops
/// which have implemented `AggregatedOpInterface` and decomposes them.
LogicalResult DecomposeAggregateOpsToGenerics(func::FuncOp funcOp) {
  IRRewriter rewriter(funcOp.getContext());
  SmallVector<Operation *> toDelete;
  SmallVector<linalg::AggregatedOpInterface> opsToDecompose;
  funcOp.walk([&](linalg::AggregatedOpInterface decomposableOp) {
    opsToDecompose.push_back(decomposableOp);
  });

  OpBuilder::InsertionGuard guard(rewriter);
  for (linalg::AggregatedOpInterface decomposableOp : opsToDecompose) {
    FailureOr<SmallVector<Value>> result =
        decomposableOp.decomposeOperation(rewriter);
    if (failed(result)) {
      failed(rewriter.notifyMatchFailure(
          decomposableOp, "linalg::SoftmaxOp could not be decomposed"));
      return failure();
    }
    // Replace the result of the original op with the `result` generated via
    // the decomposition above.
    rewriter.replaceOp(decomposableOp, *result);

    // TODO(avarma): The following code needs to be part of a separate Fusing
    //               utility.
    //               So, after doing this we can incrementally percolate down
    //               FlashAttention and Winograd.
    // Fusion later depends on couple of Ops/Values - we try to obtain the same
    // by backtracking through the generated value's def-chain.
    Operation *resultOp = (*result)[0].getDefiningOp();
    Value numerator = resultOp->getOperand(0);
    Operation *numeratorOp = numerator.getDefiningOp();

    // Rematerialize operands that are marked for this.
    SmallVector<OpOperand *> uses = llvm::to_vector(llvm::map_range(
        numerator.getUses(), [](OpOperand &use) { return &use; }));
    for (OpOperand *use : uses) {
      Operation *consumer = use->getOwner();
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPoint(consumer);
      FailureOr<linalg::ElementwiseOpFusionResult> fusionResult =
          linalg::fuseElementwiseOps(rewriter, use);
      if (succeeded(fusionResult)) {
        SmallVector<Value> replacements = llvm::to_vector(
            llvm::map_range(consumer->getResults(), [&](Value oldValue) {
              return fusionResult->replacements.lookup(oldValue);
            }));
        rewriter.replaceOp(consumer, replacements);
      }
    }
    toDelete.push_back(numeratorOp);
  }
  for (Operation *op : toDelete) {
    rewriter.eraseOp(op);
  }

  return success();
}

struct DecomposeAggregateOpsPass
    : DecomposeAggregateOpsBase<DecomposeAggregateOpsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<linalg::LinalgDialect, IREE::LinalgExt::IREELinalgExtDialect>();
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    IRRewriter rewriter(context);
    if (failed(DecomposeAggregateOpsToGenerics(getOperation())))
      return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createDecomposeAggregateOpsPass() {
  return std::make_unique<DecomposeAggregateOpsPass>();
}

} // namespace LinalgExt
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir

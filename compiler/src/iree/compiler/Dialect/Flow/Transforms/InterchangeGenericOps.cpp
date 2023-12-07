// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===--------------- InterchangeGenericOps.cpp ----------------------------===//
//
// Interchange loops in generic ops to force the reduction loops to be the most
// inner loops.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::Flow {

namespace {

struct GenericOpInterchangePattern
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    SmallVector<unsigned> interchange;
    bool needInterchange = false;
    unsigned numParallelLoop = genericOp.getNumParallelLoops();
    if (numParallelLoop == 0)
      return failure();
    for (auto iter : llvm::enumerate(genericOp.getIteratorTypesArray())) {
      if (linalg::isParallelIterator(iter.value())) {
        interchange.push_back(iter.index());
        if (iter.index() >= numParallelLoop)
          needInterchange = true;
      }
    }
    // If all the parallel loops are outter loops skip the pattern.
    if (!needInterchange)
      return failure();
    for (auto iter : llvm::enumerate(genericOp.getIteratorTypesArray())) {
      if (linalg::isReductionIterator(iter.value())) {
        interchange.push_back(iter.index());
      }
    }
    return interchangeGenericOp(rewriter, genericOp, interchange);
  }
};

struct InterchangeGenericOpsPass
    : public InterchangeGenericOpsBase<InterchangeGenericOpsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<GenericOpInterchangePattern>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createInterchangeGenericOpsPass() {
  return std::make_unique<InterchangeGenericOpsPass>();
}

} // namespace mlir::iree_compiler::IREE::Flow

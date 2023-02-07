// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtInterfaces.h"
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgTransform/StructuredTransformOpsExt.h"
#include "iree-dialects/Transforms/TransformMatchers.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using transform_ext::StructuredOpMatcher;

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

namespace {

struct RaiseSpecialOpsPass : public RaiseSpecialOpsBase<RaiseSpecialOpsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::LinalgExt::IREELinalgExtDialect>();
  }

  void runOnOperation() override {
    SmallVector<linalg::LinalgOp> softmaxRoots;
    getOperation()->walk([&](linalg::LinalgOp op) {
      StructuredOpMatcher reduction, fill, leading, trailing;
      transform_ext::MatchedReductionCaptures captures;
      transform_ext::StructuredOpMatcher fillMinusInf;
      transform_ext::StructuredOpMatcher maxReduction;
      transform_ext::StructuredOpMatcher sub;
      transform_ext::StructuredOpMatcher expOperand;
      transform_ext::StructuredOpMatcher fillzero;
      transform_ext::StructuredOpMatcher sum;
      transform_ext::StructuredOpMatcher divOperand;
      transform_ext::StructuredOpMatcher softmaxroot;
      makeSoftmaxMatcher(fillMinusInf, maxReduction, sub, expOperand, fillzero,
                         sum, divOperand, softmaxroot);
      if (matchPattern(op, softmaxroot)) {
        softmaxRoots.push_back(op);
      }
    });
    for (linalg::LinalgOp op : softmaxRoots) {
      Value src = op->getOperand(0)
                      .getDefiningOp()
                      ->getOperand(0)
                      .getDefiningOp()
                      ->getOperand(0);
      OpBuilder builder(op);
      auto softmax = builder.create<IREE::LinalgExt::SoftmaxOp>(
          op.getLoc(), src, op.getDpsInitOperand(0)->get(),
          op.getNumLoops() - 1);
      op->replaceAllUsesWith(softmax->getResults());
      op->erase();
    }
  }
};

}  // namespace

std::unique_ptr<Pass> createRaiseSpecialOps() {
  return std::make_unique<RaiseSpecialOpsPass>();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

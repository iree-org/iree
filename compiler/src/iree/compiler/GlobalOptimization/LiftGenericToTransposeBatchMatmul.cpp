// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/GlobalOptimization/PassDetail.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {
namespace GlobalOptimization {

namespace {

// Converts linalg.conv_2d_input_nhwc_filter_nhwc op to linalg.matmul
class LiftGenericToTransposeBatchMatmul
    : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp convOp,
                                PatternRewriter &rewriter) const override {
    return failure();
  }
};

struct LiftGenericToTransposeBatchMatmulPass
    : public LiftGenericToTransposeBatchMatmulBase<
          LiftGenericToTransposeBatchMatmulPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(&getContext());
    patterns.insert<LiftGenericToTransposeBatchMatmul>(context);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<Pass> createLiftGenericToTransposeBatchMatmulPass() {
  return std::make_unique<LiftGenericToTransposeBatchMatmulPass>();
}

} // namespace GlobalOptimization
} // namespace iree_compiler
} // namespace mlir

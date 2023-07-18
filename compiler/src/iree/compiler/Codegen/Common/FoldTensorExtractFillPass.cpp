// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/PassDetail.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace mlir {
namespace iree_compiler {

namespace {

struct FoldTensorExtractFillPattern : public OpRewritePattern<tensor::ExtractOp> {
public:
  using OpRewritePattern<tensor::ExtractOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractOp extractOp,
                                PatternRewriter &rewriter) const override {
    // see if tensor input of tensor.extract op is result of linalg.fill op
    auto tensorFill = extractOp.getTensor().getDefiningOp<linalg::FillOp>();
    if (!tensorFill) {
      return failure();
    }

    // get scalar input operand of linalg.fill
    Value extractedScalar = tensorFill.getInputs()[0];

    // replace tensor.extract op with op that simply produces the scalar
    rewriter.replaceOpWithNewOp<arith::ExtFOp>(
        extractOp, extractedScalar.getType(), extractedScalar);
    return success();
  }
};

struct FoldTensorExtractFillPass
    : public FoldTensorExtractFillBase<FoldTensorExtractFillPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(&getContext());
    patterns.insert<FoldTensorExtractFillPattern>(context);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createFoldTensorExtractFillPass() {
  return std::make_unique<FoldTensorExtractFillPass>();
}

} // namespace iree_compiler
} // namespace mlir

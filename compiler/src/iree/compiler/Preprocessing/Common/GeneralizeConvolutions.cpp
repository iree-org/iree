// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Preprocessing/Common/PassDetail.h"
#include "iree/compiler/Preprocessing/Common/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {

namespace {

template <typename LinalgOpType>
class GeneralizeTargetNamedOp final : public OpRewritePattern<LinalgOpType> {
 public:
  using OpRewritePattern<LinalgOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(LinalgOpType linalgOp,
                                PatternRewriter &rewriter) const override {
    FailureOr<linalg::GenericOp> genericOp =
        linalg::generalizeNamedOp(rewriter, linalgOp);
    if (failed(genericOp)) return failure();
    return success();
  }
};

struct GeneralizeConvolutionsPass
    : GeneralizeConvolutionsBase<GeneralizeConvolutionsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(&getContext());
    patterns.insert<GeneralizeTargetNamedOp<linalg::Conv2DNchwFchwOp>>(context);
    patterns.insert<GeneralizeTargetNamedOp<linalg::Conv2DNhwcHwcfOp>>(context);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<Pass> createGeneralizeConvolutionsPass() {
  return std::make_unique<GeneralizeConvolutionsPass>();
}

}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_CONVERTSPLATCONSTANTTOFILLPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

struct ConvertSplatToFill final : OpRewritePattern<arith::ConstantOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::ConstantOp constant,
                                PatternRewriter &rewriter) const override {
    auto resultType = dyn_cast<RankedTensorType>(constant.getType());
    if (!resultType) {
      return failure();
    }

    auto splatAttr = llvm::dyn_cast<SplatElementsAttr>(constant.getValue());
    if (!splatAttr) {
      return failure();
    }

    Location loc = constant.getLoc();

    // Constants have no dynamic sizes.
    Value empty = rewriter.create<tensor::EmptyOp>(constant.getLoc(),
                                                   resultType, ValueRange{});
    auto fillAttr = splatAttr.getSplatValue<TypedAttr>();
    Value fillValue = rewriter.create<arith::ConstantOp>(
        loc, resultType.getElementType(), fillAttr);
    rewriter.replaceOpWithNewOp<linalg::FillOp>(constant, resultType, fillValue,
                                                empty);
    return success();
  }
};

struct ConvertSplatConstantToFillPass final
    : impl::ConvertSplatConstantToFillPassBase<ConvertSplatConstantToFillPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, tensor::TensorDialect>();
  }
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    auto funcOp = getOperation();

    RewritePatternSet patterns(ctx);
    patterns.add<ConvertSplatToFill>(ctx);
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
  }
};

} // namespace
} // namespace mlir::iree_compiler

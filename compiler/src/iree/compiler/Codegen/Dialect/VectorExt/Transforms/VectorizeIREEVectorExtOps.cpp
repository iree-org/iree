// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/Transforms/Passes.h"
#include "iree/compiler/Codegen/Interfaces/VectorizableOpInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::VectorExt {

#define GEN_PASS_DEF_VECTORIZEIREEVECTOREXTOPSPASS
#include "iree/compiler/Codegen/Dialect/VectorExt/Transforms/Passes.h.inc"

namespace {

struct VectorizeToLayoutOpPattern final
    : OpRewritePattern<IREE::VectorExt::ToLayoutOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(IREE::VectorExt::ToLayoutOp toLayoutOp,
                                PatternRewriter &rewriter) const override {
    auto vectorizableOp =
        cast<VectorizableOpInterface>(toLayoutOp.getOperation());
    SmallVector<int64_t> vectorSizes;
    SmallVector<bool> scalableDims;
    if (!vectorizableOp.isVectorizable(vectorSizes, scalableDims)) {
      return failure();
    }
    FailureOr<SmallVector<Value>> result =
        vectorizableOp.vectorize(rewriter, vectorSizes, scalableDims);
    if (failed(result)) {
      return failure();
    }
    rewriter.replaceOp(toLayoutOp, *result);
    return success();
  }
};

struct VectorizeIREEVectorExtOpsPass final
    : impl::VectorizeIREEVectorExtOpsPassBase<VectorizeIREEVectorExtOpsPass> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<VectorizeToLayoutOpPattern>(ctx);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::VectorExt

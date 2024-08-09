// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_VECTORIZEMEMREFCOPYPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

struct VectorizeMemrefCopyPass
    : public impl::VectorizeMemrefCopyPassBase<VectorizeMemrefCopyPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, vector::VectorDialect>();
  }
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    auto funcOp = getOperation();

    RewritePatternSet patterns(ctx);
    patterns.add<linalg::CopyVectorizationPattern>(&getContext());
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
  }
};

} // namespace
} // namespace mlir::iree_compiler

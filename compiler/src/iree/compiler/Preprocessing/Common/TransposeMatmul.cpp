// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Preprocessing/Common/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::Preprocessing {

#define GEN_PASS_DEF_TRANSPOSEMATMULPASS
#include "iree/compiler/Preprocessing/Common/Passes.h.inc" // IWYU pragma: export

namespace {
struct TransposeMatmulPass
    : public impl::TransposeMatmulPassBase<TransposeMatmulPass> {
  using Base::Base;

  void runOnOperation() override {
    bool transposeLHS = input == Preprocessing::TransposeMatmulInput::Lhs;

    RewritePatternSet patterns(&getContext());
    linalg::populateTransposeMatmulPatterns(patterns, transposeLHS);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }

private:
};
} // namespace

} // namespace mlir::iree_compiler::Preprocessing

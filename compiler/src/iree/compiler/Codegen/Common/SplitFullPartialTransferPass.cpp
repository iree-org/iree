// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string>

#include "iree/compiler/Codegen/Common/Passes.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_SPLITFULLPARTIALTRANSFERPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

struct SplitFullPartialTransferPass final
    : impl::SplitFullPartialTransferPassBase<SplitFullPartialTransferPass> {
  using impl::SplitFullPartialTransferPassBase<
      SplitFullPartialTransferPass>::SplitFullPartialTransferPassBase;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);

    auto options = vector::VectorTransformsOptions().setVectorTransferSplit(
        llvm::StringSwitch<vector::VectorTransferSplit>(
            splitVectorTransfersTo.getValue())
            .Case("none", vector::VectorTransferSplit::None)
            .Case("linalg-copy", vector::VectorTransferSplit::LinalgCopy)
            .Case("vector-transfers",
                  vector::VectorTransferSplit::VectorTransfer)
            .Default(vector::VectorTransferSplit::None));
    populateVectorTransferFullPartialPatterns(patterns, options);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler

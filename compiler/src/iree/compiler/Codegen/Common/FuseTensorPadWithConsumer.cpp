// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_FUSETENSORPADWITHCONSUMERPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

struct FuseTensorPadWithConsumerPass final
    : public impl::FuseTensorPadWithConsumerPassBase<
          FuseTensorPadWithConsumerPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto funcOp = getOperation();

    RewritePatternSet patterns(context);
    patterns.insert<linalg::ExtractSliceOfPadTensorSwapPattern>(
        context, [](tensor::ExtractSliceOp) { return false; });
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler

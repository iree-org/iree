// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/TensorExt/Transforms/Transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::TensorExt {

#define GEN_PASS_DEF_RESOLVESPARSEINTERFACEOPSPASS
#include "iree/compiler/Dialect/TensorExt/Transforms/Passes.h.inc"

namespace {

struct ResolveSparseInterfaceOpsPass final
    : public impl::ResolveSparseInterfaceOpsPassBase<
          ResolveSparseInterfaceOpsPass> {
  using Base::Base;

  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    populateSparseInterfaceRewritePatterns(patterns);

    // This pass only rewrites vector accesses to sparse memrefs; it must not
    // restructure control flow. Region simplification defaults to Aggressive,
    // which merges blocks even when no pattern applies.
    GreedyRewriteConfig config;
    config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Disabled);

    if (failed(applyPatternsGreedily(op, std::move(patterns), config))) {
      op->emitOpError("failed to resolve sparse interface operations");
      return signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::TensorExt

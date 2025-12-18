// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-fold-memref-alias-ops"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_IREECODEGENFOLDMEMREFALIASOPSPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

struct IREECodegenFoldMemRefAliasOpsPass final
    : impl::IREECodegenFoldMemRefAliasOpsPassBase<
          IREECodegenFoldMemRefAliasOpsPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    GreedyRewriteConfig config;
    config.setUseTopDownTraversal().setRegionSimplificationLevel(
        GreedySimplifyRegionLevel::Normal);

    RewritePatternSet patterns(&getContext());
    memref::populateFoldMemRefAliasOpPatterns(patterns);
    (void)applyPatternsGreedily(getOperation(), std::move(patterns), config);
  }
};

} // namespace
} // namespace mlir::iree_compiler

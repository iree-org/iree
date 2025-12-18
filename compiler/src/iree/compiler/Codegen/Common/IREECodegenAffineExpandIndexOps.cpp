// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "mlir/Dialect/Affine/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-affine-expand-index-ops"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_IREECODEGENAFFINEEXPANDINDEXOPSPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

struct IREECodegenAffineExpandIndexOpsPass final
    : impl::IREECodegenAffineExpandIndexOpsPassBase<
          IREECodegenAffineExpandIndexOpsPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    GreedyRewriteConfig config;
    config.setUseTopDownTraversal().setRegionSimplificationLevel(
        GreedySimplifyRegionLevel::Normal);

    RewritePatternSet patterns(&getContext());
    affine::populateAffineExpandIndexOpsPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config))) {
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler

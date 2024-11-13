// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-strip-compilation-info"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_STRIPCOMPILATIONINFOPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

struct StripFuncOpTranslationInfo final
    : OpInterfaceRewritePattern<mlir::FunctionOpInterface> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;
  LogicalResult matchAndRewrite(mlir::FunctionOpInterface funcOp,
                                PatternRewriter &rewriter) const final {
    if (!getTranslationInfo(funcOp))
      return failure();

    rewriter.modifyOpInPlace(funcOp, [&]() {
      // If the function has translation info, erase it.
      eraseTranslationInfo(funcOp);
    });

    return success();
  }
};

struct StripLinalgOpCompilationInfo final
    : OpInterfaceRewritePattern<linalg::LinalgOp> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;
  LogicalResult matchAndRewrite(linalg::LinalgOp linalgOp,
                                PatternRewriter &rewriter) const final {
    if (!getCompilationInfo(linalgOp) && !getLoweringConfig(linalgOp))
      return failure();

    rewriter.modifyOpInPlace(linalgOp, [&]() {
      if (getCompilationInfo(linalgOp)) {
        // Erase the compilation info configuration if it exists.
        eraseCompilationInfo(linalgOp);
      }
      if (getLoweringConfig(linalgOp)) {
        // Erase the lowering configuration from root operation if it
        // exists.
        eraseLoweringConfig(linalgOp);
      }
    });

    return success();
  }
};

struct StripCompilationInfoPass final
    : impl::StripCompilationInfoPassBase<StripCompilationInfoPass> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<StripFuncOpTranslationInfo>(ctx);
    patterns.add<StripLinalgOpCompilationInfo>(ctx);
    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};
} // namespace
} // namespace mlir::iree_compiler

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

struct StripFuncOpCompilationInfo final
    : OpInterfaceRewritePattern<mlir::FunctionOpInterface> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;
  LogicalResult matchAndRewrite(mlir::FunctionOpInterface funcOp,
                                PatternRewriter &rewriter) const final {
    rewriter.modifyOpInPlace(funcOp, [&]() {
      // If the function has translation info, erase it.
      if (getTranslationInfo(funcOp)) {
        eraseTranslationInfo(funcOp);
      }
    });

    return success();
  }
};

struct StripLinalgOpCompilationInfo final
    : OpInterfaceRewritePattern<linalg::LinalgOp> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;
  LogicalResult matchAndRewrite(linalg::LinalgOp linalgOp,
                                PatternRewriter &rewriter) const final {
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
    RewritePatternSet patterns(&getContext());
    patterns.add<StripFuncOpCompilationInfo>(&getContext());
    patterns.add<StripLinalgOpCompilationInfo>(&getContext());
    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};
} // namespace
} // namespace mlir::iree_compiler

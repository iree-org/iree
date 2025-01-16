// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/Transforms/Patterns.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::Util {

#define GEN_PASS_DEF_APPLYPATTERNSPASS
#include "iree/compiler/Dialect/Util/Transforms/Passes.h.inc"

namespace {

class ApplyPatternsPass
    : public impl::ApplyPatternsPassBase<ApplyPatternsPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<BuiltinDialect, func::FuncDialect, IREE::Util::UtilDialect>();
  }

  void runOnOperation() override {
    auto *context = &getContext();
    RewritePatternSet patterns(context);

    for (auto *dialect : context->getLoadedDialects()) {
      dialect->getCanonicalizationPatterns(patterns);
    }
    for (auto op : context->getRegisteredOperations()) {
      op.getCanonicalizationPatterns(patterns, context);
    }
    IREE::Util::populateCommonPatterns(context, patterns);

    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    if (failed(applyPatternsGreedily(getOperation(), frozenPatterns))) {
      getOperation()->emitError()
          << "failed to apply patterns, likely due to a bad pattern that "
             "causes an infinite fixed point iteration";
      return signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Util

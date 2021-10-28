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
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Util {

class ApplyPatternsPass
    : public PassWrapper<ApplyPatternsPass, OperationPass<void>> {
 public:
  StringRef getArgument() const override { return "iree-util-apply-patterns"; }

  StringRef getDescription() const override {
    return "Applies some risky/IREE-specific canonicalization patterns.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<BuiltinDialect, StandardOpsDialect, IREE::Util::UtilDialect>();
  }

  void runOnOperation() override {
    auto *context = &getContext();
    OwningRewritePatternList patterns(context);

    for (auto *dialect : context->getLoadedDialects()) {
      dialect->getCanonicalizationPatterns(patterns);
    }
    for (auto *op : context->getRegisteredOperations()) {
      op->getCanonicalizationPatterns(patterns, context);
    }
    IREE::Util::populateCommonPatterns(context, patterns);

    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(getOperation(), frozenPatterns))) {
      return signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<void>> createApplyPatternsPass() {
  return std::make_unique<ApplyPatternsPass>();
}

static PassRegistration<ApplyPatternsPass> pass;

}  // namespace Util
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_DECOMPOSELINALGGENERICPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

static bool hasSupportedDecompositionBody(linalg::GenericOp op) {
  if (llvm::any_of(op.getRegionOutputArgs(), [](BlockArgument arg) {
        return !arg.use_empty() || !arg.getType().isIntOrIndexOrFloat();
      })) {
    return false;
  }
  return llvm::all_of(op.getBody()->getOperations(), [](Operation &op) {
    return op.getNumRegions() == 0 && isMemoryEffectFree(&op);
  });
}

class DecomposeLinalgGenericPass final
    : public impl::DecomposeLinalgGenericPassBase<DecomposeLinalgGenericPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, linalg::LinalgDialect>();
  }
  void runOnOperation() override {
    SmallVector<Operation *> candidates;
    Operation *root = getOperation();
    root->walk([root, &candidates](linalg::GenericOp op) {
      if (op.getOperation() != root && hasSupportedDecompositionBody(op)) {
        candidates.push_back(op);
      }
    });

    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    linalg::populateDecomposeLinalgOpsPattern(patterns);
    linalg::GenericOp::getCanonicalizationPatterns(patterns, context);
    // Upstream decomposition inserts inputs before the output arguments, but
    // remaps uses of original output arguments without accounting for them.
    // It also remaps only direct uses in the payload block. Restrict the
    // worklist to regionless, effect-free bodies with unused output arguments
    // and output types supported by the decomposition's zero initialisation.
    // Decomposition and dead-operand removal preserve that property in newly
    // created operations; pre-existing consumers need the same eligibility
    // check.
    GreedyRewriteConfig config;
    config.setStrictness(GreedyRewriteStrictness::ExistingAndNewOps);
    if (failed(
            applyOpPatternsGreedily(candidates, std::move(patterns), config))) {
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler

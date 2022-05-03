// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/DestructiveUpdateUtils.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-rewrite-linalg-destructive-updates"

namespace mlir {
namespace iree_compiler {
namespace {
struct RewriteLinalgDestructiveUpdatesPass
    : public RewriteLinalgDestructiveUpdatesBase<
          RewriteLinalgDestructiveUpdatesPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<AffineDialect, IREE::Flow::FlowDialect, linalg::LinalgDialect,
                scf::SCFDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override;
};
}  // namespace

void RewriteLinalgDestructiveUpdatesPass::runOnOperation() {
  MLIRContext *context = &getContext();
  func::FuncOp funcOp = getOperation();
  if (!isEntryPoint(funcOp)) return;

  // Rewrite destructive updates and ensure no remaining store remains to the
  // full output.

  // TODO(#...): Use of the destructive update rewrite is a hack! There needs to
  // be a way to generate loops as we need, and use the tiled op generation
  // implementation. This should be possible after moving everything to use the
  // `TilingInterface`.
  if (failed(rewriteLinalgDestructiveUpdates(funcOp))) {
    funcOp->emitError("Failed to rewrite destructive updates in:\n")
        << *funcOp.getOperation();
    return signalPassFailure();
  }

  // After rewriting destructive updates, there might be uses of compute
  // operations only in `tensor.dim` ops. Resolve these.
  RewritePatternSet resolveDimOps(context);
  memref::populateResolveRankedShapeTypeResultDimsPatterns(resolveDimOps);
  if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(resolveDimOps)))) {
    return signalPassFailure();
  }
}

std::unique_ptr<OperationPass<func::FuncOp>>
createRewriteLinalgDestructiveUpdatesPass() {
  return std::make_unique<RewriteLinalgDestructiveUpdatesPass>();
}

}  // namespace iree_compiler
}  // namespace mlir

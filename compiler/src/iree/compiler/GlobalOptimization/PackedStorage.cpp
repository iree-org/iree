// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/GlobalOptimization/Passes.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::GlobalOptimization {

#define GEN_PASS_DEF_PACKSTORAGEPASS
#include "iree/compiler/GlobalOptimization/Passes.h.inc"

namespace {
struct PackStoragePass
    : public impl::PackStoragePassBase<PackStoragePass> {

  void runOnOperation() override;
};
} // namespace

void PackStoragePass::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  return signalPassFailure();
}

} // namespace mlir::iree_compiler::GlobalOptimization

// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Transforms.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-dispatch-creation-fold-transposes"

namespace mlir::iree_compiler::DispatchCreation {

#define GEN_PASS_DEF_FOLDTRANSPOSESPASS
#include "iree/compiler/DispatchCreation/Passes.h.inc"

namespace {
struct FoldTransposesPass
    : public impl::FoldTransposesPassBase<FoldTransposesPass> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

static bool foldTransposeControlFn(OpOperand *operand) {
  return IREE::Flow::isNonNullAndOutsideDispatch(
      {operand->getOwner(), operand->get().getDefiningOp()});
}

void FoldTransposesPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  IREE::LinalgExt::populateFuseLinalgExtOpsWithTransposes(
      patterns, foldTransposeControlFn);
  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    getOperation()->emitError("Failed to converge while folding transposes");
    return signalPassFailure();
  }
}
} // namespace mlir::iree_compiler::DispatchCreation

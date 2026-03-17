// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Transforms/Transforms.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_DECOMPOSEAFFINEOPSPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

struct DecomposeAffineOpsPass final
    : impl::DecomposeAffineOpsPassBase<DecomposeAffineOpsPass> {
  void runOnOperation() override;
};

} // namespace

void DecomposeAffineOpsPass::runOnOperation() {
  IRRewriter rewriter(&getContext());
  this->getOperation()->walk([&](affine::AffineApplyOp op) {
    rewriter.setInsertionPoint(op);
    reorderOperandsByHoistability(rewriter, op);
    (void)decompose(rewriter, op);
  });
}
} // namespace mlir::iree_compiler

// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/PassDetail.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "mlir/Dialect/Affine/Transforms/Transforms.h"

namespace mlir::iree_compiler {

namespace {

struct DecomposeAffineOpsPass
    : public DecomposeAffineOpsBase<DecomposeAffineOpsPass> {
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

std::unique_ptr<Pass> createDecomposeAffineOpsPass() {
  return std::make_unique<DecomposeAffineOpsPass>();
}
} // namespace mlir::iree_compiler

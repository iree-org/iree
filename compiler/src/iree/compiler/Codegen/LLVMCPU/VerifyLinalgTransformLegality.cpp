// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_VERIFYLINALGTRANSFORMLEGALITYPASS
#include "iree/compiler/Codegen/LLVMCPU/Passes.h.inc"

namespace {
struct VerifyLinalgTransformLegalityPass
    : impl::VerifyLinalgTransformLegalityPassBase<
          VerifyLinalgTransformLegalityPass> {
  void runOnOperation() override;
};
} // namespace

void VerifyLinalgTransformLegalityPass::runOnOperation() {
  auto funcOp = getOperation();
  // For now only check that there are no Linalg transform markers.
  auto walkResult = funcOp.walk([](linalg::LinalgOp op) -> WalkResult {
    if (op->hasAttr(LinalgTransforms::kLinalgTransformMarker)) {
      return op.emitError("expected no Linalg transform markers");
    }
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted()) {
    return signalPassFailure();
  }
}
} // namespace mlir::iree_compiler

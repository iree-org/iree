// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===-- GPUGeneralizeNamedOps.cpp - Pass to generalize named linalg ops --===//
//
// The pass is to generalize named linalg ops that are better as linalg.generic
// ops in IREE.
//
//===---------------------------------------------------------------------===//

#include "iree/compiler/Codegen/Common/GPU/PassDetail.h"
#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler {

namespace {
struct GPUGeneralizeNamedOpsPass
    : public GPUGeneralizeNamedOpsBase<GPUGeneralizeNamedOpsPass> {

  void runOnOperation() override;
};
} // namespace

void GPUGeneralizeNamedOpsPass::runOnOperation() {
  auto funcOp = getOperation();
  SmallVector<linalg::LinalgOp> namedOpCandidates;
  funcOp.walk([&](linalg::LinalgOp linalgOp) {
    if (isa<linalg::BatchMatmulTransposeBOp, linalg::MatmulTransposeBOp,
            linalg::VecmatOp, linalg::MatvecOp>(linalgOp))
      namedOpCandidates.push_back(linalgOp);
  });

  IRRewriter rewriter(&getContext());
  for (auto linalgOp : namedOpCandidates) {
    rewriter.setInsertionPoint(linalgOp);
    FailureOr<linalg::GenericOp> generalizedOp =
        linalg::generalizeNamedOp(rewriter, linalgOp);
    if (failed(generalizedOp)) {
      linalgOp->emitOpError("failed to generalize operation");
      return signalPassFailure();
    }
  }
}

std::unique_ptr<OperationPass<func::FuncOp>> createGPUGeneralizeNamedOpsPass() {
  return std::make_unique<GPUGeneralizeNamedOpsPass>();
}

} // namespace mlir::iree_compiler

// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- SPIRVGeneralizeNamedOps.cpp - Pass to generalize named linalg ops --===//
//
// The pass is to generalize named linalg ops that are better as linalg.generic
// ops in IREE
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/SPIRV/PassDetail.h"
#include "iree/compiler/Codegen/SPIRV/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler {

namespace {
struct SPIRVGeneralizeNamedOpsPass
    : public SPIRVGeneralizeNamedOpsBase<SPIRVGeneralizeNamedOpsPass> {

  void runOnOperation() override;
};
} // namespace

void SPIRVGeneralizeNamedOpsPass::runOnOperation() {
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

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createSPIRVGeneralizeNamedOpsPass() {
  return std::make_unique<SPIRVGeneralizeNamedOpsPass>();
}

} // namespace mlir::iree_compiler

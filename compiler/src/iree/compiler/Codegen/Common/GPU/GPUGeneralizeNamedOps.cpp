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

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUGENERALIZENAMEDOPSPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

LogicalResult
generalizeCandidates(MLIRContext *context,
                     ArrayRef<linalg::LinalgOp> namedOpCandidates) {
  IRRewriter rewriter(context);
  for (auto linalgOp : namedOpCandidates) {
    // Pass down lowering configuration. It can exist due to user set
    // configuration from the input.
    auto config = getLoweringConfig(linalgOp);
    rewriter.setInsertionPoint(linalgOp);
    FailureOr<linalg::GenericOp> generalizedOp =
        linalg::generalizeNamedOp(rewriter, linalgOp);
    if (failed(generalizedOp)) {
      linalgOp->emitOpError("failed to generalize operation");
      return failure();
    }
    if (config) {
      setLoweringConfig(*generalizedOp, config);
    }
  }
  return success();
}

namespace {
struct GPUGeneralizeNamedOpsPass final
    : impl::GPUGeneralizeNamedOpsPassBase<GPUGeneralizeNamedOpsPass> {
  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();
    SmallVector<linalg::LinalgOp> namedOpCandidates;
    funcOp.walk([&](linalg::LinalgOp linalgOp) {
      if (isa<linalg::BatchMatmulTransposeBOp, linalg::MatmulTransposeBOp,
              linalg::VecmatOp, linalg::MatvecOp>(linalgOp))
        namedOpCandidates.push_back(linalgOp);
    });

    if (failed(generalizeCandidates(&getContext(), namedOpCandidates))) {
      return signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::iree_compiler

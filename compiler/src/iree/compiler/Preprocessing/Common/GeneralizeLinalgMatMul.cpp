// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::Preprocessing {

#define GEN_PASS_DEF_GENERALIZELINALGMATMULPASS
#include "iree/compiler/Preprocessing/Common/Passes.h.inc" // IWYU pragma: export

namespace {

struct GeneralizeLinalgMatMulPass
    : public iree_compiler::Preprocessing::impl::GeneralizeLinalgMatMulPassBase<
          GeneralizeLinalgMatMulPass> {
  using iree_compiler::Preprocessing::impl::GeneralizeLinalgMatMulPassBase<
      GeneralizeLinalgMatMulPass>::GeneralizeLinalgMatMulPassBase;
  void runOnOperation() override {
    auto funcOp = getOperation();
    SmallVector<linalg::LinalgOp> namedOpCandidates;
    funcOp.walk([&](linalg::LinalgOp linalgOp) {
      if (!IREE::Flow::isNonNullAndOutsideDispatch(linalgOp)) {
        return;
      }
      if (isa_and_nonnull<linalg::MatmulOp, linalg::MatmulTransposeBOp,
                          linalg::BatchMatmulOp,
                          linalg::BatchMatmulTransposeBOp>(linalgOp)) {
        namedOpCandidates.push_back(linalgOp);
      }
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
};
} // namespace
} // namespace mlir::iree_compiler::Preprocessing

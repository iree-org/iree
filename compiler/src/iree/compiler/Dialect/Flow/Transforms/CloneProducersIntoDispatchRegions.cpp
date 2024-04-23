// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#define DEBUG_TYPE "iree-flow-clone-producers-into-dispatch-regions"

namespace mlir::iree_compiler::IREE::Flow {

#define GEN_PASS_DEF_CLONEPRODUCERSINTODISPATCHREGIONSPASS
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h.inc"

namespace {

struct CloneProducersIntoDispatchRegionsPass
    : public IREE::Flow::impl::CloneProducersIntoDispatchRegionsPassBase<
          CloneProducersIntoDispatchRegionsPass> {
  void runOnOperation() override {
    mlir::FunctionOpInterface funcOp = getOperation();
    IRRewriter rewriter(funcOp->getContext());

    funcOp->walk([&](DispatchRegionOp regionOp) {
      if (failed(cloneProducersToRegion(rewriter, regionOp)))
        return signalPassFailure();
    });

    funcOp->walk([&](Operation *op) {
      if (!isNonNullAndOutsideDispatch(op) || !isa<linalg::GenericOp>(op)) {
        return;
      }
      if (failed(wrapOpInDispatchRegion(rewriter, op))) {
        return signalPassFailure();
      }
    });

    // Rerun the cloning again to move still clonable operations into
    // dispatches.
    funcOp->walk([&](DispatchRegionOp regionOp) {
      if (failed(cloneProducersToRegion(rewriter, regionOp)))
        return signalPassFailure();
    });
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Flow

// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/DispatchCreation/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Iterators.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define DEBUG_TYPE                                                             \
  "iree-dispatch-creation-clone-producers-into-dispatch-regions"

namespace mlir::iree_compiler::DispatchCreation {

#define GEN_PASS_DEF_CLONEPRODUCERSINTODISPATCHREGIONSPASS
#include "iree/compiler/DispatchCreation/Passes.h.inc"

namespace {

struct CloneProducersIntoDispatchRegionsPass final
    : public impl::CloneProducersIntoDispatchRegionsPassBase<
          CloneProducersIntoDispatchRegionsPass> {
  using Base::Base;
  void runOnOperation() override {
    mlir::FunctionOpInterface funcOp = getOperation();
    IRRewriter rewriter(funcOp->getContext());

    IREE::Flow::ClonableIntoDispatchOptions options;
    options.aggressive = aggressive;
    funcOp->walk([&](IREE::Flow::DispatchRegionOp regionOp) {
      if (failed(cloneProducersToRegion(rewriter, regionOp, options)))
        return signalPassFailure();
    });

    funcOp->walk<WalkOrder::PostOrder, ReverseIterator>([&](Operation *op) {
      if (isOpTriviallyDead(op)) {
        return rewriter.eraseOp(op);
      }
    });

    funcOp->walk([&](Operation *op) {
      if (!IREE::Flow::isNonNullAndOutsideDispatch(op) ||
          !isa<linalg::GenericOp, IREE::LinalgExt::GatherOp>(op)) {
        return;
      }
      if (failed(IREE::Flow::wrapOpInDispatchRegion(rewriter, op))) {
        return signalPassFailure();
      }
    });

    // Rerun the cloning again to move still clonable operations into
    // dispatches.
    funcOp->walk([&](IREE::Flow::DispatchRegionOp regionOp) {
      if (failed(cloneProducersToRegion(rewriter, regionOp, options)))
        return signalPassFailure();
    });
  }
};

} // namespace

} // namespace mlir::iree_compiler::DispatchCreation

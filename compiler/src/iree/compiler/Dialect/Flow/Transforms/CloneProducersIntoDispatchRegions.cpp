// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <memory>

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-flow-clone-producers-into-dispatch-regions"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

namespace {
struct CloneProducersIntoDispatchRegionPass
    : public CloneProducersIntoDispatchRegionsBase<
          CloneProducersIntoDispatchRegionPass> {
  void runOnOperation() override;
};
} // namespace

void CloneProducersIntoDispatchRegionPass::runOnOperation() {
  FunctionOpInterface funcOp = getOperation();
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
}

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createCloneProducersIntoDispatchRegionsPass() {
  return std::make_unique<CloneProducersIntoDispatchRegionPass>();
}

} // namespace Flow
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir

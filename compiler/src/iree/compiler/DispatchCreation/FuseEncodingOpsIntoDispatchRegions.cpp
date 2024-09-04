// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Encoding/IR/EncodingDialect.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "iree/compiler/DispatchCreation/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-dispatch-creation-producers-into-dispatch-regions"

namespace mlir::iree_compiler::DispatchCreation {

#define GEN_PASS_DEF_FUSEENCODINGOPSINTODISPATCHREGIONSPASS
#include "iree/compiler/DispatchCreation/Passes.h.inc"

namespace {

// Return true if the op is fusable with a SetEncodingOp consumer.
// For now, just check if it is a LinalgOp.
static bool isFusableWithSetEncoding(Operation *op) {
  return isa<linalg::LinalgOp>(op);
}

struct FuseEncodingOpsIntoDispatchRegionsPass
    : public DispatchCreation::impl::FuseEncodingOpsIntoDispatchRegionsPassBase<
          FuseEncodingOpsIntoDispatchRegionsPass> {
  void runOnOperation() override {
    mlir::FunctionOpInterface funcOp = getOperation();
    MLIRContext *context = &getContext();
    IRRewriter rewriter(context);

    SmallVector<IREE::Encoding::SetEncodingOp> encodingOps;
    funcOp->walk([&](IREE::Encoding::SetEncodingOp encodingOp) {
      encodingOps.push_back(encodingOp);
    });

    for (IREE::Encoding::SetEncodingOp encodingOp : encodingOps) {
      OpOperand &operand = encodingOp.getSourceMutable();
      auto producerDispatch =
          operand.get().getDefiningOp<IREE::Flow::DispatchRegionOp>();
      // Nothing to fuse with, so wrap the `encodingOp` in its own dispatch.
      if (!producerDispatch) {
        if (failed(IREE::Flow::wrapOpInDispatchRegion(rewriter, encodingOp))) {
          return signalPassFailure();
        }
        continue;
      }

      // Find producer operation inside of the dispatch region to determine if
      // fusion is possible.
      auto result = cast<OpResult>(operand.get());
      auto dispatchReturnOp = cast<IREE::Flow::ReturnOp>(
          producerDispatch.getBody().front().getTerminator());
      auto producerInRegion = dyn_cast<OpResult>(
          dispatchReturnOp->getOperand(result.getResultNumber()));
      if (!producerInRegion) {
        if (failed(IREE::Flow::wrapOpInDispatchRegion(rewriter, encodingOp))) {
          return signalPassFailure();
        }
        continue;
      }

      // Place the op in its own dispatch region if fusion is not possible.
      if (!isFusableWithSetEncoding(producerInRegion.getOwner())) {
        if (failed(IREE::Flow::wrapOpInDispatchRegion(rewriter, encodingOp))) {
          return signalPassFailure();
        }
        continue;
      }
      // Fuse the `encodingOp` into the producer dispatch region.
      if (failed(moveFollowingOpIntoDispatchRegion(rewriter, encodingOp,
                                                   producerDispatch))) {
        return signalPassFailure();
      }
    }

    // Dynamic dims may have dominance issues after pulling encoding ops into
    // producer dispatch regions, so we need to resolve tensor.dim ops.
    RewritePatternSet patterns(context);
    memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::DispatchCreation

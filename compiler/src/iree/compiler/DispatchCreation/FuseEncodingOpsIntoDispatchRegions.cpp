// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Encoding/IR/EncodingDialect.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/FormDispatchRegions.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#define DEBUG_TYPE "iree-flow-clone-producers-into-dispatch-regions"

namespace mlir::iree_compiler::IREE::Flow {

#define GEN_PASS_DEF_FUSEENCODINGOPSINTODISPATCHREGIONSPASS
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h.inc"

namespace {

struct FuseEncodingOpsIntoDispatchRegionsPass
    : public IREE::Flow::impl::FuseEncodingOpsIntoDispatchRegionsPassBase<
          FuseEncodingOpsIntoDispatchRegionsPass> {
  void runOnOperation() override {
    mlir::FunctionOpInterface funcOp = getOperation();
    IRRewriter rewriter(funcOp->getContext());

    FormDispatchRegionsPassOptions options;
    funcOp->walk([&](IREE::Encoding::SetEncodingOp encodingOp) {
      OpOperand &operand = encodingOp.getSourceMutable();
      auto producerDispatch = operand.get().getDefiningOp<DispatchRegionOp>();
      // Nothing to fuse with, so wrap the `encodingOp` in its own dispatch.
      if (!producerDispatch) {
        if (failed(wrapOpInDispatchRegion(rewriter, encodingOp))) {
          return signalPassFailure();
        }
        return;
      }

      // Find producer operation inside of the dispatch region to determine if
      // fusion is possible.
      auto result = cast<OpResult>(operand.get());
      auto dispatchReturnOp = cast<IREE::Flow::ReturnOp>(
          producerDispatch.getBody().front().getTerminator());
      auto producerInRegion = dyn_cast<OpResult>(
          dispatchReturnOp->getOperand(result.getResultNumber()));
      if (!producerInRegion) {
        if (failed(wrapOpInDispatchRegion(rewriter, encodingOp))) {
          return signalPassFailure();
        }
        return;
      }
      // The indexing map of the producer needs to be identity along all
      // outerParallelLoops. Ideally, this should be taken from the fusion
      // group root, but just set to the outer parallel loops of the producer
      // op for now.
      auto outerParallelLoops =
          getOuterParallelLoops(producerInRegion.getOwner());
      // Place the op in its own dispatch region if fusion is not possible.
      // The real producer of the operand is the DispatchRegionOp, so override
      // with the producer op inside of the dispatch region.
      if (!isFusableWithProducer(operand, outerParallelLoops, options,
                                 /*overrideProducerResult=*/producerInRegion)) {
        if (failed(wrapOpInDispatchRegion(rewriter, encodingOp))) {
          return signalPassFailure();
        }
        return;
      }
      // Fuse the `encodingOp` into the producer dispatch region.
      if (failed(moveFollowingOpIntoDispatchRegion(rewriter, encodingOp,
                                                   producerDispatch))) {
        return signalPassFailure();
      }
    });
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Flow

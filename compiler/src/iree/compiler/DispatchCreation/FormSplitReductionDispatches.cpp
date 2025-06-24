// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "iree/compiler/DispatchCreation/Passes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#define DEBUG_TYPE "iree-dispatch-creation-form-split-reduction-dispatches"

namespace mlir::iree_compiler::DispatchCreation {

#define GEN_PASS_DEF_FORMSPLITREDUCTIONDISPATCHESPASS
#include "iree/compiler/DispatchCreation/Passes.h.inc"

namespace {

/// Pass declaration.
struct FormSplitReductionDispatchesPass final
    : public impl::FormSplitReductionDispatchesPassBase<
          FormSplitReductionDispatchesPass> {
  void runOnOperation() override;
};
} // namespace

static void getReductionDims(TilingInterface op, SmallVector<unsigned> &dims) {
  for (auto [i, loopType] : llvm::enumerate(op.getLoopIteratorTypes())) {
    if (loopType == utils::IteratorType::reduction) {
      dims.push_back(i);
    }
  }
}

static LogicalResult tileOpAndWrapInDispatch(RewriterBase &rewriter,
                                             TilingInterface op) {
  IRRewriter::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(op);

  // Tile the operation.
  scf::SCFTilingOptions options;
  options.setLoopType(scf::SCFTilingOptions::LoopType::ForallOp);
  options.setReductionTilingStrategy(
      ReductionTilingStrategy::PartialReductionOuterParallel);
  // TODO: fix this
  SmallVector<OpFoldResult> tileSizes;
  for (utils::IteratorType iteratorType : op.getLoopIteratorTypes()) {
    if (iteratorType == utils::IteratorType::parallel) {
      tileSizes.push_back(rewriter.getIndexAttr(0));
    } else {
      tileSizes.push_back(rewriter.getIndexAttr(128));
    }
  }
  options.setTileSizes(tileSizes);
  SmallVector<unsigned> reductionDims;
  getReductionDims(op, reductionDims);
  options.setReductionDims(reductionDims);
  FailureOr<scf::SCFTilingResult> result =
      scf::tileUsingSCF(rewriter, op, options);
  if (failed(result)) {
    return op.emitOpError("failed to tile using scf.forall");
  }
  rewriter.replaceOp(op, result->replacements);

  if (result->loops.size() != 1) {
    return op.emitOpError("expected single tiled loop");
  }

  // Wrap loop in `flow.dispatch.region`.
  LoopLikeOpInterface loop = result->loops[0];
  FailureOr<IREE::Flow::DispatchRegionOp> maybeRegionOp =
      IREE::Flow::wrapOpInDispatchRegion(rewriter, loop);
  if (failed(maybeRegionOp)) {
    return loop.emitOpError("failed to wrap in dispatch region");
  }
  return success();
}

void FormSplitReductionDispatchesPass::runOnOperation() {
  mlir::FunctionOpInterface funcOp = getOperation();
  MLIRContext *context = &getContext();
  IRRewriter rewriter(context);

  SmallVector<TilingInterface> reductionOps;
  funcOp.walk([&](TilingInterface tilingOp) {
    // TODO: implement better selection.
    if (llvm::count(tilingOp.getLoopIteratorTypes(),
                    utils::IteratorType::reduction)) {
      reductionOps.push_back(tilingOp);
    }
  });

  for (TilingInterface op : reductionOps) {
    if (failed(tileOpAndWrapInDispatch(rewriter, op))) {
      return signalPassFailure();
    }
  }

  (void)context;
}

} // namespace mlir::iree_compiler::DispatchCreation

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
  using Base::Base;
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
                                             TilingInterface op,
                                             OpFoldResult splitSize) {
  IRRewriter::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(op);

  scf::SCFTilingOptions options;
  options.setLoopType(scf::SCFTilingOptions::LoopType::ForallOp);
  options.setReductionTilingStrategy(
      ReductionTilingStrategy::PartialReductionOuterParallel);

  // Set tile sizes.
  SmallVector<OpFoldResult> tileSizes;
  auto zeroAttr = rewriter.getIndexAttr(0);
  for (utils::IteratorType iteratorType : op.getLoopIteratorTypes()) {
    if (iteratorType == utils::IteratorType::parallel) {
      tileSizes.push_back(zeroAttr);
    } else {
      tileSizes.push_back(splitSize);
    }
  }
  options.setTileSizes(tileSizes);
  SmallVector<unsigned> reductionDims;
  getReductionDims(op, reductionDims);
  if (reductionDims.size() != 1) {
    return op.emitError("op must only have one reduction dim");
  }
  options.setReductionDims(reductionDims);

  // Tile the operation.
  FailureOr<scf::SCFTilingResult> result =
      scf::tileUsingSCF(rewriter, op, options);
  if (failed(result)) {
    return op.emitOpError("failed to tile using scf.forall");
  }
  rewriter.replaceOp(op, result->replacements);

  // Didn't tile.
  if (result->loops.size() == 0) {
    return success();
  }
  if (result->loops.size() != 1) {
    return failure();
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
    if (failed(tileOpAndWrapInDispatch(
            rewriter, op, rewriter.getIndexAttr(splitSize.getValue())))) {
      return signalPassFailure();
    }
  }
}

} // namespace mlir::iree_compiler::DispatchCreation

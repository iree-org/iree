// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Encoding/IR/EncodingDialect.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/DispatchCreation/FusionUtils.h"
#include "iree/compiler/DispatchCreation/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/CSE.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-dispatch-creation-producers-into-dispatch-regions"

namespace mlir::iree_compiler::DispatchCreation {

#define GEN_PASS_DEF_FUSEENCODINGOPSINTODISPATCHREGIONSPASS
#include "iree/compiler/DispatchCreation/Passes.h.inc"

namespace {

/// Return true if the op is fusable with a SetEncodingOp consumer. The op's
/// containing dispatch must contain only reshape ops, encoding ops, linalg ops,
/// and attention ops. Non ShapedType ops (like arith ops, dim ops, etc.) are
/// also allowed.
/// TODO(#20179): It should be done by interface methods.
static bool isFusableWithSetEncoding(Operation *op) {
  auto parentRegion = op->getParentOfType<IREE::Flow::DispatchRegionOp>();
  // Make sure the dispatch region has only one block.
  if (!llvm::hasSingleElement(parentRegion.getBody())) {
    return false;
  }
  // Check that there are no ops other than reshapes and element-wise linalg
  // ops in the dispatch region.
  Block &regionBlock = parentRegion.getBody().getBlocks().front();
  for (Operation &op : regionBlock.getOperations()) {
    if (llvm::none_of(op.getResultTypes(), llvm::IsaPred<ShapedType>)) {
      continue;
    }
    if (!isa<tensor::CollapseShapeOp, tensor::ExpandShapeOp, tensor::EmptyOp,
             IREE::Encoding::SetEncodingOp, IREE::Encoding::UnsetEncodingOp,
             linalg::LinalgOp, IREE::LinalgExt::AttentionOp>(op)) {
      return false;
    }
  }
  return true;
}

struct FuseEncodingOpsIntoDispatchRegionsPass
    : public DispatchCreation::impl::FuseEncodingOpsIntoDispatchRegionsPassBase<
          FuseEncodingOpsIntoDispatchRegionsPass> {
  void runOnOperation() override {
    mlir::FunctionOpInterface funcOp = getOperation();
    MLIRContext *context = &getContext();
    IRRewriter rewriter(context);

    // Run CSE to eliminate common encoding ops.
    DominanceInfo domInfo;
    mlir::eliminateCommonSubExpressions(rewriter, domInfo, funcOp);

    SmallVector<IREE::Encoding::SetEncodingOp> encodingOps;
    funcOp->walk([&](IREE::Encoding::SetEncodingOp encodingOp) {
      if (IREE::Flow::isNonNullAndOutsideDispatch(encodingOp)) {
        encodingOps.push_back(encodingOp);
      }
    });

    for (IREE::Encoding::SetEncodingOp encodingOp : encodingOps) {
      OpOperand &operand = encodingOp.getSourceMutable();
      std::optional<std::pair<OpResult, SmallVector<Operation *>>>
          producerChain = getProducerDispatchValueAndOpChain(operand.get());
      // Nothing to fuse with, so wrap the `encodingOp` in its own dispatch.
      if (!producerChain) {
        continue;
      }

      // Find producer operation inside of the dispatch region to determine if
      // fusion is possible.
      OpResult result = producerChain->first;
      auto producerDispatch =
          result.getDefiningOp<IREE::Flow::DispatchRegionOp>();
      auto dispatchReturnOp = cast<IREE::Flow::ReturnOp>(
          producerDispatch.getBody().front().getTerminator());
      auto producerInRegion = dyn_cast<OpResult>(
          dispatchReturnOp->getOperand(result.getResultNumber()));
      if (!producerInRegion) {
        continue;
      }

      // Place the op in its own dispatch region if fusion is not possible.
      if (!isa<IREE::Encoding::MatmulKAttr>(
              encodingOp.getResultType().getEncoding()) &&
          !isFusableWithSetEncoding(producerInRegion.getOwner())) {
        continue;
      }
      // Fuse the `encodingOp` and the producer chain into the dispatch.
      SmallVector<Operation *> dispatchConsumers(
          llvm::reverse(producerChain->second));
      dispatchConsumers.push_back(encodingOp);
      for (Operation *consumer : dispatchConsumers) {
        FailureOr<IREE::Flow::DispatchRegionOp> fusedDispatch =
            moveFollowingOpIntoDispatchRegion(rewriter, consumer,
                                              producerDispatch);
        if (failed(fusedDispatch)) {
          return signalPassFailure();
        }
        producerDispatch = fusedDispatch.value();
      }
    }

    // Dynamic dims may have dominance issues after pulling encoding ops into
    // producer dispatch regions, so we need to resolve tensor.dim ops., Also
    // run the canonicalization patterns to remove redundantly returned results.
    GreedyRewriteConfig config;
    config.enableConstantCSE(false);
    RewritePatternSet patterns(context);
    IREE::Flow::DispatchRegionOp::getCanonicalizationPatterns(patterns,
                                                              context);
    memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns), config))) {
      return signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::DispatchCreation

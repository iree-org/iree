// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===--------------- SplitReduction.cpp ----------------------------===//
//
// Split reduction dimension to increase parallelism of a linalg operation.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Dialect/LinalgExt/Transforms/Transforms.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "iree/compiler/DispatchCreation/Passes.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::DispatchCreation {

#define GEN_PASS_DEF_SPLITREDUCTIONPASS
#include "iree/compiler/DispatchCreation/Passes.h.inc"

// TODO(thomasraoux): Move to attributes.
static llvm::cl::opt<int64_t>
    splitMatmulReductionRatio("iree-dispatch-creation-split-matmul-reduction",
                              llvm::cl::desc("split ratio"), llvm::cl::init(1));

static llvm::cl::opt<int64_t> splitArgmaxReductionThreshold(
    "iree-dispatch-creation-split-argmax-threshold",
    llvm::cl::desc(
        "Minimum size of the reduction dimension to trigger argmax split"),
    llvm::cl::init(128 * 1024));

static llvm::cl::opt<int64_t> splitArgmaxTileSize(
    "iree-dispatch-creation-split-argmax-tile-size",
    llvm::cl::desc("Tile size of the reduction dimension after splitting "
                   "(i.e., the chunk size)"),
    llvm::cl::init(128));

static llvm::cl::list<int64_t> topkSplitReductionRatio(
    "iree-dispatch-creation-topk-split-reduction",
    llvm::cl::desc("comma separated list of split ratios"),
    llvm::cl::CommaSeparated);

static LogicalResult splitReductionOnMatmul(
    RewriterBase &rewriter, linalg::MatmulOp op,
    linalg::ControlSplitReductionFn controlSplitReductionFn) {
  // Since user information about compilation are passed through attributes we
  // need to make sure to propagate those.
  SmallVector<NamedAttribute> prunedAttributeList =
      linalg::getPrunedAttributeList(op);

  // Do not transform the matmul ops that have encoded operands.
  auto hasEncoding = [](Type type) -> bool {
    auto rankedTensorType = dyn_cast<RankedTensorType>(type);
    return rankedTensorType && rankedTensorType.getEncoding();
  };
  if (llvm::any_of(op.getOperandTypes(), hasEncoding)) {
    return failure();
  }

  FailureOr<linalg::SplitReductionResult> result =
      linalg::splitReduction(rewriter, op, controlSplitReductionFn);
  if (failed(result)) {
    return failure();
  }

  result->splitLinalgOp->setAttrs(prunedAttributeList);
  return result;
}

namespace {
struct SplitReductionPass final
    : public impl::SplitReductionPassBase<SplitReductionPass> {
  void runOnOperation() override {
    if (splitMatmulReductionRatio.getValue() <= 1 &&
        topkSplitReductionRatio.empty() &&
        (splitArgmaxTileSize.getValue() <= 1 ||
         splitArgmaxReductionThreshold.getValue() <= 1)) {
      return;
    }

    MLIRContext *context = &getContext();
    auto funcOp = getOperation();

    SmallVector<linalg::MatmulOp> matmulCandidates;
    SmallVector<IREE::LinalgExt::TopkOp> topkCandidates;
    SmallVector<linalg::GenericOp> argmaxCandidates;

    IRRewriter rewriter(context);
    funcOp->walk([&](Operation *op) {
      TypeSwitch<Operation *>(op)
          .Case<linalg::MatmulOp>([&](auto matmulOp) {
            if (splitMatmulReductionRatio > 1) {
              matmulCandidates.push_back(matmulOp);
            }
          })
          .Case<IREE::LinalgExt::TopkOp>([&](auto topkOp) {
            if (!topkSplitReductionRatio.empty()) {
              topkCandidates.push_back(topkOp);
            }
          })
          .Case<linalg::GenericOp>([&](auto genericOp) {
            if (splitArgmaxTileSize > 1 &&
                IREE::LinalgExt::isArgmaxOp(genericOp)) {
              // Due to isArgmaxOp, we support exactly one reduction dimension.
              SmallVector<unsigned> dims;
              genericOp.getReductionDims(dims);
              unsigned reductionDim = dims[0];
              SmallVector<int64_t, 4> loopRanges =
                  genericOp.getStaticLoopRanges();
              int64_t reductionDimSize = loopRanges[reductionDim];
              if (reductionDimSize >= splitArgmaxReductionThreshold) {
                argmaxCandidates.push_back(genericOp);
              }
            }
          });
    });

    // Split matmul ops.
    auto matmulSplitReductionControlFn =
        [&](linalg::LinalgOp op) -> linalg::SplitReductionOptions {
      // For matmul make the new parallel dimension first so that it looks
      // like a batch_matmul and can follow the same codegen.
      return {splitMatmulReductionRatio, 0, /*innerParallel=*/false};
    };
    for (auto op : matmulCandidates) {
      (void)splitReductionOnMatmul(rewriter, op, matmulSplitReductionControlFn);
    }

    // Split argmax ops.
    auto argmaxSplitReductionControlFn =
        [&](linalg::LinalgOp op) -> linalg::SplitReductionOptions {
      return {splitArgmaxTileSize, op.getNumLoops() - 1,
              /*innerParallel=*/false};
    };
    for (auto op : argmaxCandidates) {
      (void)IREE::LinalgExt::splitArgmaxReduction(
          rewriter, op, argmaxSplitReductionControlFn);
    }

    // Split topk ops.
    IREE::LinalgExt::TopkSplitReductionControlFn topkSplitReductionControlFn =
        [&](int64_t splitReductionDepth) -> int64_t {
      SmallVector<int64_t> reductionRatios(topkSplitReductionRatio.begin(),
                                           topkSplitReductionRatio.end());
      if (splitReductionDepth >= reductionRatios.size()) {
        return -1;
      } else {
        return reductionRatios[splitReductionDepth];
      }
    };
    for (auto op : topkCandidates) {
      (void)splitReduction(rewriter, op, topkSplitReductionControlFn);
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::DispatchCreation

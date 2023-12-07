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

#include "iree-dialects/Dialect/LinalgExt/Passes/Passes.h"
#include "iree-dialects/Dialect/LinalgExt/Transforms/Transforms.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::Flow {

// TODO(thomasraoux): Move to attributes.
static llvm::cl::opt<int64_t>
    splitReductionRatio("iree-flow-split-matmul-reduction",
                        llvm::cl::desc("split ratio"), llvm::cl::init(1));

static llvm::cl::list<int64_t> topkSplitReductionRatio(
    "iree-flow-topk-split-reduction",
    llvm::cl::desc("comma separated list of split ratios"),
    llvm::cl::CommaSeparated);

namespace {
/// Pattern to wrap splitReduction transformation. This also propagates
/// attributes to allow compilation info attribute to not be lost.
struct LinalgSplitReduction
    : public OpInterfaceRewritePattern<linalg::LinalgOp> {
  LinalgSplitReduction(MLIRContext *context,
                       linalg::ControlSplitReductionFn controlSplitReductionFn,
                       LinalgExt::LinalgTransformationFilter f,
                       PatternBenefit benefit = 1)
      : OpInterfaceRewritePattern<linalg::LinalgOp>(context, benefit),
        controlSplitReductionFn(controlSplitReductionFn), filter(std::move(f)) {
  }

  LogicalResult matchAndRewrite(linalg::LinalgOp op,
                                PatternRewriter &rewriter) const override {
    std::vector<std::pair<StringAttr, Attribute>> attributes;
    // Since user information about compilation are passed through attributes we
    // need to make sure to propagate those.
    if (auto matmul = dyn_cast<linalg::MatmulOp>(op.getOperation())) {
      ArrayRef<StringRef> odsAttrs = matmul.getAttributeNames();
      for (NamedAttribute kv : op->getAttrs()) {
        if (!llvm::is_contained(odsAttrs, kv.getName().getValue())) {
          attributes.push_back(std::make_pair(kv.getName(), kv.getValue()));
        }
      }
    }

    FailureOr<linalg::LinalgOp> result = LinalgExt::splitReduction(
        rewriter, op, controlSplitReductionFn, filter);
    if (failed(result))
      return failure();
    // If any attributes needs to be propagated set it.
    for (std::pair<StringAttr, Attribute> &attrib : attributes) {
      result.value()->setAttr(attrib.first, attrib.second);
    }
    return result;
  }

private:
  linalg::ControlSplitReductionFn controlSplitReductionFn;
  LinalgExt::LinalgTransformationFilter filter;
};

struct SplitReductionPass : public SplitReductionBase<SplitReductionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    if (splitReductionRatio.getValue() <= 1 &&
        topkSplitReductionRatio.empty()) {
      return;
    }

    RewritePatternSet patterns(&getContext());
    patterns.add<LinalgSplitReduction>(
        &getContext(),
        [&](linalg::LinalgOp op) -> linalg::SplitReductionOptions {
          // For matmul make the new parallel dimension first so that it looks
          // like a batch_matmul and can follow the same codegen.
          if (isa<linalg::MatmulOp>(op))
            return {int64_t(splitReductionRatio), 0, /*innerParallel=*/false};
          // Currently disable spliting reduction for non-matmul op. This will
          // get enabled after once tests are ready.
          return {int64_t(0), 0, /*innerParallel=*/false};
        },
        LinalgExt::LinalgTransformationFilter(
            ArrayRef<StringAttr>{}, StringAttr::get(&getContext(), "SPLIT")));

    LinalgExt::TopkSplitReductionControlFn splitReductionFn =
        [&](int64_t splitReductionDepth) -> int64_t {
      SmallVector<int64_t> reductionRatios(topkSplitReductionRatio.begin(),
                                           topkSplitReductionRatio.end());
      if (splitReductionDepth >= reductionRatios.size()) {
        return -1;
      } else {
        return reductionRatios[splitReductionDepth];
      }
    };
    LinalgExt::populateTopkSplitReductionPattern(
        patterns, splitReductionFn,
        LinalgExt::LinalgTransformationFilter(
            ArrayRef<StringAttr>{},
            StringAttr::get(patterns.getContext(), "SPLIT_REDUCTION")));

    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }

    // Remove all the markers at the end.
    auto funcOp = getOperation();
    funcOp->walk([&](linalg::LinalgOp op) {
      op->removeAttr(IREE::LinalgExt::LinalgTransforms::kLinalgTransformMarker);
    });
    funcOp->walk([&](LinalgExt::LinalgExtOp op) {
      op->removeAttr(IREE::LinalgExt::LinalgTransforms::kLinalgTransformMarker);
      op->removeAttr(
          mlir::iree_compiler::IREE::LinalgExt::kSplitReductionDepthMarker);
    });
  }
};

} // namespace

std::unique_ptr<Pass> createSplitReductionPass() {
  return std::make_unique<SplitReductionPass>();
}

} // namespace mlir::iree_compiler::IREE::Flow

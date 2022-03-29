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

#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

// TODO(thomasraoux): Move to attributes.
static llvm::cl::opt<int64_t> splitReductionRatio(
    "iree-flow-split-matmul-reduction", llvm::cl::desc("split ratio"),
    llvm::cl::init(1));

namespace {
/// Pattern to wrap splitReduction transformation. This also propagates
/// attributes to allow compilation info attribute to not be lost.
struct LinalgSplitReduction
    : public OpInterfaceRewritePattern<linalg::LinalgOp> {
  LinalgSplitReduction(MLIRContext *context,
                       linalg::ControlSplitReductionFn controlSplitReductionFn,
                       linalg::LinalgTransformationFilter f,
                       PatternBenefit benefit = 1)
      : OpInterfaceRewritePattern<linalg::LinalgOp>(context, benefit),
        controlSplitReductionFn(controlSplitReductionFn),
        filter(std::move(f)) {}

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

    FailureOr<linalg::LinalgOp> result =
        splitReduction(rewriter, op, controlSplitReductionFn, filter);
    if (failed(result)) return failure();
    // If any attributes needs to be propagated set it.
    for (std::pair<StringAttr, Attribute> &attrib : attributes) {
      result.getValue()->setAttr(attrib.first, attrib.second);
    }
    return result;
  }

 private:
  linalg::ControlSplitReductionFn controlSplitReductionFn;
  linalg::LinalgTransformationFilter filter;
};

struct SplitReductionPass : public SplitReductionBase<SplitReductionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    if (splitReductionRatio <= 1) return;

    RewritePatternSet patterns(&getContext());
    patterns.add<LinalgSplitReduction>(
        &getContext(),
        [&](linalg::LinalgOp op) {
          // For matmul make the new parallel dimension first so that it looks
          // like a batch_matmul and can follow the same codegen.
          if (isa<linalg::MatmulOp>(op))
            return std::make_pair(int64_t(splitReductionRatio), 0);
          // Currently disable spliting reduction for non-matmul op. This will
          // get enabled after once tests are ready.
          return std::make_pair(int64_t(0), 0);
        },
        linalg::LinalgTransformationFilter(
            ArrayRef<StringAttr>{}, StringAttr::get(&getContext(), "SPLIT")));
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<Pass> createSplitReductionPass() {
  return std::make_unique<SplitReductionPass>();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

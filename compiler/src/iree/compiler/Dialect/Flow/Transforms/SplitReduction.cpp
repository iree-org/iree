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

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgExt/Passes/Passes.h"
#include "iree-dialects/Dialect/LinalgExt/Transforms/Transforms.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-flow-split-reduction"
namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

// TODO(thomasraoux): Move to attributes.
static llvm::cl::opt<int64_t> splitReductionRatio(
    "iree-flow-split-matmul-reduction", llvm::cl::desc("split ratio"),
    llvm::cl::init(1));

static llvm::cl::list<int64_t> topkSplitReductionRatio(
    "iree-flow-topk-split-reduction",
    llvm::cl::desc("comma separated list of split ratios"),
    llvm::cl::CommaSeparated);

constexpr int64_t kTopkSplitReductionRatioDefault = 8;
constexpr int64_t kNOTopkSplitReductionRatioDefault = -1;

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

    FailureOr<linalg::LinalgOp> result = LinalgExt::splitReduction(
        rewriter, op, controlSplitReductionFn, filter);
    if (failed(result)) return failure();
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

/// Sets the split reduction value for topk operation using a simple heuristic.
/// The goal here is to increase parallelism.
static int64_t topkSplitReduceRatio(int64_t splitReductionDepth,
                                    LinalgExt::TopkOp topkOp, int64_t kValue) {
  // Step 0. If user set the split reduction ratio use it.
  if (!topkSplitReductionRatio.empty()) {
    SmallVector<int64_t, 4> reductionRatios(topkSplitReductionRatio.begin(),
                                            topkSplitReductionRatio.end());
    if (splitReductionDepth >= reductionRatios.size()) {
      return kNOTopkSplitReductionRatioDefault;
    }
    return reductionRatios[splitReductionDepth];
  }

  // Step 1. Hard to predict of advantage of splitting reduction more than 2
  // depth.
  if (splitReductionDepth > 1) return kNOTopkSplitReductionRatioDefault;

  LLVM_DEBUG({
    llvm::dbgs() << "\n--- topkSplitReduceRatio started :";
    llvm::dbgs() << "splitReductionDepth=" << splitReductionDepth
                 << "\t kValue= " << kValue << "\n";
  });

  // Step 2. Set split value for for dynamic shape.
  auto inputType = topkOp.getInputs().front().getType();
  auto rankedType = dyn_cast<RankedTensorType>(inputType);
  if (!rankedType) {
    return kTopkSplitReductionRatioDefault;
    LLVM_DEBUG({
      llvm::dbgs() << "--- tensor is not ranked tensor, setting splitk value "
                      "as default: "
                   << kTopkSplitReductionRatioDefault << "\n";
    });
  }

  // Step 3. Find a value that balances the workload between the two kernels.
  // For example, workload 640 and select reduction value 5:
  // First workload : 128,  Second workload : 200
  int64_t lastDim = rankedType.getDimSize(rankedType.getRank() - 1);
  auto findSplitKValue = [](int64_t workload, int64_t kValue) {
    for (int64_t ki = 3; ki < kValue; ki++) {
      if (workload % ki != 0) continue;
      int workload2ndDispatchRegion = kValue * ki;
      int workload1stDispatchRegion = workload / ki;
      LLVM_DEBUG({
        llvm::dbgs() << "1stKernel = " << workload1stDispatchRegion << "\t"
                     << "2ndKernel = " << workload2ndDispatchRegion << "\n";
      });
      if (workload2ndDispatchRegion > workload1stDispatchRegion &&
          workload2ndDispatchRegion > (workload / (ki - 1))) {
        return ki;
      }
    }
    return kNOTopkSplitReductionRatioDefault;
  };

  int64_t splitValue = findSplitKValue(lastDim, kValue);

  LLVM_DEBUG({
    llvm::dbgs() << " ### Split-K[" << splitReductionDepth
                 << "] Value is set to: " << splitValue << "\n";
  });
  return splitValue;
}

/// Find there is a topk operation in the given operation.
static bool hasTopk(Operation *op) {
  auto result = op->walk(
      [&](LinalgExt::TopkOp topkOp) { return WalkResult::interrupt(); });
  return result.wasInterrupted();
}

struct SplitReductionPass : public SplitReductionBase<SplitReductionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    if (!hasTopk(getOperation()) && splitReductionRatio.getValue() <= 1) {
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

    LinalgExt::populateTopkSplitReductionPattern(
        patterns, topkSplitReduceRatio,
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

}  // namespace

std::unique_ptr<Pass> createSplitReductionPass() {
  return std::make_unique<SplitReductionPass>();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

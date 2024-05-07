// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/VectorExt/IR/VectorExtOps.h"
#include "iree/compiler/Codegen/Common/GPU/PassDetail.h"
#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

using namespace IREE::VectorExt;

namespace {

struct DecomposeThreadIds : public OpRewritePattern<ThreadIdsOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ThreadIdsOp threadIdsOp,
                                PatternRewriter &rewriter) const override {
    Location loc = threadIdsOp.getLoc();
    Value tid = threadIdsOp.getTid();
    NestedLayoutAttr layout =
        dyn_cast<NestedLayoutAttr>(threadIdsOp.getLayout());
    if (!layout) {
      return rewriter.notifyMatchFailure(threadIdsOp,
                                         "NestedLayoutAttr expected.");
    }

    SmallVector<Value> virtualTids;
    for (auto [size, stride] :
         llvm::zip(layout.getThreadsPerOuter(), layout.getThreadStrides())) {
      if (size == 1) {
        virtualTids.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
        continue;
      }

      // (tid floordiv stride) mod size
      AffineExpr tidExpr = rewriter.getAffineDimExpr(0);
      AffineMap virtualTidMap = AffineMap::get(/*dims=*/1, /*syms=*/0,
                                               tidExpr.floorDiv(stride) % size);
      Value virtualTid =
          rewriter.create<affine::AffineApplyOp>(loc, virtualTidMap, tid);
      virtualTids.push_back(virtualTid);
    }

    rewriter.replaceOp(threadIdsOp, virtualTids);
    return success();
  }
};

struct DecomposeSubgroupIds : public OpRewritePattern<SubgroupIdsOp> {
  using OpRewritePattern::OpRewritePattern;

  DecomposeSubgroupIds(MLIRContext *ctx, int64_t subgroupSize,
                       PatternBenefit benefit = 1)
      : OpRewritePattern(ctx, benefit), subgroupSize(subgroupSize) {}

  LogicalResult matchAndRewrite(SubgroupIdsOp subgroupIdsOp,
                                PatternRewriter &rewriter) const override {
    Location loc = subgroupIdsOp.getLoc();
    Value tid = subgroupIdsOp.getTid();
    NestedLayoutAttr layout =
        dyn_cast<NestedLayoutAttr>(subgroupIdsOp.getLayout());
    if (!layout) {
      return rewriter.notifyMatchFailure(subgroupIdsOp,
                                         "NestedLayoutAttr expected.");
    }

    SmallVector<Value> virtualTids;
    for (auto [size, stride] : llvm::zip(layout.getSubgroupsPerWorkgroup(),
                                         layout.getSubgroupStrides())) {
      if (size == 1) {
        virtualTids.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
        continue;
      }

      // (tid floordiv (stride * subgroupSize)) mod size
      AffineExpr tidExpr = rewriter.getAffineDimExpr(0);
      AffineMap virtualTidMap =
          AffineMap::get(/*dims=*/1, /*syms=*/0,
                         tidExpr.floorDiv(stride * subgroupSize) % size);
      Value virtualTid =
          rewriter.create<affine::AffineApplyOp>(loc, virtualTidMap, tid);
      virtualTids.push_back(virtualTid);
    }

    rewriter.replaceOp(subgroupIdsOp, virtualTids);
    return success();
  }

  int64_t subgroupSize;
};

struct GPUDecomposeVectorExtOps
    : public GPUDecomposeVectorExtOpsBase<GPUDecomposeVectorExtOps> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    FunctionOpInterface funcOp = getOperation();

    std::optional<int64_t> maybeSubgroupSize = getSubgroupSize(funcOp);
    if (!maybeSubgroupSize) {
      funcOp->emitOpError("Unable to query subgroup size");
      return signalPassFailure();
    }

    RewritePatternSet patterns(context);
    patterns.add<DecomposeThreadIds>(patterns.getContext());
    patterns.add<DecomposeSubgroupIds>(patterns.getContext(),
                                       maybeSubgroupSize.value());
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createGPUDecomposeVectorExtOps() {
  return std::make_unique<GPUDecomposeVectorExtOps>();
}

} // namespace mlir::iree_compiler

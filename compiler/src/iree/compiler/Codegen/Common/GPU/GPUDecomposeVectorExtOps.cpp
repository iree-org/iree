// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "iree/compiler/Codegen/Common/GPU/PassDetail.h"
#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

using namespace IREE::VectorExt;

namespace {

struct DecomposeThreadIds : public OpRewritePattern<ThreadIdsOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ThreadIdsOp threadIdsOp,
                                PatternRewriter &rewriter) const override {
    Value tid = threadIdsOp.getTid();
    NestedLayoutAttr layout =
        dyn_cast<NestedLayoutAttr>(threadIdsOp.getLayout());
    if (!layout) {
      return rewriter.notifyMatchFailure(threadIdsOp,
                                         "NestedLayoutAttr expected.");
    }

    SmallVector<Value> virtualTids;
    int64_t rank = layout.getRank();
    // The delinearized thread IDs are returned from outer most to inner most,
    // i.e. before applying the layout described dimensions ordering.
    SmallVector<Value> threadIds = layout.computeThreadIds(tid, rewriter);

    SmallVector<Value> filteredThreadIds;
    for (auto [id, active] : llvm::zip(llvm::drop_begin(threadIds, rank),
                                       layout.getThreadActiveIds())) {
      if (active)
        filteredThreadIds.push_back(id);
    }

    // Subgroup and thread (lane) indices normalized to the order in which
    // they are used by each dimension.
    virtualTids = llvm::to_vector(
        llvm::map_range(invertPermutationVector(layout.getThreadOrder()),
                        [&](int64_t i) { return filteredThreadIds[i]; }));

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
    Value tid = subgroupIdsOp.getTid();
    NestedLayoutAttr layout =
        dyn_cast<NestedLayoutAttr>(subgroupIdsOp.getLayout());
    if (!layout) {
      return rewriter.notifyMatchFailure(subgroupIdsOp,
                                         "NestedLayoutAttr expected.");
    }

    SmallVector<Value> virtualTids;
    // The delinearized thread IDs are returned from outer most to inner most,
    // i.e. before applying the layout described dimensions ordering.
    SmallVector<Value> threadIds = layout.computeThreadIds(tid, rewriter);

    SmallVector<Value> filteredSubgroupIds;
    for (auto [id, active] :
         llvm::zip(threadIds, layout.getSubgroupActiveIds())) {
      if (active)
        filteredSubgroupIds.push_back(id);
    }

    // Subgroup and thread (lane) indices normalized to the order in which
    // they are used by each dimension.
    virtualTids = llvm::to_vector(
        llvm::map_range(invertPermutationVector(layout.getSubgroupOrder()),
                        [&](int64_t i) { return filteredSubgroupIds[i]; }));
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

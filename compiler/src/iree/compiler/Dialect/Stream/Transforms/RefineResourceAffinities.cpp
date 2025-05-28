// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Stream/Analysis/Affinity.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-stream-refine-resource-affinities"

namespace mlir::iree_compiler::IREE::Stream {

#define GEN_PASS_DEF_REFINERESOURCEAFFINITIESPASS
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// --iree-stream-refine-resource-affinities
//===----------------------------------------------------------------------===//

static IREE::Stream::AffinityAttr
refineAllocationAffinity(Value allocationResult,
                         AffinityAnalysis &affinityAnalysis) {
  // Get all usage affinities for this allocation
  SmallVector<IREE::Stream::AffinityAttr> usageAffinities;
  if (!affinityAnalysis.tryLookupResourceUsageAffinity(allocationResult,
                                                       usageAffinities)) {
    return {};
  }
  // If no usage affinities found, keep current affinity
  if (usageAffinities.empty()) {
    return {};
  }
  // Join all usage affinities, will result in a DeviceOptimalAttr if the
  // affinities are across devices.
  IREE::Stream::AffinityAttr optimalAffinity = usageAffinities.front();
  for (size_t i = 1; i < usageAffinities.size(); ++i) {
    optimalAffinity = optimalAffinity.joinOR(usageAffinities[i]);
  }
  return optimalAffinity;
}

// Updates deallocation operations to match the affinity of their corresponding
// allocations.
static void
updateDeallocationsForAllocation(Value allocationResult,
                                 IREE::Stream::AffinityAttr newAffinity,
                                 PatternRewriter &rewriter) {
  // Find all users of the allocation and update any deallocation operations
  for (auto user : allocationResult.getUsers()) {
    if (auto deallocaOp = dyn_cast<IREE::Stream::ResourceDeallocaOp>(user)) {
      if (deallocaOp.getOperand() == allocationResult) {
        rewriter.modifyOpInPlace(
            deallocaOp, [&]() { deallocaOp.setAffinityAttr(newAffinity); });
      }
    }
  }
}

template <typename OpTy>
static LogicalResult
updateAllocationAffinity(OpTy allocOp, AffinityAnalysis &affinityAnalysis,
                         PatternRewriter &rewriter) {
  LLVM_DEBUG(llvm::dbgs() << "[refine-resource-affinities] Op: " << allocOp
                          << "\n");

  auto optimalAffinity =
      refineAllocationAffinity(allocOp.getResult(), affinityAnalysis);

  if (!optimalAffinity) {
    LLVM_DEBUG(llvm::dbgs()
               << "[refine-resource-affinities] Abort: no optimal affinity\n");
    return failure();
  }

  if (allocOp.getAffinityAttr() != optimalAffinity) {
    LLVM_DEBUG({
      llvm::dbgs()
          << "[refine-resource-affinities] Updating allocation affinity: from "
          << allocOp.getAffinityAttr() << " to " << optimalAffinity << "\n";
    });
    rewriter.modifyOpInPlace(
        allocOp, [&]() { allocOp.setAffinityAttr(optimalAffinity); });

    return success();
  }

  return failure();
}

struct RefineResourceAllocPattern
    : public OpRewritePattern<IREE::Stream::ResourceAllocOp> {
  RefineResourceAllocPattern(MLIRContext *context, AffinityAnalysis &analysis)
      : OpRewritePattern<IREE::Stream::ResourceAllocOp>(context),
        affinityAnalysis(analysis) {}

  LogicalResult matchAndRewrite(IREE::Stream::ResourceAllocOp allocOp,
                                PatternRewriter &rewriter) const override {
    // Alloc ops dont have deallocations.
    return updateAllocationAffinity(allocOp, affinityAnalysis, rewriter);
  }

private:
  AffinityAnalysis &affinityAnalysis;
};

struct RefineResourceAllocaPattern
    : public OpRewritePattern<IREE::Stream::ResourceAllocaOp> {
  RefineResourceAllocaPattern(MLIRContext *context, AffinityAnalysis &analysis)
      : OpRewritePattern<IREE::Stream::ResourceAllocaOp>(context),
        affinityAnalysis(analysis) {}

  LogicalResult matchAndRewrite(IREE::Stream::ResourceAllocaOp allocaOp,
                                PatternRewriter &rewriter) const override {
    if (failed(
            updateAllocationAffinity(allocaOp, affinityAnalysis, rewriter))) {
      return failure();
    }
    updateDeallocationsForAllocation(allocaOp.getResult(),
                                     allocaOp.getAffinityAttr(), rewriter);
    return success();
  }

private:
  AffinityAnalysis &affinityAnalysis;
};

struct RefineResourceAffinitiesPass
    : public IREE::Stream::impl::RefineResourceAffinitiesPassBase<
          RefineResourceAffinitiesPass> {
  using IREE::Stream::impl::RefineResourceAffinitiesPassBase<
      RefineResourceAffinitiesPass>::RefineResourceAffinitiesPassBase;

  void runOnOperation() override {
    auto moduleOp = getOperation();

    AffinityAnalysis affinityAnalysis(moduleOp);
    if (failed(affinityAnalysis.run())) {
      return signalPassFailure();
    }

    RewritePatternSet patterns(&getContext());
    patterns.add<RefineResourceAllocPattern>(&getContext(), affinityAnalysis);
    patterns.add<RefineResourceAllocaPattern>(&getContext(), affinityAnalysis);

    if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Stream

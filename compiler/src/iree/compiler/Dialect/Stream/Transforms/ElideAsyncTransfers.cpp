// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "iree/compiler/Dialect/Stream/Analysis/Affinity.h"
#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-stream-elide-async-transfers"

namespace mlir::iree_compiler::IREE::Stream {

#define GEN_PASS_DEF_ELIDEASYNCTRANSFERSPASS
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Transfer Elision Patterns
//===----------------------------------------------------------------------===//

struct ElideCompatibleTransferPattern
    : public OpRewritePattern<IREE::Stream::AsyncTransferOp> {
  AffinityAnalysis &affinityAnalysis;
  IREE::Stream::AffinityTopologyAttrInterface &topologyAttr;

  ElideCompatibleTransferPattern(
      MLIRContext *context, AffinityAnalysis &affinityAnalysis,
      IREE::Stream::AffinityTopologyAttrInterface &topologyAttr)
      : OpRewritePattern(context), affinityAnalysis(affinityAnalysis),
        topologyAttr(topologyAttr) {}

  LogicalResult matchAndRewrite(IREE::Stream::AsyncTransferOp transferOp,
                                PatternRewriter &rewriter) const override {
    auto source = transferOp.getSource();
    auto result = transferOp.getResult();
    auto sourceType = cast<IREE::Stream::ResourceType>(source.getType());
    auto resultType = cast<IREE::Stream::ResourceType>(result.getType());

    // Don't elide transfers that change lifetime (usage casts).
    if (sourceType.getLifetime() != resultType.getLifetime()) {
      return rewriter.notifyMatchFailure(
          transferOp, "not eliding lifetime-changing transfer");
    }

    // Get source and result affinities, either from explicit attributes or
    // from affinity analysis.
    auto sourceAffinityAttr = transferOp.getSourceAffinityAttr();
    auto resultAffinityAttr = transferOp.getResultAffinityAttr();

    if (!sourceAffinityAttr) {
      sourceAffinityAttr = affinityAnalysis.lookupResourceAffinity(source);
    }
    if (!resultAffinityAttr) {
      resultAffinityAttr = affinityAnalysis.lookupResourceAffinity(result);
    }

    if (!sourceAffinityAttr || !resultAffinityAttr) {
      return rewriter.notifyMatchFailure(transferOp,
                                         "missing affinity information");
    }
    // Check if we need to keep the transfer based on topology.
    if (!topologyAttr.hasUnifiedMemory(sourceAffinityAttr,
                                       resultAffinityAttr) &&
        !topologyAttr.hasTransparentAccess(sourceAffinityAttr,
                                           resultAffinityAttr)) {
      return rewriter.notifyMatchFailure(
          transferOp, "not eliding, transfer required by topology");
    }

    rewriter.replaceOp(transferOp, source);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// --iree-stream-elide-async-transfers
//===----------------------------------------------------------------------===//

// Elides transfers that are not required based on memory compatibility
// in the device topology.
struct ElideAsyncTransfersPass
    : public IREE::Stream::impl::ElideAsyncTransfersPassBase<
          ElideAsyncTransfersPass> {
  using ElideAsyncTransfersPassBase::ElideAsyncTransfersPassBase;
  void runOnOperation() override {
    auto moduleOp = getOperation();

    // Get the topology attribute from the module.
    auto topologyAttr =
        moduleOp->getAttrOfType<IREE::Stream::AffinityTopologyAttrInterface>(
            "stream.topology");
    if (!topologyAttr) {
      return;
    }

    AffinityAnalysis affinityAnalysis(moduleOp);
    if (failed(affinityAnalysis.run())) {
      moduleOp.emitError() << "failed to run affinity analysis";
      return signalPassFailure();
    }

    RewritePatternSet patterns(&getContext());
    patterns.add<ElideCompatibleTransferPattern>(
        &getContext(), affinityAnalysis, topologyAttr);
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));

    for (auto funcOp : moduleOp.getOps<CallableOpInterface>()) {
      if (auto *region = funcOp.getCallableRegion()) {
        (void)applyPatternsGreedily(*region, frozenPatterns);
      }
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Stream

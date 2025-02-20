// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "iree/compiler/Dialect/Stream/Analysis/Affinity.h"
#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-stream-elide-async-transfers"

namespace mlir::iree_compiler::IREE::Stream {

#define GEN_PASS_DEF_ELIDEASYNCTRANSFERSPASS
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// --iree-stream-elide-async-transfers
//===----------------------------------------------------------------------===//

static bool tryElideTransferOp(IREE::Stream::AsyncTransferOp transferOp,
                               AffinityAnalysis &analysis) {
  // Only operate on transfers that are from/to the same affinity.
  auto sourceAffinityAttr =
      analysis.lookupResourceAffinity(transferOp.getSource());
  auto targetAffinityAttr =
      analysis.lookupResourceAffinity(transferOp.getResult());
  if (sourceAffinityAttr != targetAffinityAttr) {
    return false;
  }

  // If the transfer is from/to staging we need to preserve it even if the
  // affinities are redundant.
  auto sourceType =
      cast<IREE::Stream::ResourceType>(transferOp.getSource().getType());
  auto targetType =
      cast<IREE::Stream::ResourceType>(transferOp.getResult().getType());
  if (sourceType.getLifetime() == IREE::Stream::Lifetime::Staging ||
      targetType.getLifetime() == IREE::Stream::Lifetime::Staging) {
    LLVM_DEBUG({
      llvm::dbgs() << "[elide-transfers] skipping staging transfer (";
      llvm::dbgs() << sourceType;
      llvm::dbgs() << " -> ";
      llvm::dbgs() << targetType;
      llvm::dbgs() << ")\n";
    });
    return false;
  }

  LLVM_DEBUG({
    llvm::dbgs() << "[elide-transfers] converting self transfer to clone (";
    llvm::dbgs() << sourceType;
    llvm::dbgs() << " -> ";
    llvm::dbgs() << targetType;
    llvm::dbgs() << ")\n";
  });

  OpBuilder builder(transferOp);
  auto cloneOp = builder.create<IREE::Stream::AsyncCloneOp>(
      transferOp.getLoc(), targetType, transferOp.getSource(),
      transferOp.getSourceSize(), transferOp.getResultSize(),
      targetAffinityAttr ? targetAffinityAttr : sourceAffinityAttr);
  cloneOp->setDialectAttrs(transferOp->getDialectAttrs());

  transferOp.getResult().replaceAllUsesWith(cloneOp.getResult());
  transferOp.erase();

  return true;
}

// Tries to elide copies nested within |region| when safe.
// Returns true if any ops were elided.
static bool tryElideAsyncTransfersInRegion(Region &region,
                                           AffinityAnalysis &analysis) {
  bool didChange = false;
  for (auto &block : region) {
    block.walk([&](Operation *op) {
      return TypeSwitch<Operation *, WalkResult>(op)
          .Case<IREE::Stream::AsyncTransferOp>([&](auto transferOp) {
            didChange = tryElideTransferOp(transferOp, analysis) || didChange;
            return WalkResult::advance();
          })
          .Default([&](auto *op) { return WalkResult::advance(); });
    });
  }
  return didChange;
}

// Elides async transfers that are not required based on analysis.
struct ElideAsyncTransfersPass
    : public IREE::Stream::impl::ElideAsyncTransfersPassBase<
          ElideAsyncTransfersPass> {
  void runOnOperation() override {
    auto moduleOp = getOperation();
    if (moduleOp.getBody()->empty()) {
      return;
    }

    // NOTE: we currently run this only once because all current inputs only
    // need that. If we end up with more complex programs that have transfers
    // that break analysis we may need multiple runs.
    unsigned maxIterationCount = 2;

    // Try analyzing the program and eliding the unneeded copies until we reach
    // a fixed point (no more copies can be elided).
    unsigned iterationCount = 0;
    for (; iterationCount < maxIterationCount; ++iterationCount) {
      // Perform whole-program analysis.
      // TODO(benvanik): reuse allocator across iterations.
      AffinityAnalysis analysis(moduleOp);
      if (failed(analysis.run())) {
        moduleOp.emitError() << "failed to solve for affinity analysis";
        return signalPassFailure();
      }

      // Apply analysis by eliding all transfers that are safe to elide.
      // If we can't elide any we'll consider the iteration complete and exit.
      bool didChange = false;
      for (auto funcOp : moduleOp.getOps<CallableOpInterface>()) {
        if (auto *region = funcOp.getCallableRegion()) {
          didChange =
              tryElideAsyncTransfersInRegion(*region, analysis) || didChange;
        }
      }
      if (!didChange) {
        break; // quiesced
      }
    }
    if (iterationCount == maxIterationCount) {
      // If you find yourself hitting this we can evaluate increasing the
      // iteration count (if it would eventually converge) or whether we allow
      // this to happen without remarking. For now all our programs converge in
      // just one or two iterations and this needs to be tuned with more complex
      // control flow.
      moduleOp.emitRemark()
          << "transfer elision pass failed to reach a fixed point after "
          << maxIterationCount
          << " iterations; unneeded transfers may be present";
      return;
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Stream

// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Stream/Analysis/Affinity.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "llvm/ADT/SetVector.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::Stream {

#define GEN_PASS_DEF_PLACETRANSFEREXECUTIONSPASS
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h.inc"

namespace {

struct PlaceTransferExecutionsPass
    : IREE::Stream::impl::PlaceTransferExecutionsPassBase<
          PlaceTransferExecutionsPass> {
  void runOnOperation() override {
    AffinityAnalysis affinityAnalysis(getOperation());
    if (failed(affinityAnalysis.run())) {
      return signalPassFailure();
    }

    getOperation().walk([&](IREE::Stream::AsyncTransferOp transferOp) {
      if (transferOp.getExecutionAffinityAttr()) {
        return;
      }

      SmallVector<IREE::Stream::AffinityAttr> pinnedAffinities;
      if (!affinityAnalysis.tryLookupPinnedAffinities(transferOp.getResult(),
                                                      pinnedAffinities)) {
        return;
      }
      llvm::SetVector<IREE::Stream::AffinityAttr> uniquePinnedAffinities(
          pinnedAffinities.begin(), pinnedAffinities.end());
      if (uniquePinnedAffinities.size() != 1) {
        return;
      }

      auto pinnedAffinityAttr = uniquePinnedAffinities.front();
      auto targetAffinityAttr = transferOp.getTargetAffinityAttr();
      if (!targetAffinityAttr ||
          !IREE::Stream::AffinityAttr::canExecuteTogether(pinnedAffinityAttr,
                                                          targetAffinityAttr)) {
        return;
      }

      auto derivedExecutionAffinityAttr =
          transferOp.getDefaultExecutionAffinityAttr();
      if (IREE::Stream::AffinityAttr::canExecuteTogether(
              derivedExecutionAffinityAttr, pinnedAffinityAttr)) {
        return;
      }

      transferOp.setExecutionAffinityAttr(pinnedAffinityAttr);
    });
  }
};

} // namespace
} // namespace mlir::iree_compiler::IREE::Stream

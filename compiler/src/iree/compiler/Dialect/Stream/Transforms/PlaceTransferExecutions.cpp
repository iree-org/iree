// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Stream/Analysis/Affinity.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-stream-place-transfer-executions"

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
      LLVM_DEBUG(llvm::dbgs() << "affinity analysis failed\n");
      return signalPassFailure();
    }

    getOperation().walk([&](IREE::Stream::AsyncTransferOp transferOp) {
      if (transferOp.getExecutionAffinityAttr()) {
        LLVM_DEBUG(llvm::dbgs()
                   << "skipping transfer with explicit execution affinity: "
                   << transferOp << "\n");
        return;
      }

      SmallVector<IREE::Stream::AffinityAttr> pinnedAffinities;
      if (!affinityAnalysis.tryLookupPinnedAffinities(transferOp.getResult(),
                                                      pinnedAffinities)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "skipping transfer with unknown pinned affinities: "
                   << transferOp << "\n");
        return;
      }
      if (pinnedAffinities.empty() || !llvm::all_equal(pinnedAffinities)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "skipping transfer without a unique pinned affinity: "
                   << transferOp << "\n");
        return;
      }

      auto pinnedAffinityAttr = pinnedAffinities.front();
      auto targetAffinityAttr = transferOp.getTargetAffinityAttr();
      if (!targetAffinityAttr ||
          !IREE::Stream::AffinityAttr::canExecuteTogether(pinnedAffinityAttr,
                                                          targetAffinityAttr)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "skipping transfer whose pinned affinity is incompatible "
                      "with its target: "
                   << transferOp << "\n");
        return;
      }

      auto derivedExecutionAffinityAttr =
          transferOp.getDefaultExecutionAffinityAttr();
      if (IREE::Stream::AffinityAttr::canExecuteTogether(
              derivedExecutionAffinityAttr, pinnedAffinityAttr)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "skipping transfer already executing on its pinned "
                      "affinity: "
                   << transferOp << "\n");
        return;
      }

      LLVM_DEBUG(llvm::dbgs() << "placing transfer on " << pinnedAffinityAttr
                              << ": " << transferOp << "\n");
      transferOp.setExecutionAffinityAttr(pinnedAffinityAttr);
    });
  }
};

} // namespace
} // namespace mlir::iree_compiler::IREE::Stream

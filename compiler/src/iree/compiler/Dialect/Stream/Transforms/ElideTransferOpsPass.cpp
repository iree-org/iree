// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-stream-elide-transfer-ops"

namespace mlir::iree_compiler::IREE::Stream {

#define GEN_PASS_DEF_ELIDETRANSFEROPSPASS
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h.inc"


namespace {

//===----------------------------------------------------------------------===//
// Transfer Elision Patterns
//===----------------------------------------------------------------------===//

struct ElideCompatibleTransferPattern
    : public OpRewritePattern<IREE::Stream::AsyncTransferOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(IREE::Stream::AsyncTransferOp transferOp,
                               PatternRewriter &rewriter) const override {
    LLVM_DEBUG({
        llvm::dbgs() << "[elide-transfers] Op: " << transferOp << "\n";
    });

    auto source = transferOp.getSource();
    auto result = transferOp.getResult();
    auto sourceType = cast<IREE::Stream::ResourceType>(source.getType());
    auto resultType = cast<IREE::Stream::ResourceType>(result.getType());

    // Don't elide transfers that change lifetime (usage casts)
    if (sourceType.getLifetime() != resultType.getLifetime()) {
      LLVM_DEBUG({
        llvm::dbgs() << "[elide-transfers] not eliding lifetime-changing transfer: "
                    << sourceType.getLifetime() << " -> "
                    << resultType.getLifetime() << "\n";
      });
      return failure();
    }

    // Get source and result affinities
    auto sourceAffinityAttr = transferOp.getSourceAffinityAttr();
    auto resultAffinityAttr = transferOp.getResultAffinityAttr();

    if (!sourceAffinityAttr || !resultAffinityAttr) {
      LLVM_DEBUG({
        llvm::dbgs() << "[elide-transfers] missing affinity information\n";
      });
      return failure();
    }

    // Check if the transfer is between compatible devices based on topology
    auto moduleOp = transferOp->getParentOfType<mlir::ModuleOp>();
    if (!moduleOp) {
      return failure();
    }

    // Get the topology attribute from the module
    auto topologyAttr = moduleOp->getAttrOfType<IREE::Stream::AffinityTopologyAttrInterface>("stream.topology");
    if (!topologyAttr) {
      LLVM_DEBUG({
        llvm::dbgs() << "[elide-transfers] no topology information found\n";
      });
      return failure();
    }

    // Check if we need to keep the transfer based on topology
    if (topologyAttr.requiresTransfer(sourceAffinityAttr, resultAffinityAttr)) {
      LLVM_DEBUG({
        llvm::dbgs() << "[elide-transfers] not eliding , transfer required by topology\n";
      });
      return failure();
    }

    LLVM_DEBUG({
        llvm::dbgs() << "[elide-transfers] eliding transfer between compatible devices\n";
    });

    rewriter.replaceOp(transferOp, source);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// --iree-stream-elide-transfer-ops
//===----------------------------------------------------------------------===//

// Elides transfers that are not required based on memory compatibility
// in the device topology.
struct ElideTransferOpsPass
    : public IREE::Stream::impl::ElideTransferOpsPassBase<
          ElideTransferOpsPass> {
  using ElideTransferOpsPassBase::ElideTransferOpsPassBase;
  void runOnOperation() override {
    auto moduleOp = getOperation();
    if (moduleOp.getBody()->empty()) {
      return;
    }

    RewritePatternSet patterns(&getContext());
    patterns.add<ElideCompatibleTransferPattern>(&getContext());

    if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
      moduleOp.emitError() << "failed to elide transfer operations";
      return signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Stream

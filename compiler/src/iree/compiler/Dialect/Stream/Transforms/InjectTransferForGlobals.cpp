// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <queue>

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Stream/Analysis/Affinity.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/Analysis/Explorer.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir::iree_compiler::IREE::Stream {

#define DEBUG_TYPE "iree-stream-inject-transfor-for-globals"

#define GEN_PASS_DEF_INJECTTRANSFERFORGLOBALSPASS
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h.inc"

namespace {
struct InjectTransferForGlobalsPass
    : public impl::InjectTransferForGlobalsPassBase<
          InjectTransferForGlobalsPass> {
  void runOnOperation() override {
    mlir::ModuleOp moduleOp = getOperation();
    IREE::Stream::AffinityAnalysis affinityAnalysis(moduleOp);
    if (failed(affinityAnalysis.run())) {
      LDBG() << "failed on running affinity analysis";
      return;
    }

    // TODO(hanchung): Can it share the same explorer with affinity analysis?
    Explorer explorer(moduleOp.getOperation(), TraversalAction::RECURSE);
    explorer.setOpInterfaceAction<mlir::FunctionOpInterface>(
        TraversalAction::RECURSE);
    explorer.setDialectAction<mlir::scf::SCFDialect>(TraversalAction::RECURSE);
    explorer.setDialectAction<IREE::Flow::FlowDialect>(
        TraversalAction::RECURSE);
    explorer.setDialectAction<IREE::Stream::StreamDialect>(
        TraversalAction::RECURSE);
    explorer.setDialectAction<IREE::Util::UtilDialect>(
        TraversalAction::RECURSE);
    explorer.initialize();

    SmallVector<IREE::Util::GlobalOpInterface> globals;
    explorer.forEachGlobal([&](const Explorer::GlobalInfo *globalInfo) {
      if (globalInfo->isIndirect || globalInfo->op.isGlobalMutable()) {
        return;
      }
      if (AffinityAttr::lookup(globalInfo->op)) {
        LDBG() << "skip because it has affinity attr";
        return;
      }
      if (llvm::any_of(globalInfo->getLoads(), [](auto loadOp) {
            return isa<IREE::Util::InitializerOp>(loadOp->getParentOp());
          })) {
        LDBG() << "skip " << globalInfo->op
               << " because there is a load in initiailizer";
        return;
      }
      globals.push_back(globalInfo->op);
    });

    IRRewriter rewriter(&getContext());
    for (IREE::Util::GlobalOpInterface globalOp : globals) {
      LDBG() << "start from " << globalOp;
      // Perform the update outside the walk. Otherwise, some uses are dropped
      // because new operations are created. (Is it really the root cause?)
      SmallVector<std::pair<OpOperand &, IREE::Stream::AffinityAttr>>
          updateList;
      for (auto loadOp : explorer.getGlobalInfo(globalOp)->getLoads()) {
        LDBG() << "value: " << loadOp.getLoadedGlobalValue();
        explorer.walkTransitiveUses(
            loadOp.getLoadedGlobalValue(), [&](OpOperand &operand) {
              LDBG() << "use: " << *operand.getOwner();
              TypeSwitch<Operation *>(operand.getOwner())
                  .Case<IREE::Flow::DispatchOp>([&](auto dispatchOp) {
                    llvm::SmallSetVector<AffinityAttr, 4> affinityAttrs;
                    for (auto dispatchOperand : dispatchOp.getArguments()) {
                      // To properly support this, we should have an analysis to
                      // say if the operand is from a global with unknown
                      // affinity or not.
                      if (dispatchOperand == operand.get()) {
                        continue;
                      }
                      IREE::Stream::AffinityAttr attr =
                          affinityAnalysis.lookupResourceAffinity(
                              dispatchOperand);
                      if (attr) {
                        affinityAttrs.insert(attr);
                      }
                    }
                    if (affinityAttrs.size() != 1) {
                      LLVM_DEBUG({
                        llvm::dbgs() << "multiple affinities: ";
                        llvm::interleaveComma(affinityAttrs, llvm::dbgs());
                        llvm::dbgs() << "\n";
                      });
                      return;
                    }
                    LDBG() << "updating the affinity to "
                           << *affinityAttrs.begin();
                    updateList.push_back({operand, *affinityAttrs.begin()});
                  })
                  .Default([&](Operation *) {});
              return WalkResult::advance();
            });
        for (auto [operand, affinityAttr] : updateList) {
          rewriter.setInsertionPointAfterValue(operand.get());
          Value transferOp = IREE::Flow::TensorTransferOp::create(
              rewriter, loadOp.getLoc(), operand.get(), affinityAttr);
          operand.assign(transferOp);
        }
      }
    }
  }
};
} // namespace

} // namespace mlir::iree_compiler::IREE::Stream

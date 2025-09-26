// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Stream/Analysis/Affinity.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/Analysis/Explorer.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
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
  void runOnOperation() override;
};
} // namespace

using AffinityUpdatePair = std::pair<OpOperand &, IREE::Stream::AffinityAttr>;

/// Returns the update list that transfers the uses of `globalOp` to proper
/// device, if the use have multiple resource affinities.
static SmallVector<AffinityUpdatePair>
collectUpdateListForGlobal(IREE::Util::GlobalOpInterface globalOp,
                           Explorer &explorer,
                           IREE::Stream::AffinityAnalysis &affinityAnalysis) {
  SmallVector<AffinityUpdatePair> result;
  const Explorer::GlobalInfo *globalInfo = explorer.getGlobalInfo(globalOp);
  LDBG() << "updating uses for " << globalInfo->op.getGlobalName();
  for (auto loadOp : globalInfo->getLoads()) {
    explorer.walkTransitiveUses(
        loadOp.getLoadedGlobalValue(), [&](OpOperand &operand) {
          LDBG() << "use: " << *operand.getOwner();
          TypeSwitch<Operation *>(operand.getOwner())
              .Case<IREE::Flow::DispatchOp>([&](auto dispatchOp) {
                llvm::SmallSetVector<AffinityAttr, 4> affinityAttrs;
                for (auto dispatchOperand : dispatchOp.getArguments()) {
                  // To properly support this, we should have an analysis to
                  // say if the operand is from a global with unknown
                  // affinity, but not only filter itself.
                  if (dispatchOperand == operand.get()) {
                    continue;
                  }
                  if (auto attr = affinityAnalysis.lookupResourceAffinity(
                          dispatchOperand)) {
                    affinityAttrs.insert(attr);
                  }
                }
                if (affinityAttrs.size() != 1) {
                  LDBG_OS([&](raw_ostream &os) {
                    os << "multiple/empty affinities: [";
                    llvm::interleaveComma(affinityAttrs, os);
                    os << "]";
                  });
                  return;
                }
                IREE::Stream::AffinityAttr executionAffinity =
                    *affinityAttrs.begin();
                if (executionAffinity ==
                    affinityAnalysis.lookupGlobalAffinity(globalOp)) {
                  return;
                }
                LDBG() << "updating the affinity to " << executionAffinity;
                result.push_back({operand, executionAffinity});
              })
              .Default([&](Operation *) {});
          return WalkResult::advance();
        });
  }
  return result;
}

void InjectTransferForGlobalsPass::runOnOperation() {
  mlir::ModuleOp moduleOp = getOperation();
  IREE::Stream::AffinityAnalysis affinityAnalysis(moduleOp);
  if (failed(affinityAnalysis.run())) {
    LDBG() << "failed on running affinity analysis";
    return;
  }

  Explorer explorer(moduleOp.getOperation(), TraversalAction::RECURSE);
  explorer.setOpInterfaceAction<mlir::FunctionOpInterface>(
      TraversalAction::RECURSE);
  explorer.setDialectAction<mlir::scf::SCFDialect>(TraversalAction::RECURSE);
  explorer.setDialectAction<IREE::Flow::FlowDialect>(TraversalAction::RECURSE);
  explorer.setDialectAction<IREE::Util::UtilDialect>(TraversalAction::RECURSE);
  explorer.initialize();

  SmallVector<IREE::Util::GlobalOpInterface> candidates;
  explorer.forEachGlobal([&](const Explorer::GlobalInfo *globalInfo) {
    if (globalInfo->isIndirect || globalInfo->op.isGlobalMutable()) {
      return;
    }
    candidates.push_back(globalInfo->op);
  });

  // Perform the update outside the walk. Otherwise, some uses are dropped
  // because new operations are created.
  SmallVector<AffinityUpdatePair> updateList;
  for (IREE::Util::GlobalOpInterface globalOp : candidates) {
    updateList.append(
        collectUpdateListForGlobal(globalOp, explorer, affinityAnalysis));
  }

  IRRewriter rewriter(&getContext());
  for (auto [operand, affinityAttr] : updateList) {
    rewriter.setInsertionPointAfterValue(operand.get());
    Value transferOp = IREE::Flow::TensorTransferOp::create(
        rewriter, operand.get().getLoc(), operand.get(), affinityAttr);
    operand.assign(transferOp);
  }
}

} // namespace mlir::iree_compiler::IREE::Stream

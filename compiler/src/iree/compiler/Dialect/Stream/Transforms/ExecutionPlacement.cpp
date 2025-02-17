// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Stream/Analysis/Partitioning.h"
#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-stream-execution-placement"

namespace mlir::iree_compiler::IREE::Stream {

#define GEN_PASS_DEF_EXECUTIONPLACEMENTPASS
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h.inc"

namespace {

LogicalResult handleAsyncTransferOp(IREE::Stream::AsyncTransferOp transfer) {
  if (transfer.getAffinityAttr())
    return success();

  auto operand = transfer.getSource();
  auto producer = operand.getDefiningOp();
  auto streamable =
      dyn_cast_or_null<IREE::Stream::StreamableOpInterface>(producer);
  auto srcAffinity =
      dyn_cast_or_null<IREE::Stream::AffinityOpInterface>(producer);

  bool hasOneUse = operand.hasOneUse();
  if (hasOneUse && streamable && srcAffinity && srcAffinity.getAffinityAttr()) {
    transfer.setAffinityAttr(srcAffinity.getAffinityAttr());
    return success();
  }

  if (transfer.getResultAffinityAttr()) {
    transfer.setAffinityAttr(transfer.getResultAffinityAttr());
    return success();
  }

  if (transfer.getSourceAffinityAttr()) {
    transfer.setAffinityAttr(transfer.getSourceAffinityAttr());
    return success();
  }

  transfer->emitOpError("Unknown src/dest affinity");
  return failure();
}

LogicalResult handleAsyncDispatchOp(IREE::Stream::AsyncDispatchOp dispatch) {
  if (dispatch.getAffinityAttr() || dispatch.preferCloneToConsumers()) {
    return success();
  }

  llvm::SetVector<AffinityAttr> affinities;
  for (auto operand : dispatch.getResourceOperands()) {
    auto affinityOp =
        dyn_cast_or_null<AffinityOpInterface>(operand.getDefiningOp());
    if (!affinityOp) {
      continue;
    }

    if (auto transfer = dyn_cast_or_null<IREE::Stream::AsyncTransferOp>(
            operand.getDefiningOp())) {
      affinities.insert(transfer.getResultAffinityAttr());
      continue;
    }

    if (auto stream =
            dyn_cast_or_null<StreamableOpInterface>(operand.getDefiningOp())) {
      if (stream.preferCloneToConsumers()) {
        continue;
      }

      affinities.insert(affinityOp.getAffinityAttr());
    }
  }

  if (affinities.size() == 0) {
    dispatch->emitOpError("No affinities found");
    return failure();
  }

  if (affinities.size() != 1) {
    dispatch->emitOpError("Multiple affinities found");
    return failure();
  }

  if (affinities[0] == nullptr) {
    dispatch->emitOpError("Null affinity selected");
    return failure();
  }

  dispatch.setAffinityAttr(affinities[0]);
  return success();
}

struct ExecutionPlacementPass
    : public IREE::Stream::impl::ExecutionPlacementPassBase<
          ExecutionPlacementPass> {
  void runOnOperation() override {

    auto result = getOperation()->walk([](Operation *op) -> WalkResult {
      if (auto transfer = dyn_cast_or_null<IREE::Stream::AsyncTransferOp>(op)) {
        if (failed(handleAsyncTransferOp(transfer))) {
          return WalkResult::interrupt();
        }
      }
      if (auto dispatch = dyn_cast_or_null<IREE::Stream::AsyncDispatchOp>(op)) {
        if (failed(handleAsyncDispatchOp(dispatch))) {
          return WalkResult::interrupt();
        }
      }

      return WalkResult::advance();
    });

    if (result.wasInterrupted()) {
      return signalPassFailure();
    }

    // It is possible for the results of a dispatch to be used across multiple
    // devices. This occurs when the output of the dispatch is constant. Cloning
    // with all required affinities and remapping provides a separate
    // implementation per device.
    llvm::SmallVector<IREE::Stream::AsyncDispatchOp> cloneDispatches;
    getOperation()->walk([&cloneDispatches](IREE::Stream::AsyncDispatchOp op) {
      if (op.preferCloneToConsumers() && !op.getAffinityAttr()) {
        cloneDispatches.push_back(op);
      }
    });

    for (auto dispatch : llvm::reverse(cloneDispatches)) {
      llvm::SetVector<AffinityAttr> affinities;
      for (auto &use : dispatch->getUses()) {
        if (auto affinity = dyn_cast<AffinityOpInterface>(use.getOwner())) {
          affinities.insert(affinity.getAffinityAttr());
        }
      }

      OpBuilder builder(dispatch.getOperation());
      for (auto affinity : affinities) {
        auto clone = builder.clone(*dispatch);
        auto cloneDispatch = dyn_cast<IREE::Stream::AsyncDispatchOp>(clone);
        cloneDispatch.setAffinityAttr(affinity);

        for (int i = 0, s = dispatch.getNumResults(); i < s; ++i) {
          Value result = dispatch.getResult(i);
          result.replaceUsesWithIf(
              cloneDispatch.getResult(i),
              [affinity](OpOperand &operand) -> bool {
                if (auto affinityOp =
                        dyn_cast<AffinityOpInterface>(operand.getOwner())) {
                  return affinityOp.getAffinityAttr() == affinity;
                }

                return false;
              });
        }
      }

      dispatch.erase();
    }

    // Cleanup the dead ops.
    auto *context = &getContext();
    RewritePatternSet patterns(context);
    for (auto *dialect : context->getLoadedDialects()) {
      dialect->getCanonicalizationPatterns(patterns);
    }

    for (auto op : context->getRegisteredOperations()) {
      op.getCanonicalizationPatterns(patterns, context);
    }

    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    if (failed(applyPatternsGreedily(getOperation(), frozenPatterns))) {
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler::IREE::Stream

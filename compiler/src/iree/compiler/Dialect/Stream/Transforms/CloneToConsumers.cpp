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
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-stream-clone-to-consumers"

namespace mlir::iree_compiler::IREE::Stream {

#define GEN_PASS_DEF_CLONETOCONSUMERSPASS
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// --iree-stream-clone-to-consumers
//===----------------------------------------------------------------------===//

// Returns true if the given |op| can be cloned as part of this pass.
static bool canCloneOp(Operation *op) {
  if (!op) {
    return false;
  } else if (auto streamableOp =
                 dyn_cast<IREE::Stream::StreamableOpInterface>(op)) {
    return streamableOp.preferCloneToConsumers();
  } else if (mlir::isPure(op)) {
    return true;
  }
  return false;
}

// TODO(benvanik): swap this with a full analysis to find values that are on
// edges that should be cloned. For example, a solver given
// `A -> B -> C -> device0|device1 -> D` could mark A, B, and C as needing
// clones for device 0 and 1. If ops that consume values but are still cloneable
// are added we may need that to clone entire trees in one shot instead of
// needing the fixed-point iteration. It would also let us clone across branch
// and function boundaries: this simple local analysis only works in a single
// basic block.
static bool tryCloneToConsumersInRegion(Region &region,
                                        AffinityAnalysis &analysis) {
  bool didChange = false;
  SmallVector<IREE::Stream::AffinityAttr> affinities; // cached, cleared in for
  DenseMap<Operation *, Operation *> clonedOps;       // cached, cleared in for
  for (auto &block : region.getBlocks()) {
    for (auto &op : block.getOperations()) {
      // Since we aren't using affinities here and just cloning the entire
      // use-def chain we can't share cloned ops across other ops. It's possible
      // to use analysis to determine if the op we cloned for shares the same
      // affinity but the outer fixed point iteration takes care of that by
      // running analysis again with the mutated IR.
      clonedOps.clear();
      for (auto &operand : op.getOpOperands()) {
        // This simple analysis is block local and is not be able to look across
        // branches or function calls.
        auto *definingOp = operand.get().getDefiningOp();
        if (!canCloneOp(definingOp)) {
          continue;
        }

        // If we already cloned the defining op for this operand we can reuse
        // it. Note that we can only reuse ops we cloned *for this op* as other
        // ops may have different affinities.
        auto result = cast<OpResult>(operand.get());
        auto clonedIt = clonedOps.find(definingOp);
        if (clonedIt != clonedOps.end()) {
          operand.set(clonedIt->second->getResult(result.getResultNumber()));
          didChange = true;
          continue;
        }

        // Get the affinities the operand is potentially produced for.
        // This will fail if analysis failed or may return the default affinity.
        affinities.clear();
        analysis.tryLookupResourceAffinity(result, affinities);

        // Clone the producer of the operand if it has multiple affinities and
        // replace our use with it.
        if (affinities.size() > 1) {
          OpBuilder builder(&op);
          auto *clonedOp = builder.clone(*definingOp);
          clonedOps.insert(std::make_pair(definingOp, clonedOp));
          operand.set(clonedOp->getResult(result.getResultNumber()));
          didChange = true;
          continue;
        }
      }
    }
  }
  return didChange;
}

// Clones ops that request cloning to consumers when their affinity is
// ambiguous.
struct CloneToConsumersPass
    : public IREE::Stream::impl::CloneToConsumersPassBase<
          CloneToConsumersPass> {
  GreedyRewriteConfig config;
  std::shared_ptr<const FrozenRewritePatternSet> patterns;

  LogicalResult initialize(MLIRContext *context) override {
    // Inherit the same config defaults from the upstream canonicalizer pass.
    config.useTopDownTraversal = true;
    config.enableRegionSimplification = mlir::GreedySimplifyRegionLevel::Normal;

    RewritePatternSet owningPatterns(context);
    for (auto *dialect : context->getLoadedDialects()) {
      dialect->getCanonicalizationPatterns(owningPatterns);
    }
    for (RegisteredOperationName op : context->getRegisteredOperations()) {
      op.getCanonicalizationPatterns(owningPatterns, context);
    }
    patterns =
        std::make_shared<FrozenRewritePatternSet>(std::move(owningPatterns));

    return success();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    if (moduleOp.getBody()->empty()) {
      return;
    }

    // NOTE: we currently run this only once because all current inputs only
    // need that. If we end up with more complex programs that have transfers
    // that break analysis we may need multiple runs.
    unsigned maxIterationCount = 32;

    // Try analyzing the program and cloning operations until all are used on
    // a single affinity.
    unsigned iterationCount = 0;
    for (; iterationCount < maxIterationCount; ++iterationCount) {
      // Perform whole-program analysis.
      // TODO(benvanik): reuse allocator across iterations.
      AffinityAnalysis analysis(moduleOp);
      if (failed(analysis.run())) {
        moduleOp.emitError() << "failed to solve for affinity analysis";
        return signalPassFailure();
      }

      // Apply analysis by cloning all ops we can with ambiguous affinities.
      // If we can't clone any we'll consider the iteration complete and exit.
      bool didChange = false;
      for (auto funcOp : moduleOp.getOps<CallableOpInterface>()) {
        bool funcDidChange = false;
        if (auto *region = funcOp.getCallableRegion()) {
          funcDidChange = tryCloneToConsumersInRegion(*region, analysis);
        }
        if (funcDidChange) {
          if (failed(
                  applyPatternsGreedily(getOperation(), *patterns, config))) {
            llvm::errs()
                << "canonicalization failed to converge; bad IR was produced\n";
            return signalPassFailure();
          }
        }
        didChange |= funcDidChange;
      }
      if (!didChange) {
        break;
      }
    }
    if (iterationCount == maxIterationCount) {
      // If you find yourself hitting this we can evaluate increasing the
      // iteration count (if it would eventually converge) or whether we allow
      // this to happen without remarking. For now all our programs converge in
      // just one or two iterations and this needs to be tuned with more complex
      // control flow.
      moduleOp.emitRemark()
          << "clone to consumers pass failed to reach a fixed point after "
          << maxIterationCount
          << " iterations; ambiguous affinity may be present";
      return;
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Stream

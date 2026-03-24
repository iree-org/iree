// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "iree/compiler/Codegen/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/WalkResult.h"

namespace mlir::iree_compiler::IREE::GPU {

#define GEN_PASS_DEF_WIDENINNERTILEDREDUCTIONPASS
#include "iree/compiler/Codegen/Dialect/GPU/Transforms/Passes.h.inc"

namespace {
struct WidenInnerTiledReductionPass final
    : impl::WidenInnerTiledReductionPassBase<WidenInnerTiledReductionPass> {
  void runOnOperation() override;
};
} // namespace

/// Returns the VDMFMA inner_tiled op directly inside forOp if it is a
/// candidate for accumulator widening, or nullptr otherwise.
static Codegen::InnerTiledOp findWidenCandidate(scf::ForOp forOp) {
  Codegen::InnerTiledOp candidate;
  forOp.getBody()->walk([&](Codegen::InnerTiledOp innerTiledOp) {
    auto vmmaAttr = dyn_cast<VirtualMMAAttr>(innerTiledOp.getKind());
    if (!vmmaAttr || !isVDMFMAIntrinsic(vmmaAttr.getIntrinsic())) {
      return WalkResult::advance();
    }
    auto semantics =
        dyn_cast<InnerTiledSemanticsAttr>(innerTiledOp.getSemantics());
    if (!semantics || !semantics.getDistributed() ||
        semantics.getPromotedAcc()) {
      return WalkResult::advance();
    }

    Value accOutput = innerTiledOp.getOutputs()[0];
    auto iterArgs = forOp.getRegionIterArgs();
    auto *it = llvm::find(iterArgs, accOutput);
    if (it == iterArgs.end()) {
      return WalkResult::advance();
    }
    unsigned accIdx = std::distance(iterArgs.begin(), it);
    auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    if (yieldOp.getOperand(accIdx) != innerTiledOp.getResult(0)) {
      return WalkResult::advance();
    }

    candidate = innerTiledOp;
    return WalkResult::interrupt();
  });
  return candidate;
}

void WidenInnerTiledReductionPass::runOnOperation() {
  MLIRContext *context = &getContext();
  IRRewriter rewriter(context);

  Codegen::InnerTiledOp innerTiledOp;
  getOperation()->walk([&](scf::ForOp forOp)) {}

  getOperation()->walk([&](scf::ForOp forOp) {
    Codegen::InnerTiledOp innerTiledOp = findWidenCandidate(forOp);
    if (!innerTiledOp) {
      return;
    }

    Value accOutput = innerTiledOp.getOutputs()[0];
    auto blockArg = cast<BlockArgument>(accOutput);
    unsigned accIdx = blockArg.getArgNumber() - forOp.getNumInductionVars();
    Location loc = forOp.getLoc();

    // Expand the init arg before the loop: vector<2xT> -> vector<4xT>.
    rewriter.setInsertionPoint(forOp);
    Value expandedInit =
        expandAccumulator(rewriter, loc, forOp.getInitArgs()[accIdx]);

    SmallVector<Value> newInitArgs(forOp.getInitArgs());
    newInitArgs[accIdx] = expandedInit;

    auto newForOp =
        scf::ForOp::create(rewriter, loc, forOp.getLowerBound(),
                           forOp.getUpperBound(), forOp.getStep(), newInitArgs);

    // Move the old body into the new ForOp. This replaces old block args
    // with new ones (the ACC block arg is now vector<4xT>).
    rewriter.mergeBlocks(forOp.getBody(), newForOp.getBody(),
                         newForOp.getBody()->getArguments());

    // Rebuild the inner_tiled op with widened semantics. After mergeBlocks,
    // innerTiledOp is in the new body with the vector<4xT> block arg as its
    // output, but its result type is still vector<2xT>. Creating a new op
    // infers the correct result type from the output operand types.
    auto oldSemantics =
        cast<InnerTiledSemanticsAttr>(innerTiledOp.getSemantics());
    auto newSemantics = InnerTiledSemanticsAttr::get(
        context, oldSemantics.getDistributed(), oldSemantics.getOpaque(),
        /*promotedAcc=*/true);

    rewriter.setInsertionPoint(innerTiledOp);
    auto newInnerTiledOp = Codegen::InnerTiledOp::create(
        rewriter, innerTiledOp.getLoc(), innerTiledOp.getInputs(),
        innerTiledOp.getOutputs(), innerTiledOp.getIndexingMaps(),
        innerTiledOp.getIteratorTypes(), innerTiledOp.getKind(), newSemantics,
        innerTiledOp.getPermutations());

    // Preserve any discardable attributes (e.g., lowering_config).
    newInnerTiledOp->setDiscardableAttrs(
        innerTiledOp->getDiscardableAttrDictionary());

    // Replace old inner_tiled with new inner_tiled. This updates the yield
    // operand from vector<2xT> to vector<4xT>, matching the new ForOp.
    rewriter.replaceOp(innerTiledOp, newInnerTiledOp);

    // Collapse the widened result after the loop: vector<4xT> -> vector<2xT>.
    rewriter.setInsertionPointAfter(newForOp);
    Value collapsed =
        collapseAccumulator(rewriter, loc, newForOp.getResult(accIdx));

    SmallVector<Value> replacements;
    for (auto [i, oldResult] : llvm::enumerate(forOp.getResults())) {
      if (i == accIdx) {
        replacements.push_back(collapsed);
      } else {
        replacements.push_back(newForOp.getResult(i));
      }
    }
    rewriter.replaceOp(forOp, replacements);
  });
}

} // namespace mlir::iree_compiler::IREE::GPU

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

namespace mlir::iree_compiler::IREE::GPU {

#define GEN_PASS_DEF_WIDENINNERTILEDREDUCTIONPASS
#include "iree/compiler/Codegen/Dialect/GPU/Transforms/Passes.h.inc"

namespace {
struct WidenInnerTiledReductionPass final
    : impl::WidenInnerTiledReductionPassBase<WidenInnerTiledReductionPass> {
  void runOnOperation() override;
};
} // namespace

void WidenInnerTiledReductionPass::runOnOperation() {
  MLIRContext *context = &getContext();
  IRRewriter rewriter(context);

  SmallVector<Codegen::InnerTiledOp> opsToPromote;
  getOperation()->walk([&](Codegen::InnerTiledOp innerTiledOp) {
    auto forOp = dyn_cast<scf::ForOp>(innerTiledOp->getParentOp());
    if (!forOp) {
      return;
    }

    auto vmmaAttr = dyn_cast<VirtualMMAAttr>(innerTiledOp.getKind());
    if (!vmmaAttr || !isVDMFMAIntrinsic(vmmaAttr.getIntrinsic())) {
      return;
    }

    auto semantics =
        dyn_cast<InnerTiledSemanticsAttr>(innerTiledOp.getSemantics());
    if (!semantics || !semantics.getDistributed()) {
      return;
    }

    if (semantics.getPromotedAcc()) {
      return;
    }

    // ACC output must be an iter-arg block argument of the enclosing for loop.
    Value accOutput = innerTiledOp.getOutputs()[0];
    auto blockArg = dyn_cast<BlockArgument>(accOutput);
    if (!blockArg || blockArg.getOwner() != forOp.getBody()) {
      return;
    }
    unsigned argNum = blockArg.getArgNumber();
    if (argNum < forOp.getNumInductionVars()) {
      return;
    }

    // The inner_tiled result must be what the yield returns for this iter-arg.
    auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    unsigned accIdx = argNum - forOp.getNumInductionVars();
    if (yieldOp.getOperand(accIdx) != innerTiledOp.getResult(0)) {
      return;
    }

    opsToPromote.push_back(innerTiledOp);
  });

  for (auto innerTiledOp : opsToPromote) {
    auto forOp = cast<scf::ForOp>(innerTiledOp->getParentOp());
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

    rewriter.mergeBlocks(forOp.getBody(), newForOp.getBody(),
                         newForOp.getBody()->getArguments());

    // Rebuild the inner_tiled op with promoted semantics. After mergeBlocks,
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

    // Collapse the promoted result after the loop: vector<4xT> -> vector<2xT>.
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
  }
}

} // namespace mlir::iree_compiler::IREE::GPU

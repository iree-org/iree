// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file implements patterns and passes for expanding scf.if's regions
// by pulling in ops before and after the scf.if op into both regions of the
// scf.if op.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SCF/Transforms.h"
#include "mlir/Dialect/SCF/Utils/AffineCanonicalizationUtils.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"

#define DEBUG_TYPE "iree-spirv-expand-if-region"

// Note: A pull request is open to upstream this pattern:
//   https://reviews.llvm.org/D117019
// After it lands, this pattern can be replaced.

namespace mlir {
namespace iree_compiler {

/// Pulls ops at the same nest level as the given `ifOp` into both regions of
/// the if `ifOp`.
static FailureOr<scf::IfOp> pullOpsIntoIfRegions(
    scf::IfOp ifOp, function_ref<bool(Operation *)> canMoveToRegion,
    RewriterBase &rewriter) {
  // Need to pull ops into both regions.
  if (!ifOp.elseBlock()) return failure();

  // Expect to only have one block in the enclosing region. This is the common
  // case for the level where we have structured control flows and it avoids
  // traditional control flow and simplifies the analysis.
  if (!llvm::hasSingleElement(ifOp->getParentRegion()->getBlocks()))
    return failure();

  SmallVector<Operation *> allOps;
  for (Operation &op : ifOp->getBlock()->without_terminator())
    allOps.push_back(&op);

  // If no ops before or after the if op, there is nothing to do.
  if (allOps.size() == 1) return failure();

  // Return true if the given `op` is in the same region as the scf.if op.
  auto isFromSameRegion = [ifOp](Operation *op) {
    return op->getParentRegion() == ifOp->getParentRegion();
  };

  // Collect ops before and after the scf.if op.
  auto allPrevOps = llvm::makeArrayRef(allOps).take_while(
      [&ifOp](Operation *op) { return op != ifOp.getOperation(); });
  auto allNextOps =
      llvm::makeArrayRef(allOps).drop_front(allPrevOps.size() + 1);

  // Find previous ops that cannot be moved into the regions.
  SetVector<Operation *> stickyPrevOps;
  // We cannot move the op into the region if
  // - The op is part of the backward slice for computing conditions.
  // - The op is part of the backward slice for a op that cannot move.
  if (Operation *condOp = ifOp.getCondition().getDefiningOp()) {
    getBackwardSlice(condOp, &stickyPrevOps, isFromSameRegion);
    stickyPrevOps.insert(condOp);
  }
  for (Operation *op : llvm::reverse(allPrevOps)) {
    if (stickyPrevOps.contains(op)) continue;
    if (!canMoveToRegion(op)) {
      getBackwardSlice(op, &stickyPrevOps, isFromSameRegion);
      stickyPrevOps.insert(op);  // Add the current op back.
    }
  }

  // Find out previous ops that cannot be moved into the regions.
  SetVector<Operation *> stickyNextOps;
  // We cannot move the op into the region if
  // - The op is part of the forward slice for a op that cannot move.
  for (Operation *op : allNextOps) {
    if (!canMoveToRegion(op)) {
      getForwardSlice(op, &stickyNextOps, isFromSameRegion);
      stickyNextOps.insert(op);  // Add the current op back.
    }
  }

  LLVM_DEBUG({
    llvm::dbgs() << "sticky previous ops:\n";
    for (Operation *op : stickyPrevOps) llvm::dbgs() << "  " << *op << "\n";
    llvm::dbgs() << "sticky next ops:\n";
    for (Operation *op : stickyNextOps) llvm::dbgs() << "  " << *op << "\n";
  });

  // NYI support for the case where we have sticky next ops. For such cases we
  // need to analyze their operands and figure out which are later coming from
  // the if region. It can be complicated; support this only truly needed.
  if (!stickyNextOps.empty()) return failure();

  // Now get the ops that can be moved into the regions.
  SmallVector<Operation *> prevOps, nextOps;
  for (Operation *op : allPrevOps)
    if (!stickyPrevOps.contains(op)) prevOps.push_back(op);
  for (Operation *op : allNextOps)
    if (!stickyNextOps.contains(op)) nextOps.push_back(op);

  LLVM_DEBUG({
    llvm::dbgs() << "previous ops to move:\n";
    for (Operation *op : prevOps) llvm::dbgs() << "  " << *op << "\n";
    llvm::dbgs() << "next ops to move:\n";
    for (Operation *op : nextOps) llvm::dbgs() << "  " << *op << "\n";
  });
  if (prevOps.empty() && nextOps.empty()) return failure();

  Operation *parentTerminator = ifOp->getBlock()->getTerminator();
  TypeRange resultTypes = ifOp.getResultTypes();
  if (!nextOps.empty()) {
    // The if op should yield the values used by the terminator.
    resultTypes = parentTerminator->getOperandTypes();
  }

  auto newIfOp = rewriter.create<scf::IfOp>(
      ifOp.getLoc(), resultTypes, ifOp.getCondition(), ifOp.elseBlock());

  auto pullIntoBlock = [&](Block *newblock, Block *oldBlock) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(newblock);
    BlockAndValueMapping bvm;

    // Clone all ops defined before the original if op.
    for (Operation *prevOp : prevOps) rewriter.clone(*prevOp, bvm);

    // Clone all ops defined inside the original if block.
    for (Operation &blockOp : oldBlock->without_terminator())
      rewriter.clone(blockOp, bvm);

    if (nextOps.empty()) {
      // If the if op needs to return value, its builder won't automatically
      // insert terminators. Just clone the old one here.
      if (newIfOp->getNumResults())
        rewriter.clone(*oldBlock->getTerminator(), bvm);
      return;
    }

    // There are ops after the old if op. Uses of the old if op should be
    // replaced by the cloned yield value.
    auto oldYieldOp = cast<scf::YieldOp>(oldBlock->back());
    for (int i = 0, e = ifOp->getNumResults(); i < e; ++i) {
      bvm.map(ifOp->getResult(i), bvm.lookup(oldYieldOp.getOperand(i)));
    }

    // Clone all ops defined after the original if op. While doing that, we need
    // to check whether the op is used by the terminator. If so, we need to
    // yield its result value at the proper index.
    SmallVector<Value> yieldValues(newIfOp.getNumResults());
    for (Operation *nextOp : nextOps) {
      rewriter.clone(*nextOp, bvm);
      for (OpOperand &use : nextOp->getUses()) {
        if (use.getOwner() == parentTerminator) {
          unsigned index = use.getOperandNumber();
          yieldValues[index] = bvm.lookup(use.get());
        }
      }
    }

    if (!yieldValues.empty()) {
      // Again the if builder won't insert terminators automatically.
      rewriter.create<scf::YieldOp>(ifOp.getLoc(), yieldValues);
    }
  };

  pullIntoBlock(newIfOp.thenBlock(), ifOp.thenBlock());
  pullIntoBlock(newIfOp.elseBlock(), ifOp.elseBlock());

  if (nextOps.empty()) {
    rewriter.replaceOp(ifOp, newIfOp->getResults());
  } else {
    // Update the terminator to use the new if op's results.
    rewriter.updateRootInPlace(parentTerminator, [&]() {
      parentTerminator->setOperands(newIfOp->getResults());
    });
    // We have pulled in all ops following the if op into both regions. Now
    // remove them all. Do this in the reverse order.
    for (Operation *op : llvm::reverse(nextOps)) rewriter.eraseOp(op);
    rewriter.eraseOp(ifOp);
  }
  for (Operation *op : llvm::reverse(prevOps)) rewriter.eraseOp(op);

  return newIfOp;
}

namespace {

class IfRegionExpansionPattern final : public OpRewritePattern<scf::IfOp> {
 public:
  IfRegionExpansionPattern(MLIRContext *context,
                           function_ref<bool(Operation *)> canMoveToRegion,
                           PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), canMoveToRegion(canMoveToRegion) {}

  LogicalResult matchAndRewrite(scf::IfOp ifOp,
                                PatternRewriter &rewriter) const override {
    return pullOpsIntoIfRegions(ifOp, canMoveToRegion, rewriter);
  }

 private:
  std::function<bool(Operation *)> canMoveToRegion;
};

struct SPIRVExpandIfRegionPass
    : public SPIRVExpandIfRegionBase<SPIRVExpandIfRegionPass> {
  void runOnOperation() override {
    FuncOp funcOp = getOperation();
    SmallVector<scf::IfOp> candidates;
    funcOp.walk([&](scf::IfOp ifOp) { candidates.push_back(ifOp); });

    RewritePatternSet patterns(funcOp.getContext());
    auto canMoveToRegion = [](Operation *op) {
      return !op->hasAttrOfType<UnitAttr>("sticky");
    };
    populateSPIRVExpandIfRegionPatterns(patterns, canMoveToRegion);
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    for (scf::IfOp ifOp : candidates) {
      // Apply transforms per op to avoid recursive behavior.
      (void)applyOpPatternsAndFold(ifOp, frozenPatterns, /*erased=*/nullptr);
    }
  }
};

}  // namespace

void populateSPIRVExpandIfRegionPatterns(
    RewritePatternSet &patterns,
    function_ref<bool(Operation *)> canMoveToRegion) {
  patterns.insert<IfRegionExpansionPattern>(patterns.getContext(),
                                            canMoveToRegion);
}

std::unique_ptr<OperationPass<FuncOp>> createSPIRVExpandIfRegionPass() {
  return std::make_unique<SPIRVExpandIfRegionPass>();
}

}  // namespace iree_compiler
}  // namespace mlir

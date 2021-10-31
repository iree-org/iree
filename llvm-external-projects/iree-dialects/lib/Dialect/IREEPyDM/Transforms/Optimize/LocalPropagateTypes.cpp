// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "../PassDetail.h"
#include "iree-dialects/Dialect/IREEPyDM/IR/Ops.h"
#include "iree-dialects/Dialect/IREEPyDM/Transforms/Passes.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
namespace PYDM = mlir::iree_compiler::IREE::PYDM;
using namespace PYDM;

using llvm::dbgs;
#define DEBUG_TYPE "pydm_opt"

namespace {

struct LocalPropagateTypesPass
    : public LocalPropagateTypesBase<LocalPropagateTypesPass> {
  void runOnOperation() override {
    // Prepare selected canonicalization patterns.
    auto *context = &getContext();
    RewritePatternSet canonicalizePatterns(context);
    ApplyBinaryOp::getCanonicalizationPatterns(canonicalizePatterns, context);
    ApplyCompareOp::getCanonicalizationPatterns(canonicalizePatterns, context);
    AsBoolOp::getCanonicalizationPatterns(canonicalizePatterns, context);
    BoxOp::getCanonicalizationPatterns(canonicalizePatterns, context);
    DynamicBinaryPromoteOp::getCanonicalizationPatterns(canonicalizePatterns,
                                                        context);
    PromoteNumericOp::getCanonicalizationPatterns(canonicalizePatterns,
                                                  context);
    UnboxOp::getCanonicalizationPatterns(canonicalizePatterns, context);
    FrozenRewritePatternSet frozenCanonicalizePatterns(
        std::move(canonicalizePatterns));
    GreedyRewriteConfig rewriterConfig;
    rewriterConfig.enableRegionSimplification = false;

    bool changed = false;
    for (int i = 0; i < 50; ++i) {
      LLVM_DEBUG(dbgs() << "--- Local type propagation iteration " << i << "\n");
      DominanceInfo domInfo(getOperation());
      changed = false;
      if (sinkStaticInfoCasts()) changed = true;
      sinkBlockArgumentFixups();
      applyPatternsAndFoldGreedily(getOperation(), frozenCanonicalizePatterns,
                                   rewriterConfig);
      if (!changed) break;
    }
  }

  // Moving things around the CFG often creates unresolved static info casts.
  // We sink these until they don't go any further (typically eliminating them).
  // Returns whether any changes were made.
  bool sinkStaticInfoCasts() {
    bool changed = false;
    auto allCasts = getStaticInfoCasts();
    for (StaticInfoCastOp castOp : allCasts) {
      Value fromValue = castOp.value();
      ObjectType fromType = castOp.value().getType().dyn_cast<ObjectType>();
      ObjectType toType = castOp.value().getType().dyn_cast<ObjectType>();
      if (!fromType || !toType) {
        LLVM_DEBUG(dbgs() << "Skipping non-object cast: " << castOp << "\n");
        continue;
      }
      // We only want to sink refinements (where we know more on input).
      if (fromType.getPrimitiveType() && !toType.getPrimitiveType()) {
        LLVM_DEBUG(dbgs() << "Skipping non-refinement cast: " << castOp
                          << "\n");
        continue;
      }

      bool eliminatedUses = true;
      SmallVector<OpOperand *> uses;
      for (auto &use : castOp.getResult().getUses()) {
        uses.push_back(&use);
      }
      for (auto *use : uses) {
        // Most of our ops which accept objects are internally tolerant of
        // receiving a refinement.
        // TODO: Replace this with an interface query.
        if (llvm::isa<BoxOp, DynamicBinaryPromoteOp, UnboxOp>(use->getOwner())) {
          use->set(fromValue);
          changed = true;
          LLVM_DEBUG(dbgs()
                     << "Sink refined type into: " << *use->getOwner() << "\n");
        } else if (auto branchOp =
                       llvm::dyn_cast<BranchOpInterface>(use->getOwner())) {
          // We just update it directly and rely on the fix-up step after
          // to smooth it all out.
          changed = true;
          use->set(fromValue);
          LLVM_DEBUG(dbgs()
                     << "Sink refined type into: " << *use->getOwner() << "\n");
        } else {
          eliminatedUses = false;
        }
      }

      if (eliminatedUses) {
        castOp->erase();
      }
    }
    return changed;
  }

  // We may have done type refinement on branch ops but not updated successors.
  // We fix these up en-masse by adding static info casts within successor
  // blocks as needed.
  void sinkBlockArgumentFixups() {
    SmallVector<Block *> blocks;
    for (auto &block : getOperation().body()) {
      blocks.push_back(&block);
    }

    for (auto *block : blocks) {
      LLVM_DEBUG(dbgs() << "  ++ Processing block " << block << "\n");
      // In the tree of permutations, who was the prime mover?
      Block *permutedParentBlock = permutedParentBlocks[block];
      if (!permutedParentBlock) permutedParentBlock = block;

      SmallVector<Block *> predecessors(block->getPredecessors());
      for (Block *predecessor : predecessors) {
        Operation *terminator = predecessor->getTerminator();
        LLVM_DEBUG(dbgs() << "  ++ Predecessor terminator: " << *terminator << "\n");
        auto branchOp = llvm::cast<BranchOpInterface>(terminator);
        Location loc = branchOp.getLoc();
        unsigned successorIndex = 0;
        for (Block *successor : terminator->getSuccessors()) {
          if (successor == block) break;
          successorIndex += 1;
        }
        auto successorOperands = branchOp.getSuccessorOperands(successorIndex);
        assert(successorOperands && "expected branch with explicit operands");
        bool mismatch = false;
        for (auto it :
             llvm::zip(block->getArgumentTypes(), *successorOperands)) {
          if (std::get<0>(it) != std::get<1>(it).getType()) {
            mismatch = true;
            break;
          }
        }
        if (!mismatch) continue;

        // See if we have already generated a permutation.
        SmallVector<Value> permutedArguments(*successorOperands);
        Block *existingPermutation =
            findPermutation(permutedParentBlock, permutedArguments);
        if (existingPermutation) {
          LLVM_DEBUG(dbgs() << "Fixup successor " << successorIndex
                            << " with existing permutation " << existingPermutation << "\n");
          branchOp->setSuccessor(existingPermutation, successorIndex);
          continue;
        }

        // Need to permute.
        LLVM_DEBUG(dbgs() << "Fixup successor " << successorIndex << " from "
                          << branchOp << "\n");
        // If here, we are instantiating a new block with specialized arguments.
        Block *newBlock = new Block();
        newBlock->insertBefore(block);
        for (auto newArgument : permutedArguments) {
          newBlock->addArgument(newArgument.getType());
        }
        branchOp->setSuccessor(newBlock, successorIndex);

        // Now cast back to the original block types, which then stand-in
        // when inlining.
        BlockAndValueMapping mapping;
        mapping.map(block, newBlock);
        SmallVector<Value> blockArguments(newBlock->getArguments().begin(),
                                          newBlock->getArguments().end());
        OpBuilder builder(newBlock, newBlock->begin());
        for (auto it : llvm::zip(block->getArguments(), blockArguments)) {
          Value origArgument = std::get<0>(it);
          Value &newArgument = std::get<1>(it);
          Type origType = origArgument.getType();
          if (newArgument.getType() != origType) {
            newArgument =
                builder.create<StaticInfoCastOp>(loc, origType, newArgument);
          }
          mapping.map(origArgument, newArgument);
        }

        inlineBlockInto(block, newBlock, mapping);
        permutedBlocks[permutedParentBlock].push_back(newBlock);
        permutedParentBlocks[newBlock] = permutedParentBlock;
      }
    }
  }

  Block *findPermutation(Block *permutedParentBlock,
                         SmallVectorImpl<Value> &incomingArguments) {
    for (Block *candidateBlock : permutedBlocks[permutedParentBlock]) {
      if (candidateBlock->getNumArguments() != incomingArguments.size())
        continue;
      bool matched = true;
      for (auto it :
           llvm::zip(incomingArguments, candidateBlock->getArgumentTypes())) {
        Type incomingType = std::get<0>(it).getType();
        Type candidateType = std::get<1>(it);
        if (incomingType != candidateType) {
          matched = false;
          break;
        }
      }
      if (matched) return candidateBlock;
    }
    return nullptr;
  }

  void inlineBlockInto(Block *fromBlock, Block *toBlock,
                       BlockAndValueMapping &mapping) {
    for (auto &op : *fromBlock) {
      toBlock->push_back(op.clone(mapping));
    }
  }

  SmallVector<StaticInfoCastOp> getStaticInfoCasts() {
    SmallVector<StaticInfoCastOp> results;
    getOperation()->walk([&](StaticInfoCastOp op) { results.push_back(op); });
    return results;
  }

  DenseMap<Block *, Block *> permutedParentBlocks;
  DenseMap<Block *, SmallVector<Block *>> permutedBlocks;
};

}  // namespace

std::unique_ptr<OperationPass<PYDM::FuncOp>>
PYDM::createLocalPropagateTypesPass() {
  return std::make_unique<LocalPropagateTypesPass>();
}

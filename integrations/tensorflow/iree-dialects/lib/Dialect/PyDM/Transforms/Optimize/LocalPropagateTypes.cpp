// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "../PassDetail.h"
#include "iree-dialects/Dialect/PyDM/IR/PyDMOps.h"
#include "iree-dialects/Dialect/PyDM/Transforms/Passes.h"
#include "iree-dialects/Dialect/PyDM/Utils/TypeInference.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

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
    PermutedTypePropagator propagator(context);

    RewritePatternSet canonicalizePatterns(context);
    ApplyBinaryOp::getCanonicalizationPatterns(canonicalizePatterns, context);
    ApplyCompareOp::getCanonicalizationPatterns(canonicalizePatterns, context);
    AsBoolOp::getCanonicalizationPatterns(canonicalizePatterns, context);
    BoxOp::getCanonicalizationPatterns(canonicalizePatterns, context);
    DynamicBinaryPromoteOp::getCanonicalizationPatterns(canonicalizePatterns,
                                                        context);
    NegOp::getCanonicalizationPatterns(canonicalizePatterns, context);
    PromoteNumericOp::getCanonicalizationPatterns(canonicalizePatterns,
                                                  context);
    SubscriptOp::getCanonicalizationPatterns(canonicalizePatterns, context);
    UnboxOp::getCanonicalizationPatterns(canonicalizePatterns, context);
    FrozenRewritePatternSet frozenCanonicalizePatterns(
        std::move(canonicalizePatterns));
    GreedyRewriteConfig rewriterConfig;
    // During the main fixpoint iteration, we cannot simplify regions because
    // our propagator is keeping a cache of permuted blocks (we can add blocks
    // but not remove until iteration is complete).
    rewriterConfig.enableRegionSimplification = false;

    bool changed = false;
    for (int i = 0; i < 500; ++i) {
      LLVM_DEBUG(dbgs() << "--- Local type propagation iteration " << i
                        << "\n");
      if (failed(applyPatternsAndFoldGreedily(
              getOperation(), frozenCanonicalizePatterns, rewriterConfig))) {
        emitError(getOperation().getLoc())
            << "failed to converge type propagation canonicalizations";
        return signalPassFailure();
      }
      changed = false;
      if (sinkStaticInfoCasts())
        changed = true;
      if (refineResultTypes())
        changed = true;
      permuteRefinedBlocks(propagator);
      if (!changed)
        break;
    }

    // Now that iteration is complete and we are no longer using the
    // propagator, do one final canonicalization with region simplification
    // enabled. This will prune out all of the excess blocks we created.
    // Note that because we are still using a subset of dialect-specific
    // patterns, this is less than a full canonicalization pass will do.
    rewriterConfig.enableRegionSimplification = true;
    if (failed(applyPatternsAndFoldGreedily(
            getOperation(), frozenCanonicalizePatterns, rewriterConfig))) {
      emitError(getOperation().getLoc())
          << "failed to converge type propagation canonicalizations";
      return signalPassFailure();
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
        if (auto refinable =
                llvm::dyn_cast<TypeRefinableOpInterface>(use->getOwner())) {
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

  bool refineResultTypes() {
    // Process any refinable ops we encountered in the main walk.
    bool changed = false;
    LLVM_DEBUG(dbgs() << "-- Refining result types:\n");
    getOperation()->walk([&](TypeRefinableOpInterface refinable) {
      Operation *refinableOp = refinable.getOperation();
      SmallVector<Type> originalResultTypes(refinableOp->getResultTypes());
      LLVM_DEBUG(dbgs() << "  refineResultTypes: " << *refinableOp << "\n");
      if (!refinable.refineResultTypes())
        return;
      LLVM_DEBUG(dbgs() << "  refineResultTypes changed results: "
                        << *refinableOp << "\n");
      OpBuilder builder(refinableOp);
      builder.setInsertionPointAfter(refinableOp);
      for (auto it :
           llvm::zip(originalResultTypes, refinableOp->getOpResults())) {
        Type origType = std::get<0>(it);
        OpResult result = std::get<1>(it);
        Type newType = result.getType();
        if (origType == newType)
          continue;
        // Insert a static info cast.
        // In the future, we could further query the use for refinable
        // support and skip creating the op.
        LLVM_DEBUG(dbgs() << "    changed result type " << origType << " -> "
                          << newType << "\n");

        Value newResult = result;
        Operation *replaceExcept = nullptr;
        // It is possible to refine from an object (boxed) to an unboxed type.
        // In order to keep the type algebra safe, we must box back.
        if (origType.isa<ObjectType>() && newType.isa<PYDM::PrimitiveType>()) {
          auto boxed = builder.create<BoxOp>(
              refinableOp->getLoc(),
              builder.getType<ObjectType>(newType.cast<PYDM::PrimitiveType>()),
              newResult);
          replaceExcept = boxed;
          newResult = boxed;
        }
        auto casted = builder.create<StaticInfoCastOp>(refinableOp->getLoc(),
                                                       origType, newResult);
        if (!replaceExcept)
          replaceExcept = casted;
        result.replaceAllUsesExcept(casted, replaceExcept);
        changed = true;
      }
    });
    return changed;
  }

  // We may have done type refinement on branch ops but not updated successors.
  // We fix these up en-masse by permuting the blocks using the propagator.
  // This is not merely mechanical: by iterating in this way with a permutation
  // cache, it is possible to refinements that include type cycles in the CFG.
  void permuteRefinedBlocks(PermutedTypePropagator &propagator) {
    SmallVector<Block *> blocks;
    for (auto &block : getOperation().body()) {
      blocks.push_back(&block);
    }

    // This loop adds new blocks so must iterate a snapshot.
    for (auto *block : blocks) {
      auto mismatchedPredecessors =
          propagator.findMismatchedBlockPredecessors(block);
      if (mismatchedPredecessors.empty())
        continue;
      LLVM_DEBUG(dbgs() << "  ++ Processing block " << block << " ("
                        << mismatchedPredecessors.size()
                        << " mismatched predecessors)\n");

      auto *parentInfo = propagator.lookupParentBlock(block);
      for (auto &mismatch : mismatchedPredecessors) {
        Location loc = mismatch.terminator.getLoc();
        Block *permutation =
            propagator.findBlockPermutation(parentInfo, mismatch.signature);
        if (!permutation) {
          LLVM_DEBUG(dbgs() << "  -- Creating new permutation for "
                            << mismatch.signature << "\n");
          permutation = propagator.createBlockPermutation(
              loc, parentInfo, mismatch.signature.getInputs(),
              [&](Block *newBlock, Block *origBlock,
                  BlockAndValueMapping &mapping) {
                OpBuilder builder(newBlock, newBlock->begin());
                for (auto it : llvm::zip(newBlock->getArguments(),
                                         origBlock->getArguments())) {
                  Value newArgument = std::get<0>(it);
                  Type newType = newArgument.getType();
                  Value origArgument = std::get<1>(it);
                  Type origType = origArgument.getType();
                  if (newType != origType) {
                    newArgument = builder.create<StaticInfoCastOp>(
                        loc, origType, newArgument);
                    LLVM_DEBUG(dbgs() << "  -- Adding cast " << newType
                                      << " -> " << origType << "\n");
                  }
                  mapping.map(origArgument, newArgument);
                }
              });
        } else {
          LLVM_DEBUG(dbgs() << "  -- Re-using existing permutation for "
                            << mismatch.signature << "\n");
        }
        mismatch.terminator->setSuccessor(permutation, mismatch.successorIndex);
      }
    }
  }

  SmallVector<StaticInfoCastOp> getStaticInfoCasts() {
    SmallVector<StaticInfoCastOp> results;
    getOperation()->walk([&](StaticInfoCastOp op) { results.push_back(op); });
    return results;
  }
};

} // namespace

std::unique_ptr<OperationPass<PYDM::FuncOp>>
PYDM::createLocalPropagateTypesPass() {
  return std::make_unique<LocalPropagateTypesPass>();
}

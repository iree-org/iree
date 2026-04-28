// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCF.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFAttrs.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFTypes.h"
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Passes.h"
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Transforms.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/RegionUtils.h"

namespace mlir::iree_compiler::IREE::PCF {

#define GEN_PASS_DEF_TESTFOLDFORALLINTOPCFLOOPPASS
#include "iree/compiler/Codegen/Dialect/PCF/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// scf.forall + pcf.loop -> pcf.generic fold helpers
//===----------------------------------------------------------------------===//

/// Validates that the forall terminator has the expected structure for folding:
/// - All ops are tensor.parallel_insert_slice.
/// - All insert sources come from the same pcf.loop result.
/// - All insert destinations are forall shared_outs.
/// Returns the found pcf.loop on success.
static FailureOr<PCF::LoopOp> matchFoldTerminator(scf::ForallOp forallOp) {
  auto terminator =
      cast<scf::InParallelOp>(forallOp.getRegion().front().getTerminator());

  PCF::LoopOp foundLoop = nullptr;

  for (Operation &op : terminator.getYieldingOps()) {
    // All ops must be tensor.parallel_insert_slice.
    auto insertSliceOp = dyn_cast<tensor::ParallelInsertSliceOp>(&op);
    if (!insertSliceOp) {
      return failure();
    }

    // Source must be from a pcf.loop result.
    auto loopResult = dyn_cast<OpResult>(insertSliceOp.getSource());
    if (!loopResult) {
      return failure();
    }

    auto currentLoop = dyn_cast<PCF::LoopOp>(loopResult.getOwner());
    if (!currentLoop) {
      return failure();
    }

    // All inserts must reference the same pcf.loop.
    if (foundLoop && foundLoop != currentLoop) {
      return failure();
    }
    foundLoop = currentLoop;

    // Destination must be a shared_out of the forall.
    auto destArg = dyn_cast<BlockArgument>(insertSliceOp.getDest());
    if (!destArg || destArg.getOwner() != &forallOp.getRegion().front()) {
      return failure();
    }

    // Verify it's a shared_out (comes after induction vars).
    if (destArg.getArgNumber() < forallOp.getRank()) {
      return failure();
    }
  }

  if (!foundLoop) {
    return failure();
  }

  return foundLoop;
}

/// Validates the pcf.loop structure for folding:
/// - Single count argument (linearized).
/// - Loop is last op before terminator.
static LogicalResult matchFoldPCFLoop(scf::ForallOp forallOp,
                                      PCF::LoopOp loopOp) {
  // Single count argument required.
  if (loopOp.getCount().size() != 1) {
    return failure();
  }

  // Loop must be last op before terminator.
  Operation *lastOp =
      forallOp.getRegion().front().getTerminator()->getPrevNode();
  if (lastOp != loopOp) {
    return failure();
  }

  return success();
}

/// Validates pcf.loop region ref args:
/// - All users are pcf.write_slice ops.
/// - Ref args have SyncOnReturnAttr sync scope.
static LogicalResult matchFoldWriteSlices(PCF::LoopOp loopOp) {
  for (BlockArgument refArg : loopOp.getRegionRefArgs()) {
    // Check sync scope is sync_on_return.
    auto srefType = cast<PCF::ShapedRefType>(refArg.getType());
    Attribute syncScope = srefType.getSyncScope();
    if (!isa_and_nonnull<PCF::SyncOnReturnAttr>(syncScope)) {
      return failure();
    }

    // All users must be write_slice ops.
    for (Operation *user : refArg.getUsers()) {
      auto writeOp = dyn_cast<PCF::WriteSliceOp>(user);
      if (!writeOp) {
        return failure();
      }
    }
  }

  return success();
}

/// Computes the iteration count per dimension from forall bounds.
/// Returns ceildiv(ub - lb, step) for each dimension.
static SmallVector<OpFoldResult>
computeForallIterCounts(RewriterBase &rewriter, Location loc,
                        scf::ForallOp forallOp) {
  SmallVector<OpFoldResult> lowerBounds = forallOp.getMixedLowerBound();
  SmallVector<OpFoldResult> upperBounds = forallOp.getMixedUpperBound();
  SmallVector<OpFoldResult> steps = forallOp.getMixedStep();

  int64_t numDims = upperBounds.size();
  AffineExpr s0, s1, s2;
  bindSymbols(rewriter.getContext(), s0, s1, s2);
  AffineExpr numItersExpr = (s0 - s1).ceilDiv(s2);

  SmallVector<OpFoldResult> iterCountOFRs;
  for (int64_t i = 0, e = numDims; i < e; ++i) {
    OpFoldResult iterCount = affine::makeComposedFoldedAffineApply(
        rewriter, loc, numItersExpr,
        {upperBounds[i], lowerBounds[i], steps[i]});
    iterCountOFRs.push_back(iterCount);
  }
  return iterCountOFRs;
}

/// Computes actual forall induction variables from delinearized indices
/// by applying lower bounds and steps: iv = delinearized * step + lb.
static void computeForallIVs(RewriterBase &rewriter, Location loc,
                             scf::ForallOp forallOp, ValueRange delinearizedIvs,
                             IRMapping &forallMapping) {
  SmallVector<OpFoldResult> lowerBounds = forallOp.getMixedLowerBound();
  SmallVector<OpFoldResult> steps = forallOp.getMixedStep();
  int64_t numDims = forallOp.getRank();

  AffineExpr s0, s1, s2;
  bindSymbols(rewriter.getContext(), s0, s1, s2);
  AffineExpr applyLbAndStep = s0 * s1 + s2;

  for (int64_t i = 0, e = numDims; i < e; ++i) {
    Value actualIv = getValueOrCreateConstantIndexOp(
        rewriter, loc,
        affine::makeComposedFoldedAffineApply(
            rewriter, loc, applyLbAndStep,
            {delinearizedIvs[i], steps[i], lowerBounds[i]}));
    forallMapping.map(forallOp.getInductionVar(i), actualIv);
  }
}

/// Composes pcf.write_slice ops with tensor.parallel_insert_slice ops from
/// the forall terminator, creating new write_slice ops that write directly
/// to the pcf.generic's sref arguments.
///
/// This reuses composeNestedSliceParameters() by treating the
/// tensor.parallel_insert_slice as the outer slice and the pcf.write_slice as
/// the inner slice.
static void composeWriteSlicesIntoGeneric(RewriterBase &rewriter,
                                          scf::ForallOp forallOp,
                                          PCF::LoopOp loopOp,
                                          PCF::GenericOp genericOp,
                                          scf::InParallelOp terminator,
                                          IRMapping &forallMapping) {
  // Build mapping from pcf.loop result index -> generic ref arg index.
  SmallVector<unsigned> resultToRefArgIdx(loopOp->getNumResults());
  for (Operation &op : terminator.getYieldingOps()) {
    auto insertOp = cast<tensor::ParallelInsertSliceOp>(&op);
    auto loopResult = cast<OpResult>(insertOp.getSource());
    unsigned resultIdx = loopResult.getResultNumber();
    auto destArg = cast<BlockArgument>(insertOp.getDest());
    unsigned argIdx = destArg.getArgNumber() - forallOp.getRank();
    resultToRefArgIdx[resultIdx] = argIdx;
  }

  // Helper to remap OpFoldResult Values through forallMapping.
  auto remapOFR = [&](OpFoldResult ofr) -> OpFoldResult {
    if (auto val = dyn_cast<Value>(ofr)) {
      return forallMapping.lookupOrDefault(val);
    }
    return ofr;
  };

  for (Operation &op :
       llvm::make_early_inc_range(terminator.getYieldingOps())) {
    auto insertOp = cast<tensor::ParallelInsertSliceOp>(&op);
    auto loopResult = cast<OpResult>(insertOp.getSource());
    unsigned resultIdx = loopResult.getResultNumber();

    BlockArgument genericRefArg =
        genericOp.getRegionRefArgs()[resultToRefArgIdx[resultIdx]];
    BlockArgument movedRefArg = loopOp.getRegionRefArgs()[resultIdx];

    // Compose all write_slice ops that write to this ref arg.
    for (Operation *user : llvm::make_early_inc_range(movedRefArg.getUsers())) {
      auto writeOp = dyn_cast<PCF::WriteSliceOp>(user);
      if (!writeOp) {
        continue;
      }

      rewriter.setInsertionPoint(writeOp);

      SmallVector<OpFoldResult> writeOffsets = writeOp.getMixedOffsets();
      SmallVector<OpFoldResult> writeStrides = writeOp.getMixedStrides();
      SmallVector<OpFoldResult> writeSizes = writeOp.getMixedSizes();

      SmallVector<OpFoldResult> insertOffsets =
          llvm::map_to_vector(insertOp.getMixedOffsets(), remapOFR);
      SmallVector<OpFoldResult> insertSizes =
          llvm::map_to_vector(insertOp.getMixedSizes(), remapOFR);
      SmallVector<OpFoldResult> insertStrides =
          llvm::map_to_vector(insertOp.getMixedStrides(), remapOFR);

      SmallVector<OpFoldResult> composedOffsets;
      SmallVector<OpFoldResult> composedSizes;
      SmallVector<OpFoldResult> composedStrides;
      composeNestedSliceParameters(rewriter, writeOp.getLoc(), insertOffsets,
                                   insertSizes, insertStrides, writeOffsets,
                                   writeSizes, writeStrides, composedOffsets,
                                   composedSizes, composedStrides);

      // Expand source to match sref rank by adding unit dims.
      auto sourceType = cast<RankedTensorType>(writeOp.getSource().getType());
      auto srefType = cast<PCF::ShapedRefType>(genericRefArg.getType());
      int64_t srefRank = srefType.getRank();
      int64_t sourceRank = sourceType.getRank();

      Value expandedSource = writeOp.getSource();
      if (srefRank > sourceRank) {
        SmallVector<int64_t> expandedShape;
        SmallVector<ReassociationIndices> reassociation;

        // First sourceRank-1 dimensions map 1:1.
        for (int64_t i = 0; i < sourceRank - 1; ++i) {
          expandedShape.push_back(sourceType.getDimSize(i));
          reassociation.push_back({i});
        }

        // Last input dimension expands to include itself plus unit dims.
        ReassociationIndices lastGroup;
        if (sourceRank > 0) {
          expandedShape.push_back(sourceType.getDimSize(sourceRank - 1));
          lastGroup.push_back(sourceRank - 1);
        }

        // Add unit dimensions to reach sref rank.
        int64_t numUnitDims = srefRank - sourceRank;
        for (int64_t i = 0; i < numUnitDims; ++i) {
          expandedShape.push_back(1);
          lastGroup.push_back(sourceRank + i);
        }

        if (!lastGroup.empty()) {
          reassociation.push_back(lastGroup);
        }

        auto expandedType =
            RankedTensorType::get(expandedShape, sourceType.getElementType());
        expandedSource = tensor::ExpandShapeOp::create(
            rewriter, writeOp.getLoc(), expandedType, writeOp.getSource(),
            reassociation);
      }

      // Replace old write_slice with composed one.
      rewriter.replaceOpWithNewOp<PCF::WriteSliceOp>(
          writeOp, expandedSource, genericRefArg, composedOffsets,
          composedSizes, composedStrides);
    }

    rewriter.eraseOp(insertOp);
  }
}

/// Core implementation of foldForallIntoPCFLoop after matching succeeds.
static PCF::GenericOp foldForallIntoPCFLoopImpl(RewriterBase &rewriter,
                                                scf::ForallOp forallOp,
                                                PCF::LoopOp loopOp) {
  Location loc = forallOp.getLoc();
  scf::InParallelOp terminator = forallOp.getTerminator();

  // Replace RegionIterArgs with initial values (except in terminator).
  for (auto [iterArg, init] :
       llvm::zip(forallOp.getRegionIterArgs(), forallOp.getOutputs())) {
    rewriter.replaceUsesWithIf(iterArg, init, [&](OpOperand &use) {
      return use.getOwner()->getParentOp() != terminator;
    });
  }

  Value loopCount = loopOp.getCount()[0];

  // Create pcf.generic.
  auto genericOp = PCF::GenericOp::create(
      rewriter, loc,
      /*resultTypes=*/forallOp.getResultTypes(),
      /*scope=*/loopOp.getScope(),
      /*inits=*/forallOp.getOutputs(),
      /*dynamic_sizes=*/ValueRange{},
      /*is_tied=*/SmallVector<bool>(forallOp.getNumResults(), true),
      /*num_iterators=*/1);

  // Set sync scope to SyncOnReturn for the pcf.generic sref arguments.
  Attribute syncScope = PCF::SyncOnReturnAttr::get(rewriter.getContext());
  for (auto regionRefArg : genericOp.getRegionRefArgs()) {
    auto srefType = cast<PCF::ShapedRefType>(regionRefArg.getType());
    auto newSrefType = PCF::ShapedRefType::get(
        rewriter.getContext(), srefType.getShape(), srefType.getElementType(),
        srefType.getScope(), syncScope);
    regionRefArg.setType(newSrefType);
  }

  Block *forallBody = &forallOp.getRegion().front();
  Block *genericBody = &genericOp.getRegion().front();
  rewriter.setInsertionPointToStart(genericBody);

  // Compute per-dimension iteration counts.
  SmallVector<OpFoldResult> iterCountOFRs =
      computeForallIterCounts(rewriter, loc, forallOp);
  int64_t numDims = iterCountOFRs.size();

  SmallVector<Value> iterCountValues =
      llvm::map_to_vector(iterCountOFRs, [&](OpFoldResult ofr) -> Value {
        return getValueOrCreateConstantIndexOp(rewriter, loc, ofr);
      });

  // Delinearize pcf.generic id into (forall dim IVs, pcf loop id) using
  // a single delinearization with the full basis. This allows
  // linearize/delinearize pairs to cancel during canonicalization.
  SmallVector<OpFoldResult> fullBasis(iterCountOFRs);
  fullBasis.push_back(loopCount);

  BlockArgument linearId = genericOp.getIdArgs()[0];
  auto delinOp = affine::AffineDelinearizeIndexOp::create(rewriter, loc,
                                                          linearId, fullBasis);

  // Results 0..numDims-1 are forall dimension indices, last is pcf loop id.
  SmallVector<Value> forallDimIvs;
  for (int64_t i = 0; i < numDims; ++i) {
    forallDimIvs.push_back(delinOp.getResult(i));
  }
  Value pcfLoopLinearId = delinOp.getResult(numDims);

  // Reconstruct forall linear id using affine.linearize_index so that the
  // linearize/delinearize pair can be folded by canonicalization.
  Value forallLinearId =
      affine::AffineLinearizeIndexOp::create(
          rewriter, loc, forallDimIvs, ArrayRef<OpFoldResult>(iterCountOFRs),
          /*disjoint=*/true)
          .getResult();

  // Compute total forall iteration count.
  AffineExpr s0, s1;
  bindSymbols(rewriter.getContext(), s0, s1);
  OpFoldResult totalItersOFR = iterCountOFRs[0];
  for (int64_t i = 1, e = numDims; i < e; ++i) {
    totalItersOFR = affine::makeComposedFoldedAffineApply(
        rewriter, loc, s0 * s1, {totalItersOFR, iterCountOFRs[i]});
  }
  Value totalIters =
      getValueOrCreateConstantIndexOp(rewriter, loc, totalItersOFR);

  Value totalWorkers = genericOp.getCountArgs()[0];
  // Each worker handles ceil(totalWorkers / loopCount) iterations.
  AffineExpr ceilDiv = s0.ceilDiv(s1);
  Value outerStep = getValueOrCreateConstantIndexOp(
      rewriter, loc,
      affine::makeComposedFoldedAffineApply(rewriter, loc, ceilDiv,
                                            {totalWorkers, loopCount}));

  auto outerForall = scf::ForallOp::create(
      rewriter, loc, ArrayRef<OpFoldResult>{forallLinearId},
      ArrayRef<OpFoldResult>{totalIters}, ArrayRef<OpFoldResult>{outerStep},
      /*outputs=*/ValueRange{}, /*mapping=*/std::nullopt);

  // Compute actual forall induction vars inside the outer forall.
  rewriter.setInsertionPointToStart(outerForall.getBody());

  auto forallIvDelinOp = affine::AffineDelinearizeIndexOp::create(
      rewriter, loc, outerForall.getInductionVar(0), iterCountValues);

  IRMapping forallMapping;
  computeForallIVs(rewriter, loc, forallOp, forallIvDelinOp.getMultiIndex(),
                   forallMapping);

  // Move forall body operations into outer forall (except terminator).
  Block *outerForallBody = outerForall.getBody();
  Operation *outerForallTerminator = outerForallBody->getTerminator();

  for (Operation &op :
       llvm::make_early_inc_range(forallBody->without_terminator())) {
    op.moveBefore(outerForallTerminator);
  }

  // Remap induction var block arguments in moved operations.
  for (Value iv : forallOp.getInductionVars()) {
    Value mapped = forallMapping.lookup(iv);
    iv.replaceAllUsesWith(mapped);
  }

  // Compose write_slice ops with parallel_insert_slice ops.
  composeWriteSlicesIntoGeneric(rewriter, forallOp, loopOp, genericOp,
                                terminator, forallMapping);

  // Replace forall results with generic results.
  for (auto [forallResult, genericResult] :
       llvm::zip(forallOp.getResults(), genericOp.getResults())) {
    rewriter.replaceAllUsesWith(forallResult, genericResult);
  }

  rewriter.eraseOp(forallOp);

  // Convert moved pcf.loop to inner scf.forall (no mapping = parallel).
  // The pcf.loop was required to have a single count arg (linearized), so
  // pcfLoopLinearId is sufficient as the sole id.
  rewriter.setInsertionPoint(loopOp);

  auto innerForall = scf::ForallOp::create(
      rewriter, loc, ArrayRef<OpFoldResult>{pcfLoopLinearId},
      ArrayRef<OpFoldResult>{loopCount},
      ArrayRef<OpFoldResult>{getValueOrCreateConstantIndexOp(
          rewriter, loc,
          affine::makeComposedFoldedAffineApply(rewriter, loc, ceilDiv,
                                                {totalWorkers, loopCount}))},
      /*outputs=*/ValueRange{}, /*mapping=*/std::nullopt);

  // Replace loop's id arg with inner forall's induction variable.
  rewriter.replaceAllUsesWith(loopOp.getIdArgs()[0],
                              innerForall.getInductionVar(0));

  // Move operations from loop body to inner forall.
  Block *loopBody = &loopOp.getRegion().front();
  Block *innerForallBody = innerForall.getBody();

  innerForallBody->getOperations().splice(
      std::prev(innerForallBody->end()), loopBody->getOperations(),
      loopBody->begin(), std::prev(loopBody->end()));

  rewriter.eraseOp(loopOp);

  // Add terminator to generic's region.
  rewriter.setInsertionPointToEnd(genericBody);
  PCF::ReturnOp::create(rewriter, loc);

  return genericOp;
}

//===----------------------------------------------------------------------===//
// Test pass: TestFoldForallIntoPCFLoopPass
//===----------------------------------------------------------------------===//

/// Returns true if the forall op has LocalMappingAttr mapping attributes,
/// or the mapping is empty/not present.
static bool hasEmptyOrLocalMapping(scf::ForallOp forallOp) {
  std::optional<ArrayAttr> mapping = forallOp.getMapping();
  if (!mapping || mapping->empty()) {
    return true;
  }
  return llvm::all_of(mapping.value(),
                      llvm::IsaPred<IREE::Codegen::LocalMappingAttr>);
}

struct TestFoldForallIntoPCFLoopPass final
    : impl::TestFoldForallIntoPCFLoopPassBase<TestFoldForallIntoPCFLoopPass> {
  void runOnOperation() override {
    SmallVector<scf::ForallOp> forallOps;
    getOperation()->walk([&](scf::ForallOp forallOp) {
      // Only match foralls with local_mapping attribute.
      if (!hasEmptyOrLocalMapping(forallOp)) {
        return;
      }
      // Check if there's a pcf.loop with sequential scope inside.
      scf::InParallelOp terminator = forallOp.getTerminator();
      Operation *lastOp = terminator->getPrevNode();
      auto loopOp = dyn_cast_if_present<PCF::LoopOp>(lastOp);
      if (!loopOp) {
        return;
      }
      // Check for sequential scope.
      if (!isa<PCF::SequentialAttr>(loopOp.getScope())) {
        return;
      }
      forallOps.push_back(forallOp);
    });

    IRRewriter rewriter(&getContext());
    for (scf::ForallOp forallOp : forallOps) {
      rewriter.setInsertionPoint(forallOp);
      FailureOr<PCF::GenericOp> result =
          foldForallIntoPCFLoop(rewriter, forallOp);
      if (failed(result)) {
        forallOp.emitError("failed to fold forall into pcf.loop");
        return signalPassFailure();
      }
    }
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Public API: foldForallIntoPCFLoop
//===----------------------------------------------------------------------===//

FailureOr<GenericOp> foldForallIntoPCFLoop(RewriterBase &rewriter,
                                           scf::ForallOp forallOp) {
  // Step 1: Validate terminator structure.
  FailureOr<LoopOp> loopOpOrFailure = matchFoldTerminator(forallOp);
  if (failed(loopOpOrFailure)) {
    return rewriter.notifyMatchFailure(
        forallOp, "Failed to validate forall op terminator");
  }
  LoopOp loopOp = *loopOpOrFailure;

  if (failed(matchFoldPCFLoop(forallOp, loopOp))) {
    return rewriter.notifyMatchFailure(forallOp,
                                       "Failed to validate pcf.loop structure");
  }

  // Step 3: Validate write_slice ops.
  if (failed(matchFoldWriteSlices(loopOp))) {
    return rewriter.notifyMatchFailure(forallOp,
                                       "Failed to validate write_slice ops");
  }

  // Step 4: Move count definitions into place before rewriting.
  if (failed(moveValueDefinitions(rewriter, loopOp.getCount(), forallOp))) {
    return rewriter.notifyMatchFailure(
        forallOp, "Failed to move loop trip count definitions");
  }

  // All validations passed, perform the fold.
  return foldForallIntoPCFLoopImpl(rewriter, forallOp, loopOp);
}

} // namespace mlir::iree_compiler::IREE::PCF

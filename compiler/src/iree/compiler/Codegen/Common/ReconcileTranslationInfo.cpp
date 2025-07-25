// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//=== ReconcileTranslationInfo.cpp ---------------------------------------===//
//
// While lowering executable target, the pipelines used are run at a
// func-like op granularity. Each of these func-like operations set the
// workgroup size, and subgroup size as required (as part of the
// `TranslationInfo`). Eventually these have to be reconciled and set
// appropriately on the surrounding HAL ops for the host runtime to pick them
// up. In case of inconsistencies, this pass will throw an error.
//===---------------------------------------------------------------------===//

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "llvm/Support/Casting.h"
#include "mlir/Analysis/CallGraph.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_RECONCILETRANSLATIONINFOPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

class ReconcileTranslationInfoPass final
    : public impl::ReconcileTranslationInfoPassBase<
          ReconcileTranslationInfoPass> {
public:
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

//===---------------------------------------------------------------------===//
// Resolve `scf.forall` operations
//===---------------------------------------------------------------------===//

/// Verify that the mapping attribute provided is of the right form.
static FailureOr<SmallVector<IREE::Codegen::WorkgroupMappingAttr>>
verifyWorkgroupMappingAttrArray(scf::ForallOp forallOp) {
  std::optional<ArrayAttr> mappingAttr = forallOp.getMapping();
  if (!mappingAttr) {
    return forallOp.emitOpError("expected mapped for all op");
  }
  if (mappingAttr.value().empty()) {
    return forallOp.emitOpError("mapping attribute cannot be empty");
  }
  if (failed(IREE::Codegen::WorkgroupMappingAttr::verifyAttrList(
          forallOp.getContext(), forallOp.getLoc(), mappingAttr->getValue()))) {
    return failure();
  }
  SmallVector<IREE::Codegen::WorkgroupMappingAttr> workgroupMappingAttrs =
      llvm::map_to_vector(mappingAttr.value(), [](Attribute attr) {
        return cast<IREE::Codegen::WorkgroupMappingAttr>(attr);
      });
  return workgroupMappingAttrs;
}

/// Get the permutation that represents the mapping of loop dimensions to
/// process dimensions.
template <typename MappingAttrType>
SmallVector<int64_t> getMappingPermutation(ArrayRef<MappingAttrType> mapping) {
  return llvm::map_to_vector(mapping, [&](auto a) {
    return static_cast<int64_t>(mapping.size() - 1) - a.getMappingId();
  });
}
template <typename MappingAttrType>
SmallVector<int64_t>
getInvertedMappingPermutation(ArrayRef<MappingAttrType> mapping) {
  return invertPermutationVector(
      getMappingPermutation<MappingAttrType>(mapping));
}

/// Return the procId and nprocs to use for each of the distributed loops,
/// derived from `hal.interface.workgroup.id/count`s.
static std::pair<SmallVector<OpFoldResult>, SmallVector<OpFoldResult>>
getProcIdsAndNprocs(
    scf::ForallOp forallOp, RewriterBase &builder, Location loc,
    SmallVector<IREE::Codegen::WorkgroupMappingAttr> workgroupMappings,
    SmallVector<OpFoldResult> lowerBounds,
    SmallVector<OpFoldResult> upperBounds, SmallVector<OpFoldResult> steps,
    IREE::Codegen::WorkgroupId deLinearizeFrom) {
  assert(workgroupMappings.size() == lowerBounds.size() &&
         "expected as many workgroup mapping attributes as number of loops");

  auto permutation =
      getInvertedMappingPermutation<IREE::Codegen::WorkgroupMappingAttr>(
          workgroupMappings);
  applyPermutationToVector(workgroupMappings, permutation);
  applyPermutationToVector(lowerBounds, permutation);
  applyPermutationToVector(upperBounds, permutation);
  applyPermutationToVector(steps, permutation);

  SmallVector<OpFoldResult> procId(workgroupMappings.size(),
                                   builder.getIndexAttr(0));
  SmallVector<OpFoldResult> nprocs(workgroupMappings.size(),
                                   builder.getIndexAttr(1));

  AffineExpr s0, s1, s2;
  bindSymbols(builder.getContext(), s0, s1, s2);
  AffineExpr extentExpr = (s1 - s0).ceilDiv(s2);
  SmallVector<OpFoldResult> loopExtents;
  if (workgroupMappings.size() > static_cast<size_t>(deLinearizeFrom)) {
    loopExtents.resize(workgroupMappings.size() -
                       static_cast<size_t>(deLinearizeFrom));
  }
  for (int index = workgroupMappings.size() - 1; index >= 0; --index) {
    auto workgroupMapping = workgroupMappings[index];
    auto lowerBound = lowerBounds[index];
    auto upperBound = upperBounds[index];
    auto step = steps[index];
    if (workgroupMapping.getId() < deLinearizeFrom) {
      procId[index] =
          builder
              .create<IREE::HAL::InterfaceWorkgroupIDOp>(
                  loc, static_cast<unsigned>(workgroupMapping.getId()))
              .getResult();
      nprocs[index] =
          builder
              .create<IREE::HAL::InterfaceWorkgroupCountOp>(
                  loc, static_cast<unsigned>(workgroupMapping.getId()))
              .getResult();
      continue;
    }
    OpFoldResult extent = affine::makeComposedFoldedAffineApply(
        builder, loc, extentExpr, {lowerBound, upperBound, step});
    loopExtents[index] = extent;
  }

  // Delinearize the z-dim based on the loop extents.
  if (!loopExtents.empty()) {
    Value deLinearizedDimId =
        builder
            .create<IREE::HAL::InterfaceWorkgroupIDOp>(
                loc, static_cast<unsigned>(deLinearizeFrom))
            .getResult();
    OpFoldResult deLinearizedNprocs =
        builder
            .create<IREE::HAL::InterfaceWorkgroupCountOp>(
                loc, static_cast<unsigned>(deLinearizeFrom))
            .getResult();

    if (loopExtents.size() != 1) {
      auto deLinearizeOp = builder.create<affine::AffineDelinearizeIndexOp>(
          loc, deLinearizedDimId, loopExtents);
      SmallVector<OpFoldResult> orderedDelinearizedDimIds =
          llvm::map_to_vector(deLinearizeOp.getResults(),
                              [](Value v) -> OpFoldResult { return v; });
      SmallVector<OpFoldResult> orderedDelinearizedNprocs;
      AffineMap minMap = AffineMap::get(0, 2, {s0, s1}, builder.getContext());
      AffineExpr ceilDivExpr = s0.ceilDiv(s1);
      for (int index = loopExtents.size() - 1; index >= 0; --index) {
        auto extent = loopExtents[index];
        procId[index] = deLinearizeOp->getResult(index);
        OpFoldResult currNprocs = affine::makeComposedFoldedAffineMin(
            builder, loc, minMap, {extent, deLinearizedNprocs});
        nprocs[index] = currNprocs;
        deLinearizedNprocs = affine::makeComposedFoldedAffineApply(
            builder, loc, ceilDivExpr, {deLinearizedNprocs, currNprocs});
      }
    } else {
      // If there is only one z-dim mapping, just use the ID directly.
      procId[0] = deLinearizedDimId;
      nprocs[0] = deLinearizedNprocs;
    }
  }

  auto inversePermutation = invertPermutationVector(permutation);
  applyPermutationToVector(procId, inversePermutation);
  applyPermutationToVector(nprocs, inversePermutation);
  return std::make_pair(procId, nprocs);
}

/// Resolve the `forallOp` by mapping the loop induction variables to
/// processor IDs. Expected `procIds` and `nProcs` to match the number
/// of induction variables of the loop.
static LogicalResult resolveForAll(RewriterBase &rewriter,
                                   scf::ForallOp forallOp,
                                   ArrayRef<OpFoldResult> procIds,
                                   ArrayRef<OpFoldResult> nProcs,
                                   bool generateLoopNest) {
  SmallVector<OpFoldResult> mixedUpperBound = forallOp.getMixedUpperBound();
  SmallVector<OpFoldResult> mixedStep = forallOp.getMixedStep();
  Location loc = forallOp.getLoc();
  // Scale the process IDs and counts to account for the forall op steps. These
  // are the forall offsets for each process, and the bounds of these offsets.
  SmallVector<Value> procOffsets, procOffsetBounds;
  for (auto [id, count, step] : llvm::zip_equal(procIds, nProcs, mixedStep)) {
    Value procOffset = getValueOrCreateConstantIndexOp(
        rewriter, loc, IREE::LinalgExt::mulOfrs(rewriter, loc, id, step));
    Value procOffsetBound = getValueOrCreateConstantIndexOp(
        rewriter, loc, IREE::LinalgExt::mulOfrs(rewriter, loc, count, step));
    procOffsets.push_back(procOffset);
    procOffsetBounds.push_back(procOffsetBound);
  }

  // If a loop nest is not necessary, then just inline the body of the forall.
  if (!generateLoopNest) {
    rewriter.eraseOp(forallOp.getBody()->getTerminator());
    rewriter.inlineBlockBefore(forallOp.getBody(), forallOp,
                               /*argValues=*/procOffsets);
    rewriter.eraseOp(forallOp);
    return success();
  }

  // The bounds of process offsets may not match the bounds of the forall op, so
  // form a loop nest iterating from the process offsets to the loop bounds, and
  // stepping by the process offset bounds. Inline the body of the forall op
  // into the loop nest.
  SmallVector<Value> forallUbs =
      getValueOrCreateConstantIndexOp(rewriter, loc, mixedUpperBound);
  scf::LoopNest loopNest = scf::buildLoopNest(rewriter, loc, procOffsets,
                                              forallUbs, procOffsetBounds);
  SmallVector<Value> loopNestIvs = llvm::map_to_vector(
      loopNest.loops, [](scf::ForOp loop) { return loop.getInductionVar(); });
  Block *loopNestBody = loopNest.loops.back().getBody();
  rewriter.eraseOp(forallOp.getBody()->getTerminator());
  rewriter.inlineBlockBefore(forallOp.getBody(), loopNestBody->getTerminator(),
                             /*argValues=*/loopNestIvs);
  rewriter.eraseOp(forallOp);
  return success();
}

/// Resolve scf.forall operation by using the workgroup ID and counts.
static LogicalResult
resolveWorkgroupForAll(RewriterBase &rewriter, scf::ForallOp forallOp,
                       IREE::Codegen::WorkgroupId deLinearizeFrom,
                       bool generateLoopNest) {
  if (forallOp->getNumResults() != 0) {
    return forallOp.emitOpError(
        "cannot resolve for all ops with return values");
  }
  SmallVector<OpFoldResult> mixedLowerBound = forallOp.getMixedLowerBound();
  SmallVector<OpFoldResult> mixedUpperBound = forallOp.getMixedUpperBound();
  SmallVector<OpFoldResult> mixedStep = forallOp.getMixedStep();
  FailureOr<SmallVector<IREE::Codegen::WorkgroupMappingAttr>> workgroupMapping =
      verifyWorkgroupMappingAttrArray(forallOp);
  if (failed(workgroupMapping)) {
    return failure();
  }
  if (workgroupMapping->size() != mixedLowerBound.size()) {
    return forallOp.emitOpError(
        "expected as many workgroup mapping attributes as number of loops");
  }

  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(forallOp);
  Location loc = forallOp.getLoc();

  // Get process IDs and counts by querying hal.interface.workgroup.id/count ops
  // and delinearizing any dimensions of the forall beyond `deLinearizeFrom`.
  SmallVector<OpFoldResult> procIds, nProcs;
  std::tie(procIds, nProcs) = getProcIdsAndNprocs(
      forallOp, rewriter, loc, workgroupMapping.value(), mixedLowerBound,
      mixedUpperBound, mixedStep, deLinearizeFrom);

  return resolveForAll(rewriter, forallOp, procIds, nProcs, generateLoopNest);
}

/// Resolve the workgroup counts for the function based on the extents of the
/// `forAllOps`. When there are multiple foralls, the worker counts of the
/// foralls are linearized, and the maximum count is chosen as the workgroup X
/// count for the dispatch.
static LogicalResult
resolveWorkgroupCounts(RewriterBase &rewriter, mlir::FunctionOpInterface funcOp,
                       ArrayRef<scf::ForallOp> forAllOps,
                       IREE::Codegen::WorkgroupId deLinearizeFrom) {
  OpBuilder::InsertionGuard g(rewriter);
  OpFoldResult maxWorkgroupCount;
  AffineExpr s0, s1, s2;
  bindSymbols(rewriter.getContext(), s0, s1, s2);
  AffineExpr countExpr = (s1 - s0).ceilDiv(s2);
  for (scf::ForallOp forAllOp : forAllOps) {
    rewriter.setInsertionPoint(forAllOp);
    SmallVector<OpFoldResult> workgroupCounts;
    Location loc = forAllOp.getLoc();
    for (auto [lb, ub, step] : llvm::zip_equal(forAllOp.getMixedLowerBound(),
                                               forAllOp.getMixedUpperBound(),
                                               forAllOp.getMixedStep())) {
      workgroupCounts.push_back(affine::makeComposedFoldedAffineApply(
          rewriter, loc, countExpr, {lb, ub, step}));
    }
    // If there is only a single forall op, then there is no need to linearize
    // the workgroup counts, since the x, y, and z counts will match the ranges
    // of the single forall.
    if (forAllOps.size() == 1) {
      SmallVector<IREE::Codegen::WorkgroupMappingAttr> mappingAttr =
          llvm::map_to_vector(forAllOp.getMapping().value(), [](auto a) {
            return cast<IREE::Codegen::WorkgroupMappingAttr>(a);
          });
      auto permutation =
          getInvertedMappingPermutation<IREE::Codegen::WorkgroupMappingAttr>(
              mappingAttr);
      workgroupCounts = applyPermutation(workgroupCounts, permutation);
      return lowerWorkgroupCountFromSliceOp(rewriter, funcOp, workgroupCounts,
                                            static_cast<int>(deLinearizeFrom) +
                                                1);
    }
    // If there are multiple foralls, then the workgroup counts will be
    // linearized, and then the workgroup_count_from_slice op will be lowered
    // with the maximum workgroup count.
    OpFoldResult flatWorkgroupCount = rewriter.getIndexAttr(1);
    for (OpFoldResult count : workgroupCounts) {
      flatWorkgroupCount =
          IREE::LinalgExt::mulOfrs(rewriter, loc, flatWorkgroupCount, count);
    }
    auto asValue = [&](OpFoldResult ofr) {
      return getValueOrCreateConstantIndexOp(rewriter, loc, ofr);
    };
    maxWorkgroupCount =
        !maxWorkgroupCount
            ? flatWorkgroupCount
            : rewriter
                  .create<arith::MaxUIOp>(loc, asValue(maxWorkgroupCount),
                                          asValue(flatWorkgroupCount))
                  .getResult();
  }
  OpFoldResult one = rewriter.getIndexAttr(1);
  // The order of dimensions expected by `lowerWorkgroupCountFromSliceOp`
  // is {z, y, x}.
  SmallVector<OpFoldResult> workgroupCounts = {one, one, maxWorkgroupCount};
  return lowerWorkgroupCountFromSliceOp(rewriter, funcOp, workgroupCounts);
}

static LogicalResult
resolveWorkgroupForAll(RewriterBase &rewriter, FunctionOpInterface funcOp,
                       IREE::Codegen::WorkgroupId deLinearizeFrom) {
  Region &body = funcOp.getFunctionBody();

  if (body.empty()) {
    return success();
  }

  SmallVector<scf::ForallOp> workgroupForAllOps;
  funcOp.walk([&workgroupForAllOps](scf::ForallOp forAllOp) {
    std::optional<ArrayAttr> mapping = forAllOp.getMapping();
    if (!mapping) {
      return;
    }
    if (!llvm::all_of(mapping.value(),
                      llvm::IsaPred<IREE::Codegen::WorkgroupMappingAttr>)) {
      return;
    }
    workgroupForAllOps.push_back(forAllOp);
  });

  if (workgroupForAllOps.empty()) {
    // If there are no workgroup distribution loops, set the default
    // number of workgroups to {1, 1, 1}. Note: that this only kicks
    // in if the export op region has
    // `iree_tensor_ext.dispatch.workgroup_count_from_slice
    return lowerWorkgroupCountFromSliceOp(rewriter, funcOp,
                                          ArrayRef<OpFoldResult>{});
  }

  if (!llvm::hasSingleElement(body)) {
    return funcOp.emitOpError("unhandled function with multiple blocks");
  }

  if (failed(resolveWorkgroupCounts(rewriter, funcOp, workgroupForAllOps,
                                    deLinearizeFrom))) {
    return failure();
  }
  // If there are multiple forall ops, then the workgroup counts will be
  // flattened into the workgroup X dim.
  bool multiForall = workgroupForAllOps.size() > 1;
  if (multiForall) {
    deLinearizeFrom = IREE::Codegen::WorkgroupId::IdX;
  }
  for (auto [idx, forallOp] : llvm::enumerate(workgroupForAllOps)) {
    // Generate a loop nest when there are multiple foralls, because the
    // workgroup counts might not match between foralls.
    if (failed(resolveWorkgroupForAll(rewriter, forallOp, deLinearizeFrom,
                                      /*generateLoopNest=*/multiForall))) {
      return failure();
    }
  }
  return success();
}

/// If the dispatch was formed by splitting long running reductions, the
/// workgroup count needs to be modified to account for the parallel partial
/// reductions to be done. This typically means multiplying the number of
/// workgroups currently used by the number of tiles used for the split
/// reduction.
static LogicalResult
lowerSplitReductionModifierOp(RewriterBase &rewriter,
                              FunctionOpInterface entryPointFn,
                              IREE::Codegen::WorkgroupId delinearizeFrom,
                              OpFoldResult splitReductionFactor) {
  std::optional<IREE::HAL::ExecutableExportOp> exportOp =
      getEntryPoint(entryPointFn);
  if (!exportOp) {
    // not entry point.
    return success();
  }
  Block *body = exportOp->getWorkgroupCountBody();
  if (!body) {
    return success();
  }
  auto splitReduceModifiers = body->getOps<
      IREE::TensorExt::DispatchWorkgroupCountSplitReductionModifierOp>();
  if (splitReduceModifiers.empty()) {
    // Nothing to do.
    return success();
  }
  if (!llvm::hasSingleElement(splitReduceModifiers)) {
    return exportOp->emitOpError(
        "unexpected multiple "
        "iree_tensor_ext.dispatch.workgroup.splitk_modifier");
  }
  IREE::TensorExt::DispatchWorkgroupCountSplitReductionModifierOp
      splitReduceModifier = *splitReduceModifiers.begin();
  Location loc = splitReduceModifier->getLoc();

  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(splitReduceModifier);
  FailureOr<SmallVector<OpFoldResult>> materializedWorkgroupCount =
      materializeWorkgroupCountComputation(rewriter, entryPointFn,
                                           splitReductionFactor,
                                           splitReduceModifier.getWorkload());
  if (failed(materializedWorkgroupCount)) {
    return splitReduceModifier->emitOpError(
        "failed to materialize workgroup count computation in the entry point");
  }

  SmallVector<OpFoldResult> replacement =
      llvm::map_to_vector(splitReduceModifier.getSourceWorkgroupCount(),
                          [](Value v) -> OpFoldResult { return v; });
  replacement[static_cast<uint64_t>(delinearizeFrom)] =
      IREE::LinalgExt::mulOfrs(
          rewriter, splitReduceModifier.getLoc(),
          replacement[static_cast<uint64_t>(delinearizeFrom)],
          materializedWorkgroupCount->front());

  SmallVector<Value> replacementVals =
      getValueOrCreateConstantIndexOp(rewriter, loc, replacement);
  rewriter.replaceOp(splitReduceModifier, replacementVals);
  return success();
}

/// Resolve scf.forall introduced by split reductions. The expectation is that
/// there is a single such loop and its the "outermost" `scf.forall`, and this
/// is distribute along grid axis `k`.
///
/// ```mlir
/// scf.forall (%iv) = %lb to %ub step %s {
///   ... = hal.interface.workgroup.id[k]
/// }
/// ```
///
/// `k` is assumed to be the "highest" dimension of the grid used (so has to be
/// <= 3). To resolve the `scf.forall` the number of workgroups along `k` is
/// increased by a factor of `%nsplit = ceildiv(%ub - %lb, %s)`. In addition
///
/// - All uses of `hal.interface.workgroup.id[k]` within `scf.forall` is
///   replaced by `hal.interface.workgroup.id[k] % %orig_numworkgroups`, where
///   `%orig_numworkgroups` is the number of workgroups along `k` that were used
///   before the resolution of the split reduction `scf.forall`. This is same as
///   `hal.interface.workgroup.count[k] / %nsplit`.
/// - All uses of `%iv` is replaced by `hal.interface.workgroup.id[k] /
///   %nsplit`.
/// ```
static LogicalResult
resolveSplitReduceForAll(RewriterBase &rewriter, FunctionOpInterface funcOp,
                         IREE::Codegen::WorkgroupId delinearizeFrom) {
  Region &body = funcOp.getFunctionBody();
  if (body.empty()) {
    return success();
  }

  SmallVector<scf::ForallOp> splitReductionForAllOps;
  funcOp.walk([&](scf::ForallOp forAllOp) {
    auto mapping = forAllOp.getMapping();
    if (!mapping) {
      return;
    }
    if (failed(IREE::LinalgExt::SplitReductionMappingAttr::verifyAttrList(
            rewriter.getContext(), forAllOp.getLoc(), mapping->getValue(),
            /*emitDiagnosticsErrs =*/false))) {
      return;
    }
    splitReductionForAllOps.push_back(forAllOp);
  });

  if (splitReductionForAllOps.empty()) {
    return lowerSplitReductionModifierOp(rewriter, funcOp, delinearizeFrom,
                                         rewriter.getIndexAttr(1));
  }

  // For now support only a single split-reduction forall.
  if (splitReductionForAllOps.size() != 1) {
    return funcOp->emitOpError(
        "failed to resolve multiple split-reduction loops");
  }

  scf::ForallOp forallOp = splitReductionForAllOps.front();
  if (forallOp->getNumResults() != 0) {
    return forallOp.emitOpError(
        "cannot resolve for all ops with return values");
  }

  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(forallOp);
  Location loc = forallOp.getLoc();

  // Compute the number of iterations of the split-reduction loop.
  // Since the split-reduction loop will be mapped with one iteration
  // per workgroup, the product of number of iterations for all the ranks
  // is also the number of processors used for split-reduction mapping.
  SmallVector<OpFoldResult> lbs = forallOp.getMixedLowerBound();
  SmallVector<OpFoldResult> ubs = forallOp.getMixedUpperBound();
  SmallVector<OpFoldResult> steps = forallOp.getMixedStep();
  AffineExpr s0, s1, s2;
  bindSymbols(rewriter.getContext(), s0, s1, s2);
  AffineExpr numItersExpr = (s1 - s0).ceilDiv(s2);
  auto numIters =
      llvm::map_to_vector(llvm::zip_equal(lbs, ubs, steps), [&](auto it) {
        OpFoldResult lb = std::get<0>(it), ub = std::get<1>(it),
                     step = std::get<2>(it);
        return affine::makeComposedFoldedAffineApply(
            rewriter, loc, numItersExpr, {lb, ub, step});
      });
  AffineExpr linearizeExpr = s0;
  for (unsigned i = 1, e = lbs.size(); i < e; ++i) {
    AffineExpr s = rewriter.getAffineSymbolExpr(i);
    linearizeExpr = linearizeExpr * s;
  }
  OpFoldResult nSplitProcs = affine::makeComposedFoldedAffineApply(
      rewriter, loc, linearizeExpr, numIters);

  // Lower the splitk-modifier op in the entry point.
  if (failed(lowerSplitReductionModifierOp(rewriter, funcOp, delinearizeFrom,
                                           nSplitProcs))) {
    return forallOp->emitOpError("failed to lower split reduction modifier op");
  }

  auto procIdOp = rewriter.create<IREE::HAL::InterfaceWorkgroupIDOp>(
      loc, static_cast<unsigned>(delinearizeFrom));
  auto nTotalProcsOp = rewriter.create<IREE::HAL::InterfaceWorkgroupCountOp>(
      loc, static_cast<unsigned>(delinearizeFrom));
  OpFoldResult nTotalProcs = nTotalProcsOp.getResult();
  OpFoldResult origNProcs = affine::makeComposedFoldedAffineApply(
      rewriter, loc, s0.floorDiv(s1), {nTotalProcs, nSplitProcs});
  SmallVector<OpFoldResult> basis = numIters;
  basis.push_back(origNProcs);
  auto delinearizeOp = rewriter.create<affine::AffineDelinearizeIndexOp>(
      loc, procIdOp.getResult(), basis);

  Value workgroupIdReplacement = delinearizeOp.getResults().back();

  // Check that all uses of `hal.interface.workgroup.id[delinearizeFrom]` are
  // within the `scf.forall` operation.
  auto walkResult = funcOp.walk(
      [&](IREE::HAL::InterfaceWorkgroupIDOp workgroupIdOp) -> WalkResult {
        if (workgroupIdOp.getDimension().getZExtValue() !=
            llvm::to_underlying(delinearizeFrom)) {
          return WalkResult::advance();
        }
        if (workgroupIdOp == procIdOp) {
          return WalkResult::advance();
        }
        for (Operation *user : workgroupIdOp->getUsers()) {
          auto parentForallOp = user->getParentOfType<scf::ForallOp>();
          if (parentForallOp != forallOp) {
            return user->emitOpError("expected all users of `workgroup.id` to "
                                     "be within split reduction scf.forall op");
          }
        }
        rewriter.replaceAllUsesWith(workgroupIdOp, workgroupIdReplacement);
        return WalkResult::advance();
      });
  if (walkResult.wasInterrupted()) {
    return failure();
  }

  auto splitReduceOpProcIds =
      llvm::map_to_vector(delinearizeOp.getResults().drop_back(),
                          [](Value v) -> OpFoldResult { return v; });
  auto splitReduceMapping = llvm::map_to_vector(
      forallOp.getMapping()->getValue(), [](Attribute attr) {
        return cast<IREE::LinalgExt::SplitReductionMappingAttr>(attr);
      });

  SmallVector<int64_t> mappingPermutation =
      getMappingPermutation<IREE::LinalgExt::SplitReductionMappingAttr>(
          splitReduceMapping);
  applyPermutationToVector(splitReduceOpProcIds, mappingPermutation);
  applyPermutationToVector(numIters, mappingPermutation);
  return resolveForAll(rewriter, forallOp, splitReduceOpProcIds, numIters,
                       /*generateLoopNest = */ false);
}

//===---------------------------------------------------------------------===//
// End Resolve `scf.forall` operations
//===---------------------------------------------------------------------===//

// Reconcile workgroup sizes across all translation infos.
static FailureOr<SmallVector<int64_t>> reconcileWorkgroupSize(
    ArrayRef<IREE::Codegen::TranslationInfoAttr> translationInfos) {
  if (translationInfos.empty()) {
    return SmallVector<int64_t>{};
  }
  SmallVector<int64_t> reconciledWorkgroupSize =
      llvm::to_vector(translationInfos.front().getWorkgroupSize());
  for (auto translationInfo : translationInfos.drop_front()) {
    auto workGroupSize = llvm::to_vector(translationInfo.getWorkgroupSize());
    if (workGroupSize != reconciledWorkgroupSize) {
      return failure();
    }
  }
  return reconciledWorkgroupSize;
}

// Reconcile subgroup size across all translation infos.
static FailureOr<int64_t> reconcileSubgroupSize(
    ArrayRef<IREE::Codegen::TranslationInfoAttr> translationInfos) {
  if (translationInfos.empty()) {
    return 0;
  }
  int64_t subgroupSize = translationInfos.front().getSubgroupSize();
  for (auto translationInfo : translationInfos.drop_front()) {
    if (subgroupSize != translationInfo.getSubgroupSize()) {
      return failure();
    }
  }
  return subgroupSize;
}

/// Helper function to retrieve the target-func-attrs value from translation
/// info.
static DictionaryAttr
getTargetFuncAttrs(IREE::Codegen::TranslationInfoAttr translationInfo) {
  auto translationConfig = translationInfo.getConfiguration();
  if (!translationConfig) {
    return nullptr;
  }
  auto attr = translationConfig.getAs<DictionaryAttr>("llvm_func_attrs");
  if (!attr) {
    return nullptr;
  }
  return attr;
}

void ReconcileTranslationInfoPass::runOnOperation() {
  auto variantOp = getOperation();
  auto innerModuleOp = variantOp.getInnerModule();
  MLIRContext *context = &getContext();

  if (foldSplitReductionLoopIntoWorkgroupMappingLoop) {
    RewritePatternSet foldLoopPattern(context);
    populateFoldSplitReductionAndWorkgroupMappingLoops(foldLoopPattern);
    if (failed(
            applyPatternsGreedily(innerModuleOp, std::move(foldLoopPattern)))) {
      innerModuleOp.emitOpError(
          "failed to fold split-reduction loop and workgroup mapping loop");
      return signalPassFailure();
    }
  }

  // Get the symbol table of the inner module to lookup exported functions.
  SymbolTable symbolTable(innerModuleOp);

  // Construct the call-graph for the inner module. We traverse this when
  // reconciling translation info.
  CallGraph callGraph(innerModuleOp);

  IRRewriter rewriter(&getContext());
  auto exportOps = variantOp.getOps<IREE::HAL::ExecutableExportOp>();

  for (auto exportOp : exportOps) {
    SmallVector<IREE::Codegen::TranslationInfoAttr> translationInfos;
    auto rootFuncOp = llvm::dyn_cast_if_present<FunctionOpInterface>(
        symbolTable.lookup(exportOp.getSymNameAttr()));
    if (!rootFuncOp || rootFuncOp.isExternal()) {
      // Skip external functions.
      continue;
    }

    // Resolve workgroup distribution related `scf.forall` ops.
    if (failed(resolveWorkgroupForAll(rewriter, rootFuncOp, distributeAlong))) {
      variantOp.emitOpError(
          "failed to resolve workgroup distribution forall ops");
      return signalPassFailure();
    }

    if (failed(
            resolveSplitReduceForAll(rewriter, rootFuncOp, distributeAlong))) {
      variantOp.emitOpError("failed to resolve split reduction forall ops");
    }

    std::queue<FunctionOpInterface> nodeQueue;
    nodeQueue.push(rootFuncOp);

    llvm::SmallDenseSet<FunctionOpInterface> visitedFunctions;

    // Walk the callgraph from the export root to find all translation info
    // attributes and determine whether they are consistent.
    while (!nodeQueue.empty()) {
      FunctionOpInterface funcOp = nodeQueue.front();
      if (!visitedFunctions.insert(funcOp).second) {
        rootFuncOp.emitOpError(
            "recursive function call in translation info reconciliation");
        return signalPassFailure();
      }

      if (CallGraphNode *node =
              callGraph.lookupNode(&funcOp.getFunctionBody())) {
        for (CallGraphNode::Edge callEdge : *node) {
          if (callEdge.getTarget()->isExternal()) {
            // Skip external calls.
            continue;
          }
          auto calledFunc = callEdge.getTarget()
                                ->getCallableRegion()
                                ->getParentOfType<FunctionOpInterface>();
          nodeQueue.push(calledFunc);
        }
      }
      nodeQueue.pop();

      auto translationInfo = getTranslationInfo(funcOp);
      if (!translationInfo) {
        // No translation info means nothing to reconcile.
        continue;
      }
      translationInfos.push_back(translationInfo);

      // The following is moving the target-func-attrs specification from
      // translation info into the func-like op. This is not the best
      // place to do this, but the intent is after this pass all the
      // lowering configs and translation infos will be deleted.
      DictionaryAttr targetFuncAttrs = getTargetFuncAttrs(translationInfo);
      if (targetFuncAttrs) {
        funcOp->setAttr("llvm_func_attrs", targetFuncAttrs);
      }
    }

    // Reconcile workgroup sizes.
    FailureOr<SmallVector<int64_t>> reconciledWorkgroupSize =
        reconcileWorkgroupSize(translationInfos);
    if (failed(reconciledWorkgroupSize)) {
      exportOp.emitOpError("failed to reconcile workgroup sizes");
      return signalPassFailure();
    }
    if (reconciledWorkgroupSize->size() > 3) {
      exportOp.emitOpError(
          "reconciled workgroup size is greater than 3 (illegal)");
      return signalPassFailure();
    }
    std::array<int64_t, 3> workgroupSize = {1, 1, 1};
    for (auto [index, size] :
         llvm::enumerate(reconciledWorkgroupSize.value())) {
      workgroupSize[index] = size;
    }
    auto workgroupSizeArrayAttr = rewriter.getIndexArrayAttr(workgroupSize);
    exportOp.setWorkgroupSizeAttr(workgroupSizeArrayAttr);

    // Reconcile subgroup sizes.
    FailureOr<int64_t> reconciledSubgroupSize =
        reconcileSubgroupSize(translationInfos);
    if (failed(reconciledSubgroupSize)) {
      exportOp.emitOpError("failed to reconcile subgroup size");
      return signalPassFailure();
    }
    if (reconciledSubgroupSize.value() != 0) {
      exportOp.setSubgroupSizeAttr(
          rewriter.getIndexAttr(reconciledSubgroupSize.value()));
    }
  }

  // Erase all the lowering configs and translation infos after we have finished
  // processing all exported functions.
  SmallVector<IREE::TensorExt::DispatchWorkloadOrdinalOp> ordinalOps;
  innerModuleOp->walk([&ordinalOps](Operation *op) {
    if (auto funcOp = dyn_cast<FunctionOpInterface>(op)) {
      eraseTranslationInfo(funcOp);
    }
    eraseLoweringConfig(op);
    if (auto ordinalOp =
            dyn_cast<IREE::TensorExt::DispatchWorkloadOrdinalOp>(op)) {
      ordinalOps.push_back(ordinalOp);
    }
  });

  // Discard all ordinal ops.
  for (auto ordinalOp : ordinalOps) {
    rewriter.replaceOp(ordinalOp, ordinalOp.getOperand());
  }
}

} // namespace mlir::iree_compiler

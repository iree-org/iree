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

#include <algorithm>
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
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
/// process dimensions. The mapping list is expected to contain a set of
/// mappings with consecutive mapping IDs (i.e., all mapping IDs between the
/// highest and lowest mapping ID in the set must be present in the list). So,
/// the mapping `[x, z]` would not be legal, but the mappings `[x, z, y]` or
/// `[y, z]` would both be legal. The resulting permutation will permute the
/// list of mapping IDs into descending order by mapping ID. For example:
///
/// [#iree_codegen.workgroup_mapping<z>,
///  #iree_codegen.workgroup_mapping<x>,
///  #iree_codegen.workgroup_mapping<y>]
///
/// Would result in a permutation of [0, 2, 1], and:
///
/// [#iree_codegen.workgroup_mapping<x>,
///  #iree_codegen.workgroup_mapping<y>]
///
/// Would result in a permutation of [1, 0].
template <typename MappingAttrType>
SmallVector<int64_t> getMappingPermutation(ArrayRef<MappingAttrType> mapping) {
  int64_t mappingBase =
      std::min_element(mapping.begin(), mapping.end(), [](auto a, auto b) {
        return a.getMappingId() < b.getMappingId();
      })->getMappingId();
  return llvm::map_to_vector(mapping, [&](auto a) {
    int64_t normalizedMappingId = a.getMappingId() - mappingBase;
    return static_cast<int64_t>(mapping.size() - 1) - normalizedMappingId;
  });
}

/// Returns the inverse of `getMappingPermutation`.
template <typename MappingAttrType>
SmallVector<int64_t>
getInvertedMappingPermutation(ArrayRef<MappingAttrType> mapping) {
  return invertPermutationVector(
      getMappingPermutation<MappingAttrType>(mapping));
}

/// Resolve the `forallOp` by mapping the loop induction variables to
/// processor IDs. Expected `procIds` and `nProcs` to match the number
/// of induction variables of the loop.
static LogicalResult resolveForAll(RewriterBase &rewriter,
                                   scf::ForallOp forallOp,
                                   ArrayRef<OpFoldResult> procIds,
                                   ArrayRef<OpFoldResult> nProcs,
                                   ArrayRef<bool> generateLoopNest) {
  assert(generateLoopNest.size() == forallOp.getRank());

  SmallVector<Value> loopNestIvs;
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(forallOp);

  SmallVector<OpFoldResult> mixedLbs = forallOp.getMixedLowerBound();
  SmallVector<OpFoldResult> mixedUbs = forallOp.getMixedUpperBound();
  SmallVector<OpFoldResult> mixedSteps = forallOp.getMixedStep();
  assert(mixedLbs.size() == procIds.size());
  assert(mixedLbs.size() == nProcs.size());
  Location loc = forallOp.getLoc();
  Operation *bodyInsertionPoint = forallOp;
  for (auto [index, lb] : llvm::enumerate(mixedLbs)) {
    OpFoldResult ub = mixedUbs[index], step = mixedSteps[index],
                 numProc = nProcs[index], procId = procIds[index];
    Value forLb = getValueOrCreateConstantIndexOp(
        rewriter, loc,
        IREE::LinalgExt::mulAddOfrs(rewriter, loc, procId, step, lb));
    if (!generateLoopNest[index]) {
      loopNestIvs.push_back(forLb);
      continue;
    }

    Value forUb = getValueOrCreateConstantIndexOp(rewriter, loc, ub);
    Value forStep = getValueOrCreateConstantIndexOp(
        rewriter, loc, IREE::LinalgExt::mulOfrs(rewriter, loc, numProc, step));
    auto loop = scf::ForOp::create(rewriter, loc, forLb, forUb, forStep);
    loopNestIvs.push_back(loop.getInductionVar());
    bodyInsertionPoint = loop.getBody()->getTerminator();
    rewriter.setInsertionPointToStart(loop.getBody());
  }
  Block *forAllBody = forallOp.getBody();
  rewriter.eraseOp(forAllBody->getTerminator());
  rewriter.inlineBlockBefore(forAllBody, bodyInsertionPoint, loopNestIvs);
  rewriter.eraseOp(forallOp);
  return success();
}

/// Collapse all the dimensions of the `scf.forall` that have mapping ID
/// "greater" than `delinearizeFrom` into a single dimension. This dimension
/// is placed at the same place as the position of `delinearizeFrom` in the
/// original loop.
scf::ForallOp
collapseForAllOpDimensions(RewriterBase &rewriter, scf::ForallOp forallOp,
                           IREE::Codegen::WorkgroupId delinearizeFrom) {
  SmallVector<OpFoldResult> mixedLbs = forallOp.getMixedLowerBound();
  SmallVector<OpFoldResult> mixedUbs = forallOp.getMixedUpperBound();
  SmallVector<OpFoldResult> mixedSteps = forallOp.getMixedStep();
  auto mapping = llvm::map_to_vector(
      forallOp.getMapping()->getValue(), [](Attribute attr) {
        return cast<IREE::Codegen::WorkgroupMappingAttr>(attr);
      });

  // Collect all dimensions that are to be collapsed.
  auto delinearizeFromAttr = IREE::Codegen::WorkgroupMappingAttr::get(
      rewriter.getContext(), delinearizeFrom);
  SmallVector<int64_t> collapsedDims;
  SmallVector<IREE::Codegen::WorkgroupMappingAttr> collapsedAttrs;
  for (auto [index, mappingAttr] : llvm::enumerate(mapping)) {
    if (mappingAttr < delinearizeFromAttr) {
      continue;
    }
    collapsedDims.push_back(index);
    collapsedAttrs.push_back(mappingAttr);
  }

  AffineExpr s0, s1, s2;
  bindSymbols(rewriter.getContext(), s0, s1, s2);
  AffineExpr nIterExpr = (s1 - s0).ceilDiv(s2);
  Location loc = forallOp.getLoc();
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(forallOp);

  SmallVector<OpFoldResult> nIters =
      llvm::map_to_vector(collapsedDims, [&](int64_t index) {
        return affine::makeComposedFoldedAffineApply(
            rewriter, loc, nIterExpr,
            {mixedLbs[index], mixedUbs[index], mixedSteps[index]});
      });

  OpFoldResult one = rewriter.getIndexAttr(1);
  OpFoldResult combinedNIters = one;
  // The reverse below is just to preserve some lit-test behavior.
  for (auto nIter : llvm::reverse(nIters)) {
    combinedNIters =
        IREE::LinalgExt::mulOfrs(rewriter, loc, combinedNIters, nIter);
  }

  // Create the collapsed forall op.
  SmallVector<OpFoldResult> newLbs, newUbs, newSteps;
  SmallVector<Attribute> newMapping;
  OpFoldResult zero = rewriter.getIndexAttr(0);
  std::optional<int> delinearizedFromIndex;
  for (auto [index, mappingAttr] : llvm::enumerate(mapping)) {
    if (mappingAttr < delinearizeFromAttr) {
      newLbs.push_back(mixedLbs[index]);
      newUbs.push_back(mixedUbs[index]);
      newSteps.push_back(mixedSteps[index]);
      newMapping.push_back(mappingAttr);
      continue;
    }
    if (mappingAttr == delinearizeFromAttr) {
      newLbs.push_back(zero);
      newUbs.push_back(combinedNIters);
      newSteps.push_back(one);
      newMapping.push_back(mappingAttr);
      delinearizedFromIndex = newMapping.size() - 1;
    }
  }

  scf::ForallOp newForOp = scf::ForallOp::create(
      rewriter, loc, newLbs, newUbs, newSteps,
      /*outputs = */ ValueRange{}, rewriter.getArrayAttr(newMapping));

  Block *newForallBody = newForOp.getBody();
  {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(newForallBody->getTerminator());
    SmallVector<int64_t> invertPermutation =
        getInvertedMappingPermutation<IREE::Codegen::WorkgroupMappingAttr>(
            collapsedAttrs);
    applyPermutationToVector(nIters, invertPermutation);
    SmallVector<OpFoldResult> basis = llvm::to_vector(nIters);

    std::optional<SmallVector<Value>> ivs = newForOp.getLoopInductionVars();
    auto delinearizeOp = affine::AffineDelinearizeIndexOp::create(
        rewriter, loc, ivs.value()[delinearizedFromIndex.value()], basis);
    auto replacementIvs = llvm::to_vector(delinearizeOp.getResults());
    applyPermutationToVector(replacementIvs,
                             invertPermutationVector(invertPermutation));
    int replacementIvIndex = 0;
    SmallVector<Value> remappedIvs;
    unsigned nonLinearizedIVIdx = 0;
    for (auto [index, mappingAttr] : llvm::enumerate(mapping)) {
      if (mappingAttr < delinearizeFromAttr) {
        remappedIvs.push_back((*ivs)[nonLinearizedIVIdx++]);
        continue;
      }
      if (mappingAttr == delinearizeFromAttr) {
        nonLinearizedIVIdx++;
      }
      OpFoldResult remappedIv = IREE::LinalgExt::mulAddOfrs(
          rewriter, loc, replacementIvs[replacementIvIndex++],
          mixedSteps[index], mixedLbs[index]);
      remappedIvs.push_back(
          getValueOrCreateConstantIndexOp(rewriter, loc, remappedIv));
    }
    Block *forallBody = forallOp.getBody();
    rewriter.eraseOp(forallBody->getTerminator());
    rewriter.inlineBlockBefore(forallBody, newForallBody->getTerminator(),
                               remappedIvs);
  }
  rewriter.eraseOp(forallOp);
  return newForOp;
}

/// Resolve scf.forall operation by using the workgroup ID and counts.
static FailureOr<SmallVector<OpFoldResult>>
resolveWorkgroupForAll(RewriterBase &rewriter, scf::ForallOp forallOp,
                       IREE::Codegen::WorkgroupId delinearizeFrom,
                       bool multiForall) {
  if (forallOp->getNumResults() != 0) {
    return forallOp.emitOpError(
        "cannot resolve for all ops with return values");
  }
  if (forallOp.getRank() > llvm::to_underlying(delinearizeFrom) + 1) {
    forallOp = collapseForAllOpDimensions(rewriter, forallOp, delinearizeFrom);
  }
  assert(forallOp.getRank() <= llvm::to_underlying(delinearizeFrom) + 1);
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

  std::array<int64_t, 3> maxWorkgroupCountArray =
      getMaxWorkgroupCount(forallOp);
  SmallVector<int64_t> maxWorkgroupCount =
      llvm::to_vector(maxWorkgroupCountArray);
  maxWorkgroupCount.resize(forallOp.getRank(), ShapedType::kDynamic);

  AffineExpr s0, s1, s2;
  bindSymbols(rewriter.getContext(), s0, s1, s2);
  AffineExpr nItersExpr = (s1 - s0).ceilDiv(s2);

  SmallVector<OpFoldResult> numWorkgroupsList, mappedProcIds, mappedNProcs;
  SmallVector<bool> generateLoopNest(forallOp.getRank(), true);
  numWorkgroupsList.resize(forallOp.getRank());
  for (auto [index, mapping] : llvm::enumerate(workgroupMapping.value())) {
    int64_t mappingID = mapping.getMappingId();
    OpFoldResult procId = rewriter
                              .create<IREE::HAL::InterfaceWorkgroupIDOp>(
                                  loc, static_cast<unsigned>(mappingID))
                              .getResult();
    OpFoldResult nprocs = rewriter
                              .create<IREE::HAL::InterfaceWorkgroupCountOp>(
                                  loc, static_cast<unsigned>(mappingID))
                              .getResult();

    mappedProcIds.push_back(procId);
    mappedNProcs.push_back(nprocs);

    OpFoldResult &numWorkgroups =
        numWorkgroupsList[numWorkgroupsList.size() - 1 - mappingID];
    numWorkgroups = affine::makeComposedFoldedAffineApply(
        rewriter, loc, nItersExpr,
        {mixedLowerBound[index], mixedUpperBound[index], mixedStep[index]});

    int64_t maxCount = maxWorkgroupCount[mappingID];
    if (!multiForall) {
      if (ShapedType::isDynamic(maxCount)) {
        // Dynamic value indicates there is no limit.
        generateLoopNest[index] = false;
        continue;
      }
      if (std::optional<int64_t> numWorkgroupsStatic =
              getConstantIntValue(numWorkgroups)) {
        if (numWorkgroupsStatic.value() <= maxCount) {
          generateLoopNest[index] = false;
        }
        continue;
      }
    }

    if (!ShapedType::isDynamic(maxCount)) {
      OpFoldResult boundedNumWorkgroups = affine::makeComposedFoldedAffineMin(
          rewriter, loc,
          AffineMap::get(
              0, 1,
              {s0, getAffineConstantExpr(maxCount, rewriter.getContext())},
              rewriter.getContext()),
          numWorkgroups);
      numWorkgroups = boundedNumWorkgroups;
    }
  }
  if (failed(resolveForAll(rewriter, forallOp, mappedProcIds, mappedNProcs,
                           generateLoopNest))) {
    return forallOp.emitOpError("failed to resolve scf.forall op");
  }
  return numWorkgroupsList;
}

/// Resolve all `scf.forall`s within a given `funcOp`. If there are multiple
/// `scf.forall`s they all are resolved to use the same number of workgroups.
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

  // If there are multiple forall ops, then the workgroup counts will be
  // flattened into the workgroup X dim.
  bool multiForall = workgroupForAllOps.size() > 1;
  if (multiForall) {
    deLinearizeFrom = IREE::Codegen::WorkgroupId::IdX;
  }
  SmallVector<SmallVector<OpFoldResult>> numWorkgroupsLists;
  rewriter.setInsertionPointAfter(workgroupForAllOps.back());
  for (auto [idx, forallOp] : llvm::enumerate(workgroupForAllOps)) {
    // Generate a loop nest when there are multiple foralls, because the
    // workgroup counts might not match between foralls.
    FailureOr<SmallVector<OpFoldResult>> numWorkgroups =
        resolveWorkgroupForAll(rewriter, forallOp, deLinearizeFrom,
                               /*multiForall =*/multiForall);

    if (failed(numWorkgroups)) {
      return failure();
    }
    numWorkgroupsLists.push_back(numWorkgroups.value());
  }
  // For the first loop resolve the number of workgroups
  SmallVector<OpFoldResult> maxNumWorkgroups =
      numWorkgroupsLists.pop_back_val();
  Location loc = funcOp.getLoc();
  auto asValue = [&](OpFoldResult ofr) {
    return getValueOrCreateConstantIndexOp(rewriter, loc, ofr);
  };
  for (SmallVector<OpFoldResult> numWorkgroupsList : numWorkgroupsLists) {
    for (auto [idx, numWorkgroups] : llvm::enumerate(numWorkgroupsList)) {
      maxNumWorkgroups[idx] =
          rewriter
              .create<arith::MaxUIOp>(loc, asValue(numWorkgroups),
                                      asValue(maxNumWorkgroups[idx]))
              .getResult();
    }
  }
  if (failed(lowerWorkgroupCountFromSliceOp(
          rewriter, funcOp, maxNumWorkgroups,
          llvm::to_underlying(deLinearizeFrom) + 1))) {
    return failure();
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
  SmallVector<bool> generateLoopNests(forallOp.getRank(), false);
  return resolveForAll(rewriter, forallOp, splitReduceOpProcIds, numIters,
                       generateLoopNests);
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

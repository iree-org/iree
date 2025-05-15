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
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"

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
SmallVector<int64_t>
getMappingPermutation(ArrayRef<IREE::Codegen::WorkgroupMappingAttr> mapping) {
  return invertPermutationVector(llvm::map_to_vector(mapping, [&](auto a) {
    return static_cast<int64_t>(mapping.size() - 1) - a.getMappingId();
  }));
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

  auto permutation = getMappingPermutation(workgroupMappings);
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

/// Resolve scf.forall operation by using the workgroup ID and counts.
static LogicalResult
resolveWorkgroupForAll(RewriterBase &rewriter, scf::ForallOp forallOp,
                       IREE::Codegen::WorkgroupId deLinearizeFrom) {
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

  SmallVector<OpFoldResult> procIds;
  SmallVector<OpFoldResult> nProcs;
  std::tie(procIds, nProcs) = getProcIdsAndNprocs(
      forallOp, rewriter, loc, workgroupMapping.value(), mixedLowerBound,
      mixedUpperBound, mixedStep, deLinearizeFrom);

  Block *parentBlock = forallOp->getBlock();
  Block *remainingBlock =
      rewriter.splitBlock(parentBlock, Block::iterator(forallOp));
  for (auto [id, step] : llvm::zip_equal(procIds, mixedStep)) {
    rewriter.setInsertionPointToEnd(parentBlock);
    AffineExpr s0, s1;
    bindSymbols(rewriter.getContext(), s0, s1);
    AffineExpr expr = s1 * s0;
    id = affine::makeComposedFoldedAffineApply(rewriter, loc, expr, {id, step});
  }
  Block *loopBody = forallOp.getBody();
  rewriter.eraseOp(loopBody->getTerminator());

  // The forall's number of iterations might not match the number of processes,
  // so create a loop nest starting from the process IDs, and ending at the
  // forall loop bounds.
  SmallVector<Value> lbs =
      getValueOrCreateConstantIndexOp(rewriter, loc, procIds);
  SmallVector<Value> ubs = getValueOrCreateConstantIndexOp(
      rewriter, loc, forallOp.getMixedUpperBound());
  SmallVector<Value> steps =
      getValueOrCreateConstantIndexOp(rewriter, loc, nProcs);
  scf::LoopNest loopNest = scf::buildLoopNest(rewriter, loc, lbs, ubs, steps);
  SmallVector<Value> argReplacements = llvm::map_to_vector(
      loopNest.loops, [](scf::ForOp loop) { return loop.getInductionVar(); });
  Block *loopNestBlock = loopNest.loops.back().getBody();
  Operation *yieldOp = loopNestBlock->getTerminator();
  rewriter.inlineBlockBefore(loopBody, yieldOp, argReplacements);
  rewriter.mergeBlocks(remainingBlock, parentBlock, ValueRange{});
  rewriter.eraseOp(forallOp);
  return success();
}

static LogicalResult
resolveWorkgroupCounts(RewriterBase &rewriter, mlir::FunctionOpInterface funcOp,
                       ArrayRef<scf::ForallOp> forAllOps,
                       IREE::Codegen::WorkgroupId deLinearizeFrom) {
  OpBuilder::InsertionGuard g(rewriter);
  bool multiForall = forAllOps.size() > 1;
  OpFoldResult maxWorkgroupCount;
  for (scf::ForallOp forAllOp : forAllOps) {
    rewriter.setInsertionPoint(forAllOp);
    SmallVector<OpFoldResult> lowerBounds = forAllOp.getMixedLowerBound();
    SmallVector<OpFoldResult> upperBounds = forAllOp.getMixedUpperBound();
    SmallVector<OpFoldResult> steps = forAllOp.getMixedStep();
    SmallVector<OpFoldResult> workgroupCount(lowerBounds.size());
    AffineExpr s0, s1, s2;
    bindSymbols(rewriter.getContext(), s0, s1, s2);
    AffineExpr countExpr = (s1 - s0).ceilDiv(s2);
    Location loc = forAllOp.getLoc();
    for (auto [index, lb, ub, step] :
         llvm::enumerate(lowerBounds, upperBounds, steps)) {
      workgroupCount[index] = affine::makeComposedFoldedAffineApply(
          rewriter, loc, countExpr, {lb, ub, step});
    }
    // If there is only a single forall op, then there is no need to linearize
    // the workgroup counts, since the x, y, and z counts will match the ranges
    // of the single forall.
    if (!multiForall) {
      auto mappingAttr =
          llvm::map_to_vector(forAllOp.getMapping().value(), [](auto a) {
            return cast<IREE::Codegen::WorkgroupMappingAttr>(a);
          });
      auto permutation = getMappingPermutation(mappingAttr);
      workgroupCount = applyPermutation(workgroupCount, permutation);
      return lowerWorkgroupCountFromSliceOp(rewriter, funcOp, workgroupCount,
                                            static_cast<int>(deLinearizeFrom) +
                                                1);
    }
    // If there are multiple foralls, then the workgroup counts will be
    // linearized, and then the workgroup_count_from_slice op will be lowered
    // with the maximum workgroup count.
    AffineExpr prodExpr = rewriter.getAffineConstantExpr(1);
    for (int i = 0; i < workgroupCount.size(); ++i) {
      prodExpr = prodExpr * rewriter.getAffineSymbolExpr(i);
    }
    OpFoldResult flatWorkgroupCount = affine::makeComposedFoldedAffineApply(
        rewriter, loc, prodExpr, workgroupCount);
    if (!maxWorkgroupCount) {
      maxWorkgroupCount = flatWorkgroupCount;
      continue;
    }
    maxWorkgroupCount =
        rewriter
            .create<arith::MaxUIOp>(loc,
                                    getValueOrCreateConstantIndexOp(
                                        rewriter, loc, maxWorkgroupCount),
                                    getValueOrCreateConstantIndexOp(
                                        rewriter, loc, flatWorkgroupCount))
            .getResult();
  }
  OpFoldResult one = rewriter.getIndexAttr(1);
  // The order of dimensions expected by `lowerWorkgroupCountFromSliceOp`
  // is {z, y, x}.
  SmallVector<OpFoldResult> workgroupCount = {one, one, maxWorkgroupCount};
  return lowerWorkgroupCountFromSliceOp(rewriter, funcOp, workgroupCount);
}

static LogicalResult
resolveWorkgroupForAll(RewriterBase &rewriter, FunctionOpInterface funcOp,
                       IREE::Codegen::WorkgroupId deLinearizeFrom) {
  Region &body = funcOp.getFunctionBody();

  if (body.empty()) {
    return success();
  }

  auto forAllOps = body.getOps<scf::ForallOp>();
  SmallVector<scf::ForallOp> workgroupForAllOps =
      llvm::filter_to_vector(forAllOps, [&](scf::ForallOp forAllOp) {
        auto mapping = forAllOp.getMapping();
        if (!mapping) {
          return false;
        }
        if (!llvm::all_of(mapping.value(), [](Attribute attr) {
              return isa<IREE::Codegen::WorkgroupMappingAttr>(attr);
            })) {
          return false;
        }
        return true;
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

  // If there are multiple forall ops, then we need to fully linearize and
  // delinearize the workgroup IDs, because the iteration spaces may not match.
  if (workgroupForAllOps.size() > 1) {
    deLinearizeFrom = IREE::Codegen::WorkgroupId::IdX;
  }
  for (auto [idx, forallOp] : llvm::enumerate(workgroupForAllOps)) {
    // If there are multiple foralls, then the workgroup IDs come in as a single
    // linearized ID, because the shapes of each forall might not be compatible.
    if (failed(resolveWorkgroupForAll(rewriter, forallOp, deLinearizeFrom))) {
      return failure();
    }
  }
  return success();
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
    return int64_t();
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

  auto exportOps = variantOp.getOps<IREE::HAL::ExecutableExportOp>();

  // reconciliation for multiple export ops is unsupported.
  if (!llvm::hasSingleElement(exportOps)) {
    return;
  }
  auto exportOp = *exportOps.begin();
  IRRewriter rewriter(&getContext());

  SmallVector<IREE::Codegen::TranslationInfoAttr> translationInfos;
  auto walkResult =
      innerModuleOp->walk([&](FunctionOpInterface funcOp) -> WalkResult {
        // Resolve workgroup distribution related `scf.forall` ops.
        if (failed(resolveWorkgroupForAll(rewriter, funcOp, distributeAlong))) {
          return failure();
        }

        auto translationInfo = getTranslationInfo(funcOp);
        if (!translationInfo) {
          return WalkResult::advance();
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
        return WalkResult::advance();
      });
  if (walkResult.wasInterrupted()) {
    variantOp.emitOpError(
        "failed in iree-codegen-reconcile-translation-info pass");
    return signalPassFailure();
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
  for (auto [index, size] : llvm::enumerate(reconciledWorkgroupSize.value())) {
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
  if (reconciledSubgroupSize.value() != int64_t()) {
    exportOp.setSubgroupSizeAttr(
        rewriter.getIndexAttr(reconciledSubgroupSize.value()));
  }

  // Erase all the lowering configs and translation infos.
  innerModuleOp->walk([](Operation *op) {
    if (auto funcOp = dyn_cast<FunctionOpInterface>(op)) {
      eraseTranslationInfo(funcOp);
    }
    eraseLoweringConfig(op);
  });
}

} // namespace mlir::iree_compiler

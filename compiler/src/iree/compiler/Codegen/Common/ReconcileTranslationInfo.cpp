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

#include "iree/compiler/Codegen/Common/PassDetail.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
namespace mlir::iree_compiler {

namespace {

class ReconcileTranslationInfoPass
    : public ReconcileTranslationInfoBase<ReconcileTranslationInfoPass> {
public:
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
  SmallVector<IREE::Codegen::WorkgroupMappingAttr> workgroupMappingAttrs;
  llvm::SmallDenseSet<IREE::Codegen::WorkgroupMappingAttr, 4> attrSet;
  for (auto attr : mappingAttr.value()) {
    auto typedAttr = dyn_cast<IREE::Codegen::WorkgroupMappingAttr>(attr);
    if (!attr) {
      return forallOp.emitOpError("expected all the mapping attribute to be of "
                                  "`WorkgroupMappingAttr` type");
    }
    if (attrSet.contains(typedAttr)) {
      return forallOp.emitOpError("illegal to repeat mapping specification");
    }
    attrSet.insert(typedAttr);
    workgroupMappingAttrs.push_back(typedAttr);
  }
  return workgroupMappingAttrs;
}

/// Return the procId and nprocs to use for each of the distributed loops,
/// derived from `hal.interface.workgroup.id/count`s.
static FailureOr<
    std::pair<SmallVector<OpFoldResult>, SmallVector<OpFoldResult>>>
getProcIdsAndNprocs(
    scf::ForallOp forallOp, RewriterBase &builder, Location loc,
    ArrayRef<IREE::Codegen::WorkgroupMappingAttr> workgroupMappings,
    ArrayRef<OpFoldResult> lowerBounds, ArrayRef<OpFoldResult> upperBounds,
    ArrayRef<OpFoldResult> steps) {
  if (workgroupMappings.size() != lowerBounds.size()) {
    return forallOp.emitOpError(
        "expected as many workgroup mapping attributes as number of loops");
  }
  SmallVector<OpFoldResult> procId(workgroupMappings.size(),
                                   builder.getIndexAttr(0));
  SmallVector<OpFoldResult> nprocs(workgroupMappings.size(),
                                   builder.getIndexAttr(1));
  SmallVector<OpFoldResult> loopExtents(workgroupMappings.size());
  int64_t maxDelinearizedDim = 0;
  AffineExpr s0, s1, s2;
  bindSymbols(builder.getContext(), s0, s1, s2);
  AffineExpr extentExpr = (s1 - s0).ceilDiv(s2);
  for (auto [index, workgroupMapping, lowerBound, upperBound, step] :
       llvm::enumerate(workgroupMappings, lowerBounds, upperBounds, steps)) {
    switch (workgroupMapping.getId()) {
    case IREE::Codegen::WorkgroupId::IdX:
      procId[index] =
          builder.create<IREE::HAL::InterfaceWorkgroupIDOp>(loc, 0).getResult();
      nprocs[index] =
          builder.create<IREE::HAL::InterfaceWorkgroupCountOp>(loc, 0)
              .getResult();
      break;
    case IREE::Codegen::WorkgroupId::IdY:
      procId[index] =
          builder.create<IREE::HAL::InterfaceWorkgroupIDOp>(loc, 1).getResult();
      nprocs[index] =
          builder.create<IREE::HAL::InterfaceWorkgroupCountOp>(loc, 1)
              .getResult();
      break;
    case IREE::Codegen::WorkgroupId::IdZ: {
      int64_t delinearizedDim = workgroupMapping.getDelinearizedDim();
      maxDelinearizedDim = std::max(delinearizedDim, maxDelinearizedDim);
      loopExtents[delinearizedDim] = affine::makeComposedFoldedAffineApply(
          builder, loc, extentExpr, {lowerBound, upperBound, step});
      break;
    }
    }
  }

  if (loopExtents.empty()) {
    // No z-dimension handling. Just return procid and nprocs.
    return std::make_pair(procId, nprocs);
  }
  loopExtents.resize(maxDelinearizedDim);

  // If there is only one z-dim mapping, just use the ID directly.
  SmallVector<OpFoldResult> orderedDelinearizedDimIds;
  SmallVector<OpFoldResult> orderedDelinearizedNprocs;
  Value zDimId = builder.create<IREE::HAL::InterfaceWorkgroupIDOp>(loc, 2);
  OpFoldResult zNprocs =
      builder.create<IREE::HAL::InterfaceWorkgroupCountOp>(loc, 2).getResult();
  AffineMap minMap = AffineMap::get(0, 2, {s0, s1}, builder.getContext());
  AffineExpr ceilDivExpr = s0.ceilDiv(s1);
  if (loopExtents.size() != 1) {
    auto delinearizeOp = builder.create<affine::AffineDelinearizeIndexOp>(
        loc, zDimId, loopExtents);
    orderedDelinearizedDimIds = llvm::map_to_vector(
        delinearizeOp.getResults(), [](Value v) -> OpFoldResult { return v; });
    for (auto extent : loopExtents) {
      OpFoldResult currNprocs = affine::makeComposedFoldedAffineMin(
          builder, loc, minMap, {extent, zNprocs});
      orderedDelinearizedNprocs.push_back(currNprocs);
      zNprocs = affine::makeComposedFoldedAffineApply(builder, loc, ceilDivExpr,
                                                      {zNprocs, currNprocs});
    }
  } else {
    orderedDelinearizedDimIds = {OpFoldResult(zDimId)};
    orderedDelinearizedNprocs = {zNprocs};
  }

  // Map the ordered dimIds and procs back to the mapping attributes
  for (auto [index, workgroupMapping, lowerBound, upperBound, step] :
       llvm::enumerate(workgroupMappings, lowerBounds, upperBounds, steps)) {
    if (workgroupMapping.getId() == IREE::Codegen::WorkgroupId::IdZ) {
      procId[index] =
          orderedDelinearizedDimIds[workgroupMapping.getDelinearizedDim()];
      nprocs[index] =
          orderedDelinearizedNprocs[workgroupMapping.getDelinearizedDim()];
    }
  }
  return std::make_pair(procId, nprocs);
}

/// Resolve scf.forall operation by using the workgroup ID and counts.
static LogicalResult resolveForAllOp(RewriterBase &rewriter,
                                     scf::ForallOp forallOp) {
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

  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(forallOp);

  FailureOr<std::pair<SmallVector<OpFoldResult>, SmallVector<OpFoldResult>>>
      procInfo = getProcIdsAndNprocs(forallOp, rewriter, forallOp.getLoc(),
                                     workgroupMapping.value(), mixedLowerBound,
                                     mixedUpperBound, mixedStep);
  if (failed(procInfo)) {
    return failure();
  }
  ArrayRef<OpFoldResult> procId(procInfo->first);
  ArrayRef<OpFoldResult> nprocs(procInfo->second);

  /// For now this is assuming that number of workgroups is exactly equal to
  /// the iterations for each loop dimension. Just inline the forall body into
  /// the parent.
  Block *parentBlock = forallOp->getBlock();
  Block *remainingBlock =
      rewriter.splitBlock(parentBlock, Block::iterator(forallOp));
  auto argReplacements =
      getValueOrCreateConstantIndexOp(rewriter, forallOp.getLoc(), procId);
  Block *loopBody = forallOp.getBody();
  rewriter.eraseOp(loopBody->getTerminator());
  rewriter.mergeBlocks(loopBody, parentBlock, argReplacements);
  rewriter.mergeBlocks(remainingBlock, parentBlock, ValueRange{});
  rewriter.eraseOp(forallOp);
  return success();
}

static LogicalResult resolveWorkgroupCount(RewriterBase &rewriter,
                                           mlir::FunctionOpInterface funcOp,
                                           scf::ForallOp forAllOp) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(forAllOp);
  SmallVector<OpFoldResult> lowerBounds = forAllOp.getMixedLowerBound();
  SmallVector<OpFoldResult> upperBounds = forAllOp.getMixedUpperBound();
  SmallVector<OpFoldResult> steps = forAllOp.getMixedStep();
  SmallVector<OpFoldResult> workgroupCount(lowerBounds.size());
  AffineExpr s0, s1, s2;
  bindSymbols(rewriter.getContext(), s0, s1, s2);
  AffineExpr countExpr = (s1 - s0).ceilDiv(s2);
  for (auto [index, lb, ub, step] :
       llvm::enumerate(lowerBounds, upperBounds, steps)) {
    workgroupCount[index] = affine::makeComposedFoldedAffineApply(
        rewriter, forAllOp.getLoc(), countExpr, {lb, ub, step});
  }
  return lowerWorkgroupCountFromSliceOp(rewriter, funcOp, workgroupCount);
}

static LogicalResult resolveForallOps(RewriterBase &rewriter,
                                      FunctionOpInterface funcOp) {
  Region &body = funcOp.getFunctionBody();

  if (body.empty()) {
    return success();
  }

  if (!llvm::hasSingleElement(body)) {
    return funcOp.emitOpError("unhandled function with multiple blocks");
  }

  auto forAllOps = body.getOps<scf::ForallOp>();
  if (forAllOps.empty()) {
    return success();
  }
  SmallVector<scf::ForallOp> workgroupForAllOps = llvm::to_vector(
      llvm::make_filter_range(forAllOps, [&](scf::ForallOp forAllOp) {
        auto mapping = forAllOp.getMapping();
        if (!mapping) {
          return false;
        }
        if (!llvm::all_of(mapping.value(), [](Attribute attr) {
              return isa<IREE::Codegen::WorkgroupIdAttr>(attr);
            })) {
          return false;
        }
        return true;
      }));

  if (!llvm::hasSingleElement(workgroupForAllOps)) {
    return funcOp.emitOpError(
        "unhandled resolution of multiple scf.forall ops withing the function");
  }

  scf::ForallOp forallOp = *forAllOps.begin();
  if (failed(resolveWorkgroupCount(rewriter, funcOp, forallOp))) {
    return failure();
  }

  return resolveForAllOp(rewriter, *forAllOps.begin());
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
  if (!llvm::hasSingleElement(exportOps)) {
    variantOp.emitOpError("reconciliation for multiple export ops unsupported");
    return signalPassFailure();
  }
  auto exportOp = *exportOps.begin();
  IRRewriter rewriter(&getContext());

  SmallVector<IREE::Codegen::TranslationInfoAttr> translationInfos;
  innerModuleOp->walk([&](FunctionOpInterface funcOp) {
    auto translationInfo = getTranslationInfo(funcOp);
    if (!translationInfo) {
      return;
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
  });

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

std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createReconcileTranslationInfoPass() {
  return std::make_unique<ReconcileTranslationInfoPass>();
}

} // namespace mlir::iree_compiler

// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/PCF/IR/PCF.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_COMBINESPLITKWORKGROUPLOOPPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

/// Pattern to combine scf.forall with split-k reduction mapping containing
/// a pcf.loop into a single pcf.generic operation.
struct CombineSplitKWorkgroupLoopPattern
    : public OpRewritePattern<scf::ForallOp> {
  using OpRewritePattern<scf::ForallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForallOp forallOp,
                                PatternRewriter &rewriter) const override;

private:
  /// Validate that the forall has split-k reduction mapping.
  LogicalResult matchSplitKMapping(scf::ForallOp forallOp) const;

  /// Validate the scf.forall.in_parallel terminator and collect the pcf.loop.
  LogicalResult matchTerminator(scf::ForallOp forallOp,
                                IREE::PCF::LoopOp &loopOp) const;

  /// Validate the pcf.loop operation.
  LogicalResult matchPCFLoop(scf::ForallOp forallOp,
                             IREE::PCF::LoopOp loopOp) const;

  /// Collect and validate pcf.write_slice ops.
  LogicalResult
  collectWriteSlices(IREE::PCF::LoopOp loopOp,
                     DenseMap<unsigned, SmallVector<IREE::PCF::WriteSliceOp>>
                         &writesByResult) const;
};

struct CombineSplitKWorkgroupLoopPass final
    : public impl::CombineSplitKWorkgroupLoopPassBase<
          CombineSplitKWorkgroupLoopPass> {
  using Base::Base;
  void runOnOperation() override;
};

} // namespace

LogicalResult CombineSplitKWorkgroupLoopPattern::matchSplitKMapping(
    scf::ForallOp forallOp) const {
  std::optional<ArrayAttr> maybeMapping = forallOp.getMappingAttr();
  if (!maybeMapping || !maybeMapping.value() || maybeMapping.value().empty()) {
    return failure();
  }

  // Check that all mappings are split_reduction_mapping.
  return success(
      llvm::all_of(maybeMapping.value(),
                   llvm::IsaPred<IREE::LinalgExt::SplitReductionMappingAttr>));
}

LogicalResult CombineSplitKWorkgroupLoopPattern::matchTerminator(
    scf::ForallOp forallOp, IREE::PCF::LoopOp &loopOp) const {
  auto terminator =
      cast<scf::InParallelOp>(forallOp.getRegion().front().getTerminator());

  IREE::PCF::LoopOp foundLoop = nullptr;

  for (Operation &op : terminator.getYieldingOps()) {
    // Check a) All ops are tensor.parallel_insert_slice.
    auto insertSliceOp = dyn_cast<tensor::ParallelInsertSliceOp>(&op);
    if (!insertSliceOp) {
      return failure();
    }

    // Check b) Source is from a pcf.loop result.
    auto loopResult = dyn_cast<OpResult>(insertSliceOp.getSource());
    if (!loopResult) {
      return failure();
    }

    auto currentLoop = dyn_cast<IREE::PCF::LoopOp>(loopResult.getOwner());
    if (!currentLoop) {
      return failure();
    }

    // All inserts must reference the same pcf.loop.
    if (foundLoop && foundLoop != currentLoop) {
      return failure();
    }
    foundLoop = currentLoop;

    // Check c) Destination is a shared_out of the forall.
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

  loopOp = foundLoop;
  return success();
}

LogicalResult CombineSplitKWorkgroupLoopPattern::matchPCFLoop(
    scf::ForallOp forallOp, IREE::PCF::LoopOp loopOp) const {
  // Check scope is workgroup with linearize.
  auto workgroupScope =
      dyn_cast<IREE::Codegen::WorkgroupAttr>(loopOp.getScope());
  if (!workgroupScope || !workgroupScope.getLinearize()) {
    return failure();
  }

  // Check single count argument.
  if (loopOp.getCount().size() != 1) {
    return failure();
  }

  // Check count dominates the forall.
  Value countValue = loopOp.getCount()[0];
  if (auto defOp = countValue.getDefiningOp()) {
    if (!defOp->isBeforeInBlock(forallOp)) {
      return failure();
    }
  }

  // Verify loop is last op before terminator.
  Operation *lastOp =
      forallOp.getRegion().front().getTerminator()->getPrevNode();
  if (lastOp != loopOp) {
    return failure();
  }

  return success();
}

LogicalResult CombineSplitKWorkgroupLoopPattern::collectWriteSlices(
    IREE::PCF::LoopOp loopOp,
    DenseMap<unsigned, SmallVector<IREE::PCF::WriteSliceOp>> &writesByResult)
    const {
  for (auto [idx, refArg] : llvm::enumerate(loopOp.getRegionRefArgs())) {
    // Check sync scope is sync_on_parent.
    auto srefType = cast<IREE::PCF::ShapedRefType>(refArg.getType());
    auto syncScope = srefType.getSyncScope();
    auto syncOnParent =
        dyn_cast_or_null<IREE::PCF::SyncOnParentAttr>(syncScope);
    if (!syncOnParent) {
      return failure();
    }

    // Collect write_slice ops that write to this ref arg.
    for (Operation *user : refArg.getUsers()) {
      auto writeOp = dyn_cast<IREE::PCF::WriteSliceOp>(user);
      if (!writeOp) {
        return failure(); // All users must be write_slice.
      }
      writesByResult[idx].push_back(writeOp);
    }
  }

  return success();
}

LogicalResult CombineSplitKWorkgroupLoopPattern::matchAndRewrite(
    scf::ForallOp forallOp, PatternRewriter &rewriter) const {
  // Step 1: Match split-k reduction mapping.
  if (failed(matchSplitKMapping(forallOp))) {
    return rewriter.notifyMatchFailure(forallOp,
                                       "does not have split_reduction_mapping");
  }

  // Step 2: Validate terminator and collect pcf.loop.
  IREE::PCF::LoopOp loopOp;
  if (failed(matchTerminator(forallOp, loopOp))) {
    return rewriter.notifyMatchFailure(
        forallOp, "terminator does not have valid parallel_insert_slice ops "
                  "from pcf.loop");
  }

  // Step 3: Validate pcf.loop.
  if (failed(matchPCFLoop(forallOp, loopOp))) {
    return rewriter.notifyMatchFailure(forallOp,
                                       "pcf.loop does not meet requirements");
  }

  // Step 4: Collect write_slice ops.
  DenseMap<unsigned, SmallVector<IREE::PCF::WriteSliceOp>> writesByResult;
  if (failed(collectWriteSlices(loopOp, writesByResult))) {
    return rewriter.notifyMatchFailure(
        forallOp, "pcf.loop results have invalid write_slice usage");
  }

  Location loc = forallOp.getLoc();

  // Step 7: Replace RegionIterArgs with initial values.
  for (auto [iterArg, init] :
       llvm::zip(forallOp.getRegionIterArgs(), forallOp.getOutputs())) {
    for (OpOperand &use : llvm::make_early_inc_range(iterArg.getUses())) {
      if (!isa<tensor::ParallelInsertSliceOp>(use.getOwner())) {
        use.set(init);
      }
    }
  }

  // Step 8: Extract loop count (we'll need it later).
  Value loopCount = loopOp.getCount()[0];

  // Step 8.5: Compute total iteration count for forall before creating generic.
  rewriter.setInsertionPoint(forallOp);
  Value forallTotalIters = nullptr;
  for (auto [lb, ub, step] :
       llvm::zip(forallOp.getMixedLowerBound(), forallOp.getMixedUpperBound(),
                 forallOp.getMixedStep())) {
    Value lbVal = getValueOrCreateConstantIndexOp(rewriter, loc, lb);
    Value ubVal = getValueOrCreateConstantIndexOp(rewriter, loc, ub);
    Value stepVal = getValueOrCreateConstantIndexOp(rewriter, loc, step);

    Value range =
        arith::SubIOp::create(rewriter, loc, ubVal, lbVal).getResult();
    Value count =
        arith::CeilDivSIOp::create(rewriter, loc, range, stepVal).getResult();

    if (!forallTotalIters) {
      forallTotalIters = count;
    } else {
      forallTotalIters =
          arith::MulIOp::create(rewriter, loc, forallTotalIters, count)
              .getResult();
    }
  }

  // Step 8.6: Compute total workgroup count = forallTotalIters * loopCount.
  Value totalWorkgroupCount =
      arith::MulIOp::create(rewriter, loc, forallTotalIters, loopCount)
          .getResult();

  // Step 8.7: Create workgroup count hint.
  [[maybe_unused]] LogicalResult hintRes = createWorkgroupCountHint(
      rewriter, loc, {totalWorkgroupCount}, /*maxWorkgroupParallelDims=*/1,
      /*reverse=*/false);
  assert(succeeded(hintRes) &&
         "Unexpected failure to construct workgroup count hint");

  // Step 9: Create pcf.generic.
  auto genericOp = IREE::PCF::GenericOp::create(
      rewriter, loc,
      /*resultTypes=*/forallOp.getResultTypes(),
      /*scope=*/loopOp.getScope(),
      /*inits=*/forallOp.getOutputs(),
      /*dynamic_sizes=*/ValueRange{},
      /*is_tied=*/SmallVector<bool>(forallOp.getNumResults(), true),
      /*num_iterators=*/1);

  // Step 9.5: Set sync scope to parent only for the pcf.generic sref arguments.
  Attribute syncScope = IREE::PCF::SyncOnParentAttr::get(rewriter.getContext());
  for (auto regionRefArg : genericOp.getRegionRefArgs()) {
    auto srefType = cast<IREE::PCF::ShapedRefType>(regionRefArg.getType());
    auto newSrefType = IREE::PCF::ShapedRefType::get(
        rewriter.getContext(), srefType.getShape(), srefType.getElementType(),
        srefType.getScope(), syncScope);
    regionRefArg.setType(newSrefType);
  }

  // Get forall body and terminator before we start transforming
  Block *forallBody = &forallOp.getRegion().front();
  auto terminator = cast<scf::InParallelOp>(forallBody->getTerminator());

  // Step 10: Move insertion point into generic body before computing iteration
  // counts
  Block *genericBody = &genericOp.getRegion().front();
  rewriter.setInsertionPointToStart(genericBody);

  //  Step 11: Compute scf.forall iteration counts (now inside generic).
  SmallVector<Value> iterCounts;
  Value totalIters = nullptr;

  for (auto [lb, ub, step] :
       llvm::zip(forallOp.getMixedLowerBound(), forallOp.getMixedUpperBound(),
                 forallOp.getMixedStep())) {
    // count = ceildiv(ub - lb, step).
    Value lbVal = getValueOrCreateConstantIndexOp(rewriter, loc, lb);
    Value ubVal = getValueOrCreateConstantIndexOp(rewriter, loc, ub);
    Value stepVal = getValueOrCreateConstantIndexOp(rewriter, loc, step);

    Value range =
        arith::SubIOp::create(rewriter, loc, ubVal, lbVal).getResult();
    Value count =
        arith::CeilDivSIOp::create(rewriter, loc, range, stepVal).getResult();
    iterCounts.push_back(count);

    if (!totalIters) {
      totalIters = count;
    } else {
      totalIters =
          arith::MulIOp::create(rewriter, loc, totalIters, count).getResult();
    }
  }

  // Step 12: Delinearize pcf.generic id into 2D (quotient and remainder).

  BlockArgument linearId = genericOp.getIdArgs()[0];

  // Compute: forallLinearId = linearId / loopCount, pcfLoopLinearId = linearId
  // % loopCount
  Value forallLinearId =
      arith::DivSIOp::create(rewriter, loc, linearId, loopCount).getResult();
  Value pcfLoopLinearId =
      arith::RemSIOp::create(rewriter, loc, linearId, loopCount).getResult();

  // Step 13: Create outer scf.for for the forall dimensions.
  Value totalWorkers = genericOp.getCountArgs()[0];
  Value outerStep =
      arith::CeilDivSIOp::create(rewriter, loc, totalWorkers, loopCount)
          .getResult();

  auto outerFor =
      scf::ForOp::create(rewriter, loc, forallLinearId, totalIters, outerStep);

  // Step 14: Compute actual forall induction vars and inline body.
  rewriter.setInsertionPointToStart(outerFor.getBody());

  auto forallIvDelinOp = affine::AffineDelinearizeIndexOp::create(
      rewriter, loc, outerFor.getInductionVar(), iterCounts);
  ValueRange delinearizedIvs = forallIvDelinOp.getMultiIndex();

  // Apply steps and offsets: iv = delinearized * step + lb.
  IRMapping forallMapping;
  SmallVector<Value> actualForallIvs;
  for (auto [delinIv, lb, step, forallIv] :
       llvm::zip(delinearizedIvs, forallOp.getMixedLowerBound(),
                 forallOp.getMixedStep(), forallOp.getInductionVars())) {
    Value lbVal = getValueOrCreateConstantIndexOp(rewriter, loc, lb);
    Value stepVal = getValueOrCreateConstantIndexOp(rewriter, loc, step);
    Value scaled =
        arith::MulIOp::create(rewriter, loc, delinIv, stepVal).getResult();
    Value actualIv =
        arith::AddIOp::create(rewriter, loc, scaled, lbVal).getResult();
    forallMapping.map(forallIv, actualIv);
  }

  // Map iter args to generic ref args
  for (auto [iterArg, refArg] :
       llvm::zip(forallOp.getRegionIterArgs(), genericOp.getRegionRefArgs())) {
    forallMapping.map(iterArg, refArg);
  }

  // Step 15: Collect information about parallel_insert_slice ops before
  // inlining. We need this information to compose write_slices after inlining.
  struct InsertSliceInfo {
    unsigned resultIdx;
    unsigned argIdx;
    SmallVector<OpFoldResult> offsets;
    SmallVector<OpFoldResult> sizes;
    SmallVector<OpFoldResult> strides;
  };
  SmallVector<InsertSliceInfo> insertSliceInfos;

  for (Operation &op : terminator.getYieldingOps()) {
    auto insertOp = cast<tensor::ParallelInsertSliceOp>(&op);
    auto loopResult = cast<OpResult>(insertOp.getSource());
    unsigned resultIdx = loopResult.getResultNumber();
    auto destArg = cast<BlockArgument>(insertOp.getDest());
    unsigned argIdx = destArg.getArgNumber() - forallOp.getRank();

    InsertSliceInfo info;
    info.resultIdx = resultIdx;
    info.argIdx = argIdx;
    info.offsets = insertOp.getMixedOffsets();
    info.sizes = insertOp.getMixedSizes();
    info.strides = insertOp.getMixedStrides();
    insertSliceInfos.push_back(info);
  }

  // Move forall body operations into outer for (except the terminator)
  Block *outerForBody = outerFor.getBody();
  Operation *outerForTerminator = outerForBody->getTerminator();

  // Move all operations except the terminator before the outerFor's yield
  for (Operation &op :
       llvm::make_early_inc_range(forallBody->without_terminator())) {
    op.moveBefore(outerForTerminator);
  }

  // Now remap the block arguments in the moved operations
  for (BlockArgument arg : forallBody->getArguments()) {
    arg.replaceAllUsesWith(forallMapping.lookup(arg));
  }

  // The pcf.loop should now be in the outer for body - find it
  IREE::PCF::LoopOp movedLoopOp = nullptr;
  for (Operation &op : outerForBody->without_terminator()) {
    if (auto loop = dyn_cast<IREE::PCF::LoopOp>(&op)) {
      movedLoopOp = loop;
      break;
    }
  }
  assert(movedLoopOp && "Failed to find moved pcf.loop");

  // Step 16: Compose tensor.parallel_insert_slice with pcf.write_slice.
  // Process the terminator BEFORE erasing the forall
  for (Operation &op :
       llvm::make_early_inc_range(terminator.getYieldingOps())) {
    auto insertOp = cast<tensor::ParallelInsertSliceOp>(&op);

    // Find the corresponding insert slice info based on the loop result
    auto loopResult = cast<OpResult>(insertOp.getSource());
    unsigned resultIdx = loopResult.getResultNumber();

    // Find the matching insertSliceInfo
    const InsertSliceInfo *matchingInfo = nullptr;
    for (const InsertSliceInfo &info : insertSliceInfos) {
      if (info.resultIdx == resultIdx) {
        matchingInfo = &info;
        break;
      }
    }
    assert(matchingInfo && "Failed to find matching insert slice info");

    BlockArgument genericRefArg =
        genericOp.getRegionRefArgs()[matchingInfo->argIdx];
    BlockArgument movedRefArg = movedLoopOp.getRegionRefArgs()[resultIdx];

    // Find all write_slice ops that write to this ref arg and compose them
    for (Operation *user : llvm::make_early_inc_range(movedRefArg.getUsers())) {
      auto writeOp = dyn_cast<IREE::PCF::WriteSliceOp>(user);
      if (!writeOp) {
        continue;
      }

      rewriter.setInsertionPoint(writeOp);

      SmallVector<OpFoldResult> writeOffsets = writeOp.getMixedOffsets();
      SmallVector<OpFoldResult> writeStrides = writeOp.getMixedStrides();
      SmallVector<OpFoldResult> writeSizes = writeOp.getMixedSizes();

      // The write_slice writes rank-N tensor, but the final sref is rank-M
      // where M > N. We need to expand the source to match the sref rank by
      // adding unit dims.

      auto sourceType = cast<RankedTensorType>(writeOp.getSource().getType());
      auto srefType = cast<IREE::PCF::ShapedRefType>(genericRefArg.getType());
      int64_t srefRank = srefType.getRank();
      int64_t sourceRank = sourceType.getRank();

      SmallVector<int64_t> expandedShape;
      SmallVector<ReassociationIndices> reassociation;

      // For expand_shape, reassociation groups OUTPUT dims that came from same
      // INPUT dim We keep the first sourceRank-1 dims as-is, and expand the
      // last source dim to include itself plus (srefRank - sourceRank) unit
      // dims

      // First sourceRank-1 dimensions map 1:1
      for (int64_t i = 0; i < sourceRank - 1; ++i) {
        expandedShape.push_back(sourceType.getDimSize(i));
        reassociation.push_back({i});
      }

      // Last input dimension expands to include itself plus all unit dims
      ReassociationIndices lastGroup;
      if (sourceRank > 0) {
        expandedShape.push_back(sourceType.getDimSize(sourceRank - 1));
        lastGroup.push_back(sourceRank - 1);
      }

      // Add unit dimensions to reach sref rank
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
      Value expandedSource = tensor::ExpandShapeOp::create(
                                 rewriter, writeOp.getLoc(), expandedType,
                                 writeOp.getSource(), reassociation)
                                 .getResult();

      // Compose offsets, sizes, and strides.
      // The write dimensions map to the first N dimensions of insert,
      // and we need to add the insert's base offsets to the write offsets.
      // Then append the remaining insert dimensions.
      SmallVector<OpFoldResult> composedOffsets;
      SmallVector<OpFoldResult> composedSizes;
      SmallVector<OpFoldResult> composedStrides;

      // First writeRank dimensions: add write and insert offsets
      for (size_t i = 0; i < writeOffsets.size(); ++i) {
        // Compose offset: writeOffset + insertOffset
        Value writeOff = getValueOrCreateConstantIndexOp(
            rewriter, writeOp.getLoc(), writeOffsets[i]);
        Value insertOff = getValueOrCreateConstantIndexOp(
            rewriter, writeOp.getLoc(), matchingInfo->offsets[i]);
        Value composedOff = arith::AddIOp::create(rewriter, writeOp.getLoc(),
                                                  writeOff, insertOff)
                                .getResult();
        composedOffsets.push_back(composedOff);

        // Size comes from write (it's a subset of the insert size)
        composedSizes.push_back(writeSizes[i]);

        // Stride: multiply write and insert strides
        Value writeStride = getValueOrCreateConstantIndexOp(
            rewriter, writeOp.getLoc(), writeStrides[i]);
        Value insertStride = getValueOrCreateConstantIndexOp(
            rewriter, writeOp.getLoc(), matchingInfo->strides[i]);
        Value composedStride = arith::MulIOp::create(rewriter, writeOp.getLoc(),
                                                     writeStride, insertStride)
                                   .getResult();
        composedStrides.push_back(composedStride);
      }

      // Remaining dimensions from insert (these are the "extra" dimensions)
      for (size_t i = writeOffsets.size(); i < matchingInfo->offsets.size();
           ++i) {
        composedOffsets.push_back(matchingInfo->offsets[i]);
        composedSizes.push_back(matchingInfo->sizes[i]);
        composedStrides.push_back(matchingInfo->strides[i]);
      }

      // Replace old write_slice with composed one
      rewriter.replaceOpWithNewOp<IREE::PCF::WriteSliceOp>(
          writeOp, expandedSource, genericRefArg, composedOffsets,
          composedSizes, composedStrides);
    }

    // Erase the parallel_insert_slice after composing all write_slices for it
    rewriter.eraseOp(insertOp);
  }

  // Replace all uses of the forall results with the generic results
  for (auto [forallResult, genericResult] :
       llvm::zip(forallOp.getResults(), genericOp.getResults())) {
    rewriter.replaceAllUsesWith(forallResult, genericResult);
  }

  // Now erase the forall
  rewriter.eraseOp(forallOp);

  // After replacing the forall, we can erase the terminator
  // (though it should already be gone as part of forall deletion)
  // rewriter.eraseOp(terminator); // Not needed - forall deletion handles it

  // Step 17: Convert the moved pcf.loop to inner scf.for.
  rewriter.setInsertionPoint(movedLoopOp);

  // Step = min(totalWorkers - loopCount * forallLinearId, loopCount).
  Value product =
      arith::MulIOp::create(rewriter, loc, loopCount, forallLinearId)
          .getResult();
  Value diff =
      arith::SubIOp::create(rewriter, loc, totalWorkers, product).getResult();
  Value innerStep =
      arith::MinSIOp::create(rewriter, loc, diff, loopCount).getResult();

  auto innerFor =
      scf::ForOp::create(rewriter, loc, pcfLoopLinearId, loopCount, innerStep);

  // Move pcf.loop body operations (except return) into inner for.
  Block *loopBody = &movedLoopOp.getRegion().front();
  Block *innerForBody = innerFor.getBody();

  // Replace the loop's id arg with the inner for's induction variable
  movedLoopOp.getIdArgsMutable()[0].replaceAllUsesWith(
      innerFor.getInductionVar());

  // Move operations from loop body to inner for (excluding the pcf.return
  // terminator)
  innerForBody->getOperations().splice(
      std::prev(innerForBody->end()), loopBody->getOperations(),
      loopBody->begin(), std::prev(loopBody->end()));

  // Erase moved pcf.loop.
  rewriter.eraseOp(movedLoopOp);

  // Add terminator to the generic's region.
  rewriter.setInsertionPointToEnd(genericBody);
  IREE::PCF::ReturnOp::create(rewriter, loc);

  return success();
}

void CombineSplitKWorkgroupLoopPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<CombineSplitKWorkgroupLoopPattern>(&getContext());
  walkAndApplyPatterns(getOperation(), std::move(patterns));
}

} // namespace mlir::iree_compiler

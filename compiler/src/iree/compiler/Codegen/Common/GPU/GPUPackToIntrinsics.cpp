// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/GPULoweringConfigUtils.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUInterfaces.h"
#include "iree/compiler/Codegen/Dialect/GPU/Transforms/Passes.h"
#include "iree/compiler/Codegen/Dialect/GPU/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUPACKTOINTRINSICSPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {
struct GPUPackToIntrinsicsPass final
    : impl::GPUPackToIntrinsicsPassBase<GPUPackToIntrinsicsPass> {
  void runOnOperation() override;
};
} // namespace

linalg::LinalgOp removeUnitExtentDimsfromMaps(linalg::LinalgOp baseLinalgOp,
                                              RewriterBase &rewriter) {
  auto linalgOp = cast<linalg::GenericOp>(baseLinalgOp);
  SmallVector<AffineMap> indexingMaps = linalgOp.getIndexingMapsArray();
  if (indexingMaps.empty())
    return baseLinalgOp;

  // 1. Check if any of the iteration dimensions are unit-trip count. They will
  //    end up being unit-trip count if they are used to index into a unit-dim
  //    tensor/memref.
  AffineMap invertedMap =
      inversePermutation(concatAffineMaps(indexingMaps, rewriter.getContext()));
  auto iteratorTypes = linalgOp.getIteratorTypesArray();
  if (!invertedMap) {
    return baseLinalgOp;
  }

  SmallVector<int64_t> allShapesSizes;
  for (OpOperand &opOperand : linalgOp->getOpOperands())
    llvm::append_range(allShapesSizes, linalgOp.getShape(&opOperand));

  llvm::SmallDenseSet<unsigned> unitDims;
  for (const auto &expr : enumerate(invertedMap.getResults())) {
    if (AffineDimExpr dimExpr = dyn_cast<AffineDimExpr>(expr.value())) {
      if (allShapesSizes[dimExpr.getPosition()] == 1 &&
          iteratorTypes[expr.index()] == utils::IteratorType::reduction)
        unitDims.insert(expr.index());
    }
  }

  SmallVector<AffineMap> newIndexingMaps;
  for (auto indexingMap : indexingMaps) {
    SmallVector<AffineExpr> newExprs;
    for (auto [idx, e] : llvm::enumerate(indexingMap.getResults())) {
      AffineExpr newExpr = e;
      if (auto binaryExpr = llvm::dyn_cast<AffineBinaryOpExpr>(e)) {
        for (auto s : unitDims) {
          if (binaryExpr.getLHS().isFunctionOfDim(s)) {
            newExpr = binaryExpr.getRHS();
          }
          if (binaryExpr.getRHS().isFunctionOfDim(s)) {
            newExpr = binaryExpr.getLHS();
          }
        }
      }
      newExprs.push_back(newExpr);
    }
    newIndexingMaps.push_back(AffineMap::get(indexingMap.getNumDims(), 0,
                                             newExprs, rewriter.getContext()));
  }
  auto newOp = rewriter.create<linalg::GenericOp>(
      linalgOp.getLoc(), linalgOp.getDpsInits().getType(),
      linalgOp.getDpsInputs(), linalgOp.getDpsInits(), newIndexingMaps,
      iteratorTypes, /*bodyBuild=*/nullptr,
      linalg::getPrunedAttributeList(linalgOp));
  rewriter.inlineRegionBefore(linalgOp.getRegion(), newOp.getRegion(),
                              newOp.getRegion().begin());
  rewriter.replaceOp(linalgOp, newOp.getResults());
  return newOp;
}

LogicalResult packToIntrinsic(linalg::LinalgOp linalgOp,
                              RewriterBase &rewriter) {
  auto loweringConfig =
      getLoweringConfig<IREE::GPU::LoweringConfigAttr>(linalgOp);
  assert(loweringConfig && "Packing unconfigured op");

  IREE::GPU::MmaInterfaceAttr kind = getMmaKind(loweringConfig);
  assert(kind && "Packing op without mma kind");

  FailureOr<linalg::ContractionDimensions> contractionDims =
      linalg::inferContractionDims(linalgOp);
  if (failed(contractionDims)) {
    return rewriter.notifyMatchFailure(linalgOp,
                                       "failed to infer contraction dims");
  }

  if (contractionDims->m.empty() || contractionDims->n.empty() ||
      contractionDims->k.empty()) {
    return rewriter.notifyMatchFailure(
        linalgOp, "contraction like operation missing critical dimension");
  }

  auto zero = rewriter.getIndexAttr(0);
  SmallVector<OpFoldResult> packedSizes(linalgOp.getNumLoops(), zero);

  auto [m, n, k] = kind.getMNKShape();
  packedSizes[contractionDims->m.back()] = rewriter.getIndexAttr(m);
  packedSizes[contractionDims->n.back()] = rewriter.getIndexAttr(n);
  packedSizes[contractionDims->k.back()] = rewriter.getIndexAttr(k);
  FailureOr<linalg::PackResult> maybeResult =
      linalg::pack(rewriter, linalgOp, packedSizes);
  if (failed(maybeResult)) {
    return rewriter.notifyMatchFailure(linalgOp, "packing failed");
  }
  setLoweringConfig(maybeResult->packedLinalgOp, loweringConfig);
  return success();
}

struct ConvertToMultiMma final : OpInterfaceRewritePattern<linalg::LinalgOp> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;
  LogicalResult matchAndRewrite(linalg::LinalgOp linalgOp,
                                PatternRewriter &rewriter) const override {
    auto loweringConfig =
        getLoweringConfig<IREE::GPU::LoweringConfigAttr>(linalgOp);
    if (!loweringConfig) {
      return failure();
    }
    IREE::GPU::MmaInterfaceAttr kind = getMmaKind(loweringConfig);
    if (!kind) {
      return failure();
    }
    if (failed(convertContractionToInnerTiledMma(rewriter, linalgOp, kind))) {
      return failure();
    }
    return success();
  }
};

/// This pattern hoists pack & unpack ops out of scf.for op.
struct ExpandDestinationForOp final
    : OpRewritePattern<scf::YieldOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(scf::YieldOp yieldOp,
                                PatternRewriter &rewriter) const override {

    if (yieldOp->getParentOp()->hasAttr("expand_destination_hoisted"))
      return failure();
    Location loc = yieldOp.getLoc();
    // MLIRContext *ctx = getContext();
    auto unpackOp =
        yieldOp.getOperand(0).getDefiningOp<linalg::UnPackOp>();

    llvm::dbgs()<<"\n---------------------------------------------------------------------------\nunpackOp : "<<unpackOp<<"\n\n";
    // No unpack op to hoist out.
    if (!unpackOp)
      return failure();

    auto innerTiledOp = unpackOp.getOperand(0).getDefiningOp<IREE::Codegen::InnerTiledOp>();
    llvm::dbgs()<<"innerTiledOp : "<<innerTiledOp<<"\n\n";


    // // Ignore trivially foldable collapse ops.
    // if (collapseOp.getSrcType().getRank() ==
    //     collapseOp.getResultType().getRank()) {
    //   return failure();
    // }

    // Get the destination to expand.
    // Value insertDest = parallelInsertOp.getDest();
    Value yieldOpOperand = yieldOp.getOperand(0);
    llvm::dbgs()<<"yieldOpOperand is "<<yieldOpOperand<<"\n";

    // Get the enclosing scf.for op.
    // OpResult tiedResult = parallelInsertOp.getTiedOpResult();
    // int64_t tiedResultIdx = tiedResult.getResultNumber();
    auto parentOp = yieldOp->getParentOp();
    llvm::dbgs()<<"parent op is "<<parentOp<<"\n";
    // int64_t tiedResultIdx = tiedResult.getResultNumber();

    auto forOp = dyn_cast<scf::ForOp>(parentOp);
    if (!forOp)
      return failure();

    // // We only want this pattern if the forall op result is being written to a
    // // full slice. Otherwise the hoisted collapse op is not foldable.
    // for (Operation *foralluser : tiedResult.getUsers()) {
    //   auto storeOp =
    //       dyn_cast<IREE::TensorExt::DispatchTensorStoreOp>(foralluser);
    //   if (!storeOp)
    //     return failure();
    //   if (!isFullSlice(storeOp, storeOp.getTargetType(),
    //                    storeOp.getTargetDims())) {
    //     return failure();
    //   }
    // }

    // // This allows us to assume that the extract/inserts in the loop are
    // // disjoint and makes the application of this pattern safe.
    // if (!forallOpHasMappingType<IREE::Codegen::WorkgroupMappingAttr>(
    //         forallOp)) {
    //   return failure();
    // }
    // This pattern only supports for ops with single
    // output.
    SmallVector<Value> forOutputs(forOp.getResults());
    llvm::dbgs()<<"Number of Outputs is "<<forOutputs.size()<<"\n";

    // SmallVector<ReassociationIndices> reIndices =
    //     unpackOp.getReassociationIndices();
    // auto packedDestShape = unpackOp.getSource().getType().getShape();
    // SmallVector<int64_t> totalInnerSizes;
    // // Get the shape of the outer expand which will be the new destination
    // // of the scf.forall and the total size of inner dimensions per uncollapsed
    // // dimension.
    // if (failed(getExpandedShape(reIndices, collapseOp.getSrcType().getShape(),
    //                             insertDest, expandedDestShape,
    //                             totalInnerSizes))) {
    //   return failure();
    // }

    // // Verify that the users of destination are valid to expand and collect all
    // // such users.
    // SmallVector<tensor::ExtractSliceOp> expandableUsers;
    // if (failed(verifyAndCollectExpandableUsers(
    //         insertDest, collapseOp.getReassociationIndices(), parallelInsertOp,
    //         expandableUsers))) {
    //   return failure();
    // }

    // // Expand the users of the destination.
    // rewriter.setInsertionPointToStart(forOp.getBody());
    // expandVerifiedUsers(rewriter, loc, ctx, expandableUsers, totalInnerSizes,
    //                     reIndices, forallOp, parallelInsertOp);
    rewriter.setInsertionPoint(forOp);

    // -----------------------------------------------------
    // // Create the pack -> new scf.for -> unpack chain.
    // -----------------------------------------------------

    // auto packedDestShape = unpackOp.getSource().getType().getShape();
    // auto reducedSrcType =
    //     cast<RankedTensorType>(forOutputs[0].getType())
    //         .clone(packedDestShape);
    // auto packedDest = rewriter.create<linalg::PackOp>(
    //     loc, expandedDestType, forallOutputs[tiedResultIdx], reIndices);
    
    // auto reducedSrcType =
    //     RankedTensorType::Builder(forOutputs[0].getType());
    // auto reducedSrc = linalg::UnPackOp::createDestinationTensor(
    //     rewriter, loc, forOutputs[0], unpackOp.getOuterDimsPerm());

    // auto reducedDestType =
    //     RankedTensorType::Builder(unpackOp.getDestType());
    // auto reducedDest = tensor::createCanonicalRankReducingExtractSliceOp(
    //     rewriter, loc, unpackOp.getDest(), reducedDestType);


    Value input = linalg::PackOp::createDestinationTensor(
      rewriter, loc, forOp.getInitArgs()[0], unpackOp.getMixedTiles(),
      unpackOp.getInnerDimsPos(), unpackOp.getOuterDimsPerm());

    llvm::dbgs()<<"Debug\n";

    auto packedDest = rewriter.create<linalg::PackOp>(
        loc, forOp.getInitArgs()[0], input, unpackOp.getInnerDimsPos(), unpackOp.getMixedTiles(),
         /*padding=*/std::nullopt, unpackOp.getOuterDimsPerm());

    // forOutputs[0] = packedDest;
    llvm::dbgs()<<"packedDest is "<<packedDest<<"\n";
    llvm::dbgs()<<"packedDest result is "<<packedDest.getResult()<<"\n";

    // ---------------------------------------------------------------------------------------------------

    // The new vector init_args of the loop.
    // SmallVector<Value> newInitArgs;
    // for (Value initArg : forOp.getInitArgs()) {
    //   if (auto vectorInitArg = dyn_cast<RankedTensorType>(initArg)) {
    //     llvm::dbgs()<<"Entered If condition\n";
    //     initArg =packedDest.getResult();
    //   }
    //   newInitArgs.push_back(initArg);
    // }

    // auto initArgs = llvm::to_vector<8>(forOp.getInitArgs());
    // for(auto arg : initArgs)
    // {
    //   llvm::dbgs()<<"forop inititerargs "<<arg<<"\n";
    // }
    // llvm::dbgs()<<"forop inititerargs "<<initArgs[0]<<"\n";
    scf::ForOp newForOp = rewriter.create<scf::ForOp>(
        loc, forOp.getLowerBound(), forOp.getUpperBound(),
        forOp.getStep(), ValueRange{packedDest});

    llvm::dbgs()<<"New ForOp created is "<<newForOp<<"\n";    
    // auto collapsedResultOp = rewriter.create<tensor::CollapseShapeOp>(
    //     loc, cast<ShapedType>(forallOp->getResult(tiedResultIdx).getType()),
    //     newForallOp->getResult(tiedResultIdx), reIndices);
    // SmallVector<int64_t> newOuterDimsPerm(unpackOp.getOuterDimsPerm());
    
    // --------------------------------------------------------------------------------------------------
    llvm::dbgs()<<forOp.getResults().size()<<"\n";

    Value empty = linalg::UnPackOp::createDestinationTensor(
      rewriter, loc, newForOp.getResults()[0], unpackOp.getMixedTiles(),
      unpackOp.getInnerDimsPos(), unpackOp.getOuterDimsPerm());

    llvm::dbgs()<<"Debug-2"<<empty<<"\n";

    auto unpackedOutput = rewriter.create<linalg::UnPackOp>(
      loc, newForOp.getResults()[0], empty, unpackOp.getInnerDimsPos(),
      unpackOp.getMixedTiles(), unpackOp.getOuterDimsPerm());

    
    llvm::dbgs()<<"new unpack op is "<<unpackedOutput<<"\n";


    rewriter.modifyOpInPlace(
      yieldOp, [&]() { yieldOp->setOperand(0, innerTiledOp.getResult(0)); });

    rewriter.modifyOpInPlace(
      innerTiledOp, [&]() { innerTiledOp->setOperand(2, newForOp.getRegionIterArgs()[0]); });


    // Merge the old scf.for block which has the expanded users into the new
    // scf.forall which has the expanded destination.
    SmallVector<Value> ivs = {newForOp.getInductionVar()};
    SmallVector<Value> argReplacements(ivs);
    argReplacements.append(newForOp.getRegionIterArgs().begin(),
                           newForOp.getRegionIterArgs().end());


    // scf::YieldOp forTerminator = cast<scf::YieldOp>(newForOp.getBody()->getTerminator());
    // forTerminator.erase();
    rewriter.mergeBlocks(forOp.getBody(), newForOp.getBody(),
                         argReplacements);

    // rewriter.modifyOpInPlace(
    //   yieldOp, [&]() { yieldOp->setOperand(0, innerTiledOp.getResult(0)); });

    // Replaces the uses of the old scf.forall with the new scf.forall
    for (int idx = 0; idx < forOp->getNumResults(); ++idx) {
      if (idx == 0) {
        forOp->getResult(idx).replaceAllUsesWith(
            unpackedOutput->getResult(0));
      } else {
        forOp->getResult(idx).replaceAllUsesWith(
            newForOp->getResult(idx));
      }
    }
    llvm::dbgs()<<"Return Success\n";
    newForOp->setAttr("expand_destination_hoisted", rewriter.getUnitAttr());
    return success();
  }
};
// ############################################################################

void GPUPackToIntrinsicsPass::runOnOperation() {
  MLIRContext *context = &getContext();
  auto funcOp = getOperation();

  // Step 1. Pack candidate linalg ops to specified shapes.
  IRRewriter rewriter(funcOp);
  SmallVector<linalg::LinalgOp> packingCandidates;
  funcOp->walk([&](linalg::LinalgOp linalgOp) {
    auto loweringConfig =
        getLoweringConfig<IREE::GPU::LoweringConfigAttr>(linalgOp);
    if (!loweringConfig) {
      return;
    }
    if (!getMmaKind(loweringConfig)) {
      return;
    }
    packingCandidates.push_back(linalgOp);
  });

  for (auto candidate : packingCandidates) {
    rewriter.setInsertionPoint(candidate);
    linalg::LinalgOp lianlgOp =
        removeUnitExtentDimsfromMaps(candidate, rewriter);
    rewriter.setInsertionPoint(lianlgOp);
    if (failed(packToIntrinsic(lianlgOp, rewriter))) {
      funcOp.emitError() << "failed to pack operation marked with intrinsic\n";
      return signalPassFailure();
    }
  }

  // Step 2. Convert configured linalg ops to inner_tiled ops with multi-MMA
  // intrinsic kinds.
  {
    RewritePatternSet patterns(context);
    patterns.add<ConvertToMultiMma>(context);
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      funcOp.emitError() << "failed to convert linalg to multi-MMA inner_tiled";
      return signalPassFailure();
    }
  }

  // Step 3. Run layout propagation patterns to pull in adjacent un-configured
  // ops.
  RewritePatternSet patterns(context);
  linalg::ControlPropagationFn control = [](OpOperand *opOperand) -> bool {
    Operation *producer = opOperand->get().getDefiningOp();
    Operation *consumer = opOperand->getOwner();
    return !getLoweringConfig(producer) && !getLoweringConfig(consumer);
  };

  linalg::populateDataLayoutPropagationPatterns(patterns, control);
  patterns.add<ExpandDestinationForOp>(context);
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}

} // namespace mlir::iree_compiler

// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//=== TileAndDistributeToWorkgroupsPass.cpp - Tile to workgroups pass ----===//
//
// This pass distributes the operations within the module to workgroups. This
// pass is created to move tile and distribution out of flow level and into
// the backends. For now this is mostly a bridge pass to connect things during
// the transition, and eventually might just be deprecated in favor of a
// utility method.
//
//===---------------------------------------------------------------------===//

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgExt/Passes/Transforms.h"
#include "iree/compiler/Codegen/Common/CommonPasses.h"
#include "iree/compiler/Codegen/Common/EncodingInfo.h"
#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Codegen/Dialect/LoweringConfig.h"
#include "iree/compiler/Codegen/Interfaces/PartitionableLoopsInterface.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-tile-and-distribute-to-workgroups"

namespace mlir {
namespace iree_compiler {

/// Method to return the configuration to use for first-level tile and
/// distribute. Returns the
/// - Root op of the dispatch. If no root op was found returns `nullptr`.
/// - tileSizes to use
/// - interchange
/// - loops to be partitioned (the tile sizes for the non-partitioned loop are
///   set to 0)
/// - static loop ranges - this is meant to be an optimization hint. It recovers
///   the static values that the workload of the dispatch corresponds to.
// TODO: Remove the use of static loop ranges. This is used to set the number of
// workgroups to a static value. Ideally this should not be done and the static
// and dyamic cases are handled the same way. When the tile+distribute moves
// away from using `scf.for` to using a construct that better captures
// distribution (like `scf.forall`) this information can be dropped.
static LogicalResult getTileAndDistributeConfig(
    ArrayRef<Operation *> computeOps, Operation *&dispatchRootOp,
    SmallVectorImpl<int64_t> &tileSizes,
    SmallVectorImpl<int64_t> &staticLoopRanges,
    SmallVectorImpl<int64_t> &interchange,
    SmallVectorImpl<unsigned> &partitionableLoops) {
  // Find the lowering configuration of the root operation.
  Operation *rootOp = nullptr;
  for (Operation *op : llvm::reverse(computeOps)) {
    if (getLoweringConfig(op)) {
      rootOp = op;
      break;
    }
  }
  if (!rootOp) {
    // THere is no lowering configuration. Return `null`.
    dispatchRootOp = nullptr;
    return success();
  }
  dispatchRootOp = rootOp;

  auto partitionableLoopInterface =
      dyn_cast<PartitionableLoopsInterface>(rootOp);
  if (!partitionableLoopInterface) {
    // Just return. All the in-out vectors are empty that should default
    // the number of workgroups to {1, 1, 1}
    return success();
  }

  partitionableLoops =
      partitionableLoopInterface.getPartitionableLoops(std::nullopt);
  IREE::Codegen::LoweringConfigAttr rootOpConfig = getLoweringConfig(rootOp);
  if (!rootOpConfig) {
    return rootOp->emitOpError(
        "unable to find configuration of root op to define workgroup count "
        "region");
  }
  tileSizes.assign(rootOpConfig.getTileSizeVals(0));
  interchange.assign(rootOpConfig.getTileInterchangeVals(0));

  // Set tile sizes of non-partitioned loops to 0.
  llvm::SmallDenseSet<unsigned> partitionableLoopsSet;
  partitionableLoopsSet.insert(partitionableLoops.begin(),
                               partitionableLoops.end());
  for (auto loopId : llvm::seq<unsigned>(0, tileSizes.size())) {
    if (partitionableLoopsSet.count(loopId)) continue;
    tileSizes[loopId] = 0;
  }

  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(*rootOp)) {
    staticLoopRanges = linalgOp.getStaticLoopRanges();
  }
  staticLoopRanges.resize(tileSizes.size(), ShapedType::kDynamic);

  return success();
}

/// Get the materialization information from a `tensor.pack` operation.
static FailureOr<IREE::LinalgExt::MaterializeEncodingInfo>
getMaterializationInfo(tensor::PackOp packOp) {
  IREE::LinalgExt::MaterializeEncodingInfo encodingInfo;
  SmallVector<OpFoldResult> mixedTileSizes = packOp.getMixedTiles();
  encodingInfo.innerTileSizes.reserve(mixedTileSizes.size());
  for (auto tileSize : mixedTileSizes) {
    if (tileSize.is<Value>()) {
      encodingInfo.innerTileSizes.push_back(ShapedType::kDynamic);
    } else {
      encodingInfo.innerTileSizes.push_back(
          llvm::cast<IntegerAttr>(tileSize.get<Attribute>()).getInt());
    }
  }
  encodingInfo.innerDimsPos = llvm::to_vector(packOp.getInnerDimsPos());
  encodingInfo.outerDimsPerm = llvm::to_vector(packOp.getOuterDimsPerm());
  encodingInfo.srcRank = packOp.getSourceRank();
  return encodingInfo;
}

//===---------------------------------------------------------------------===//
// Patterns to lower operations that are used to compute the number of
// workgroups.
//===---------------------------------------------------------------------===//

/// The `flow.dispatch.workgroup_count_from_dag_root` op is lowered to
/// a sequence of `affine.apply affine_map<()[s0, s1] -> ceildDiv(s0,
/// s1)>(workload, tileSize)`. for each of the dimensions. When tile size is
/// zero, number of workgroups is set to 1.
static LogicalResult lowerDispatchWorkgroupCountForDagRootOp(
    RewriterBase &rewriter,
    IREE::Flow::DispatchWorkgroupCountFromDagRootOp workgroupCountOp,
    ArrayRef<int64_t> givenTileSizes, ArrayRef<int64_t> givenStaticLoopRanges,
    ArrayRef<int64_t> givenInterchange, ArrayRef<unsigned> partitionedLoops,
    int maxWorkgroupParallelDims) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(workgroupCountOp);
  auto workloadValues = workgroupCountOp.getOperands();
  SmallVector<OpFoldResult> tileSizes = llvm::map_to_vector(
      givenTileSizes,
      [&](int64_t v) -> OpFoldResult { return rewriter.getIndexAttr(v); });

  Attribute zero = rewriter.getIndexAttr(0);
  tileSizes.resize(workloadValues.size(), zero);
  SmallVector<int64_t> staticLoopRanges =
      llvm::to_vector(givenStaticLoopRanges);
  staticLoopRanges.resize(workloadValues.size(), ShapedType::kDynamic);
  Location loc = workgroupCountOp.getLoc();
  auto numTiles = llvm::map_to_vector(
      llvm::zip_equal(workloadValues, staticLoopRanges, tileSizes),
      [&](std::tuple<Value, int64_t, OpFoldResult> p) -> OpFoldResult {
        auto tileSize = std::get<2>(p);
        if (isConstantIntValue(tileSize, 0)) {
          return rewriter.getIndexAttr(1);
        }

        int64_t staticLoopRange = std::get<1>(p);
        OpFoldResult workload =
            (staticLoopRange == ShapedType::kDynamic
                 ? OpFoldResult(std::get<0>(p))
                 : OpFoldResult(rewriter.getIndexAttr(staticLoopRange)));
        AffineExpr s0, s1;
        bindSymbols(rewriter.getContext(), s0, s1);
        SmallVector<OpFoldResult> mapOperands = {workload, tileSize};
        return affine::makeComposedFoldedAffineApply(
            rewriter, loc, s0.ceilDiv(s1), mapOperands);
      });
  // If there is interchange, first apply interchange on the number of tiles.
  if (!givenInterchange.empty()) {
    SmallVector<OpFoldResult> interchangedNumTiles = numTiles;
    for (auto [index, loop] : llvm::enumerate(givenInterchange)) {
      interchangedNumTiles[loop] = numTiles[index];
    }
    numTiles = interchangedNumTiles;
  }

  // Prune the numtiles for just the partitioned loops. Iterate in reverse
  // since the number of workgroups is specified from fastest varying to
  // slowest varying.
  SmallVector<Value> numWorkgroups;
  for (auto partitionedLoop : llvm::reverse(partitionedLoops)) {
    if (partitionedLoop >= tileSizes.size()) continue;
    if (isConstantIntValue(tileSizes[partitionedLoop], 0)) continue;
    Value numTileAlongDim = getValueOrCreateConstantIndexOp(
        rewriter, loc, numTiles[partitionedLoop]);
    if (numWorkgroups.size() == maxWorkgroupParallelDims) {
      // IREE runtime only has 3 ID dimensions. After all the num of tiles are
      // combined into one.
      AffineExpr s0 = rewriter.getAffineSymbolExpr(0);
      AffineExpr s1 = rewriter.getAffineSymbolExpr(1);
      numWorkgroups.back() = affine::makeComposedAffineApply(
          rewriter, loc, s0 * s1, {numWorkgroups.back(), numTileAlongDim});
      continue;
    }
    numWorkgroups.push_back(numTileAlongDim);
  }
  Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  numWorkgroups.resize(workgroupCountOp.getNumResults(), one);
  rewriter.replaceOp(workgroupCountOp, numWorkgroups);
  return success();
}

/// Lowers the computation within the workgroup count region for the ops
/// that are handled by default.
static LogicalResult lowerWorkgroupCount(
    RewriterBase &rewriter, func::FuncOp entryPointFn,
    ArrayRef<OpFoldResult> workgroupCount, ArrayRef<int64_t> tileSizes,
    ArrayRef<int64_t> staticLoopRanges, ArrayRef<int64_t> interchange,
    ArrayRef<unsigned> partitionedLoops, int maxWorkgroupParallelDims) {
  FailureOr<IREE::HAL::ExecutableExportOp> exportOp =
      getEntryPoint(entryPointFn);
  if (failed(exportOp)) {
    return entryPointFn.emitOpError(
        "expected function to be entry point function");
  }
  Block *body = exportOp->getWorkgroupCountBody();
  if (!body) {
    return exportOp->emitOpError("unexpected empty workgroup count region");
  }
  SmallVector<Operation *> countOps;
  for (Operation &op : *body) {
    if (isa<IREE::Flow::DispatchWorkgroupCountFromSliceOp,
            IREE::Flow::DispatchWorkgroupCountFromDagRootOp>(&op)) {
      countOps.push_back(&op);
    }
  }
  if (countOps.empty()) {
    // If there are no default handled `flow.dispatch.workgroup_count`
    // operation, do nothing. do nothing.
    return success();
  }
  if (!llvm::hasSingleElement(countOps)) {
    return exportOp->emitOpError(
        "unexpected multiple flow.dispatch.workgroup_count_from_dag_root "
        "operations "
        "in body");
  }

  return TypeSwitch<Operation *, LogicalResult>(countOps[0])
      .Case<IREE::Flow::DispatchWorkgroupCountFromSliceOp>([&](auto countOp) {
        return lowerWorkgroupCountFromSliceOp(rewriter, countOp, entryPointFn,
                                              workgroupCount,
                                              maxWorkgroupParallelDims);
      })
      .Case<IREE::Flow::DispatchWorkgroupCountFromDagRootOp>([&](auto countOp) {
        return lowerDispatchWorkgroupCountForDagRootOp(
            rewriter, countOp, tileSizes, staticLoopRanges, interchange,
            partitionedLoops, maxWorkgroupParallelDims);
      })
      .Default([&](Operation *) { return success(); });
}

//===---------------------------------------------------------------------===//
// Patterns and methods for tile and distribute of Linalg ops to workgroups.
//===---------------------------------------------------------------------===//

namespace {

struct TileAndDistributeToWorkgroupsPass
    : public TileAndDistributeToWorkgroupsBase<
          TileAndDistributeToWorkgroupsPass> {
  TileAndDistributeToWorkgroupsPass(
      int32_t maxWorkgroupParallelDims,
      linalg::DistributionMethod distributionMethod) {
    this->maxWorkgroupParallelDims = maxWorkgroupParallelDims;
    this->distributionMethod = (int32_t)distributionMethod;
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, IREE::Flow::FlowDialect,
                    IREE::HAL::HALDialect, linalg::LinalgDialect,
                    IREE::LinalgExt::IREELinalgExtDialect, scf::SCFDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override;
};
}  // namespace

void TileAndDistributeToWorkgroupsPass::runOnOperation() {
  MLIRContext *context = &getContext();
  IREE::HAL::ExecutableVariantOp variantOp = getOperation();
  ModuleOp innerModule = variantOp.getInnerModule();
  llvm::StringMap<IREE::HAL::ExecutableExportOp> entryPoints =
      getAllEntryPoints(innerModule);

  if (maxWorkgroupParallelDims > kNumMaxParallelDims) {
    innerModule.emitError(
        "maxWorkgroupParallelDims set to more than allowed MaxParallelDims");
  }

  for (func::FuncOp funcOp : innerModule.getOps<func::FuncOp>()) {
    auto exportOp = entryPoints.lookup(funcOp.getName());
    if (!exportOp) continue;

    SmallVector<Operation *> computeOps = getComputeOps(funcOp);
    SmallVector<int64_t> tileSizes, staticLoopRanges, interchange;
    SmallVector<unsigned> partitionableLoops;
    Operation *dispatchRootOp = nullptr;
    if (failed(getTileAndDistributeConfig(computeOps, dispatchRootOp, tileSizes,
                                          staticLoopRanges, interchange,
                                          partitionableLoops))) {
      funcOp.emitOpError("failed to get tile and distribute configuration");
      return signalPassFailure();
    }

    IRRewriter rewriter(context);
    // If there are no compute ops, nothing more to do.
    if (!dispatchRootOp || computeOps.empty()) {
      if (failed(lowerWorkgroupCount(
              rewriter, funcOp,
              /*workgroupCountVals =*/ArrayRef<OpFoldResult>{},
              /*tileSizes =*/ArrayRef<int64_t>{},
              /*staticLoopRanges =*/ArrayRef<int64_t>{},
              /*interchange =*/ArrayRef<int64_t>{},
              /*partitionedLoops =*/ArrayRef<unsigned>{},
              maxWorkgroupParallelDims))) {
        funcOp.emitOpError(
            "failed to lower workgroup count region when no compute ops in the "
            "dispatch");
        return signalPassFailure();
      }
      continue;
    }

    // Configure the linalg options.
    // Tile size selection function.
    auto tileSizeFn = [&](OpBuilder &builder,
                          Operation *op) -> SmallVector<Value, 4> {
      // Check if tile sizes are deduced from the configuration. If so use
      // those.
      return llvm::map_to_vector<4>(tileSizes, [&](int64_t ts) -> Value {
        return builder.create<arith::ConstantIndexOp>(op->getLoc(), ts);
      });
    };

    linalg::DistributionMethod distributionMethodValue =
        (linalg::DistributionMethod)(distributionMethod.getValue());
    auto linalgTilingOptions =
        linalg::LinalgTilingOptions()
            .setDistributionOptions(getIREELinalgLoopDistributionOptions(
                tileSizes, distributionMethodValue, maxWorkgroupParallelDims))
            .setInterchange(llvm::map_to_vector<4>(
                interchange,
                [](int64_t v) -> unsigned { return static_cast<unsigned>(v); }))
            .setLoopType(linalg::LinalgTilingLoopType::Loops)
            .setTileSizeComputationFunction(tileSizeFn);

    FailureOr<IREETileAndFuseResult> tileAndFuseResult =
        tileAndFuseDispatchUsingSCFForOp(
            rewriter, cast<TilingInterface>(computeOps.back()),
            linalgTilingOptions);
    if (failed(tileAndFuseResult)) {
      funcOp.emitOpError("Tile+Distribute failed");
      return signalPassFailure();
    }

    // Materialize the computation for workgroup counts.
    auto workgroupCountOfr =
        getAsOpFoldResult(tileAndFuseResult->workgroupCount);
    if (failed(lowerWorkgroupCount(
            rewriter, funcOp, workgroupCountOfr, tileSizes, staticLoopRanges,
            interchange, partitionableLoops, maxWorkgroupParallelDims))) {
      funcOp.emitOpError("workgroup count lowering failed");
      return signalPassFailure();
    }

    // Resolve the `tensor.dim` operations in workgroup count region.
    {
      RewritePatternSet patterns(exportOp->getContext());
      memref::populateResolveRankedShapeTypeResultDimsPatterns(patterns);
      if (failed(applyPatternsAndFoldGreedily(exportOp, std::move(patterns)))) {
        exportOp.emitOpError("`tensor.dim` resolution in exportOp failed");
        return signalPassFailure();
      }
    }

    {
      RewritePatternSet patterns(context);
      populateTileAndDistributeToWorkgroupsCleanupPatterns(patterns,
                                                           linalgTilingOptions);
      if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
        funcOp.emitOpError("Tile+Distribute clean up patterns failed");
        return signalPassFailure();
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "--- After Tile + Distribute ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });

    {
      SmallVector<int64_t> staticNumWorkgroup = getStaticNumWorkgroups(funcOp);
      // Apply linalg tiling optimization patterns, which includes folding
      // casting ops into tiled operations.
      RewritePatternSet patterns(context);
      linalg::populateLinalgTilingCanonicalizationPatterns(patterns);
      tensor::populateFoldTensorEmptyPatterns(patterns);
      populateFoldAffineMinInDistributedLoopsPatterns(patterns,
                                                      staticNumWorkgroup);
      context->getOrLoadDialect<tensor::TensorDialect>()
          ->getCanonicalizationPatterns(patterns);
      context->getOrLoadDialect<IREE::LinalgExt::IREELinalgExtDialect>()
          ->getCanonicalizationPatterns(patterns);
      if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
        funcOp.emitOpError("tiling canonicalizations failed");
        return signalPassFailure();
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "--- After Canonicalize ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });

    // After rewriting destructive updates, there might be uses of compute
    // operations only in `tensor.dim` ops. Resolve these.
    RewritePatternSet resolveDimOps(context);
    memref::populateResolveRankedShapeTypeResultDimsPatterns(resolveDimOps);
    if (failed(
            applyPatternsAndFoldGreedily(funcOp, std::move(resolveDimOps)))) {
      funcOp.emitOpError("resolving ranked shaped results dims failed");
      return signalPassFailure();
    }
  }
}

std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createTileAndDistributeToWorkgroupsPass(
    int32_t maxWorkgroupParallelDims,
    linalg::DistributionMethod distributionMethod) {
  return std::make_unique<TileAndDistributeToWorkgroupsPass>(
      maxWorkgroupParallelDims, distributionMethod);
}

}  // namespace iree_compiler
}  // namespace mlir

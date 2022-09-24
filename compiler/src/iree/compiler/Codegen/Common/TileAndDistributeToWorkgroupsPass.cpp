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
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgExt/Passes/Transforms.h"
#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Codegen/Dialect/LoweringConfig.h"
#include "iree/compiler/Codegen/Interfaces/PartitionableLoopsInterface.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-tile-and-distribute-to-workgroups"

namespace mlir {
namespace iree_compiler {

/// Find the root operation of the dispatch, the one (and preferably only one)
/// that has a lowering configuration.
static FailureOr<Operation *> getRootOp(ArrayRef<Operation *> computeOps) {
  for (auto op : computeOps) {
    IREE::Codegen::LoweringConfigAttr loweringConfig = getLoweringConfig(op);
    if (!loweringConfig) continue;
    return op;
  }
  return failure();
}

/// Find the number of workload values. It is the number of loops of the last of
/// the compute ops.
/// TODO(ravishankarm): This is an implicit link between the way the dispatch is
/// created and the backend codegen. Fix this by propagating the number of
/// workload entries from Flow to HAL level.
static FailureOr<unsigned> getNumWorkloadValues(
    ArrayRef<Operation *> computeOps) {
  if (computeOps.empty()) return failure();
  TilingInterface tilingRoot = dyn_cast<TilingInterface>(computeOps.back());
  if (!tilingRoot) {
    return tilingRoot->emitOpError(
        "expected the root of tile and fuse operations to implement the "
        "`TilingInterface`");
  }
  return tilingRoot.getLoopIteratorTypes().size();
}

/// Fallback lowering of `flow.dispatch.workgroup_count_from_dag_root` to {1, 1,
/// 1}.
static LogicalResult lowerToUnitWorkgroupCount(
    IREE::Flow::DispatchWorkgroupCountFromDagRootOp workgroupCountOp) {
  OpBuilder builder(workgroupCountOp.getContext());
  builder.setInsertionPoint(workgroupCountOp);
  Value one =
      builder.create<arith::ConstantIndexOp>(workgroupCountOp->getLoc(), 1);
  SmallVector<Value> replacements(workgroupCountOp->getNumResults(), one);
  workgroupCountOp->replaceAllUsesWith(replacements);
  workgroupCountOp.erase();
  return success();
}

/// Method to lower the `flow.dispatch.workgroup_count_from_dag_root` op into
/// the actual computation that returns the number of workgroups.
static LogicalResult lowerDispatchWorkgroupCountFromDagRootOp(
    IREE::Flow::DispatchWorkgroupCountFromDagRootOp workgroupCountOp,
    ArrayRef<Operation *> computeOps, SmallVectorImpl<int64_t> &tileSizes,
    SmallVector<int64_t> &interchange) {
  auto workloadValues = workgroupCountOp.operands();

  // Find the lowering configuration of the root operation.
  FailureOr<Operation *> rootOp = getRootOp(computeOps);
  if (failed(rootOp)) {
    return lowerToUnitWorkgroupCount(workgroupCountOp);
  }

  auto partitionableLoopInterface =
      dyn_cast<PartitionableLoopsInterface>(*rootOp);
  if (!partitionableLoopInterface) {
    return lowerToUnitWorkgroupCount(workgroupCountOp);
  }

  SmallVector<unsigned> partitionableLoops =
      partitionableLoopInterface.getPartitionableLoops(kNumMaxParallelDims);

  IREE::Codegen::LoweringConfigAttr rootOpConfig = getLoweringConfig(*rootOp);
  if (!rootOpConfig) {
    return rootOp.value()->emitOpError(
        "unable to find configuration of root op to define workgroup count "
        "region");
  }
  tileSizes.assign(rootOpConfig.getTileSizeVals(0));
  interchange.assign(rootOpConfig.getTileInterchangeVals(0));

  // Resize tile sizes to the number of loops setting inner loops to 0.
  tileSizes.resize(workloadValues.size(), 0);
  // Check that the interchange vector is also equal to the number of loops
  if (!interchange.empty()) {
    if (interchange.size() < workloadValues.size()) {
      auto seq = llvm::seq<int64_t>(interchange.size(), workloadValues.size());
      interchange.append(seq.begin(), seq.end());
    }
    interchange.resize(workloadValues.size());
  }
  // For now assert that number of partitionable loops are less than the
  // supported max.
  // TODO(ravishankarm): Relax this restriction.
  if (partitionableLoops.size() > kNumMaxParallelDims) {
    return workgroupCountOp.emitOpError(
               "expected number of partitionable loops to be less than or "
               "equal to ")
           << kNumMaxParallelDims;
  }

  MLIRContext *context = workgroupCountOp.getContext();
  OpBuilder builder(context);
  // Add as many arguments as the number of loops
  Location loc = workgroupCountOp.getLoc();

  // AffineMap for the number of workgroups = ceilDiv(workload, tileSize)
  SmallVector<Value> numTiles;
  numTiles.reserve(workloadValues.size());
  builder.setInsertionPoint(workgroupCountOp);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  llvm::DenseSet<unsigned> partitionableLoopsSet;
  partitionableLoopsSet.insert(partitionableLoops.begin(),
                               partitionableLoops.end());

  for (auto workload : llvm::enumerate(workloadValues)) {
    if (!partitionableLoopsSet.count(workload.index())) {
      tileSizes[workload.index()] = 0;
    }
    int64_t tileSize = tileSizes[workload.index()];

    if (tileSize == 0) {
      numTiles.push_back(one);
      continue;
    }

    // When the loop range is known to be static, let's directly use it.
    int64_t loopRange = ShapedType::kDynamicSize;

    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(*rootOp)) {
      loopRange = linalgOp.getStaticLoopRanges()[workload.index()];
    }

    if (loopRange != ShapedType::kDynamicSize) {
      if (tileSize == 1) {
        Value workload = builder.create<arith::ConstantIndexOp>(loc, loopRange);
        numTiles.push_back(workload);
        continue;
      }
      int64_t nTileI64 = (loopRange + tileSize - 1) / tileSize;
      Value nTiles = builder.create<arith::ConstantIndexOp>(loc, nTileI64);
      numTiles.push_back(nTiles);
    } else {
      if (tileSize == 1) {
        numTiles.push_back(workload.value());
        continue;
      }
      AffineExpr s0;
      bindSymbols(workgroupCountOp.getContext(), s0);
      AffineMap numTilesMap = AffineMap::get(0, 1, s0.ceilDiv(tileSize));
      Value nTiles =
          builder.create<AffineApplyOp>(loc, numTilesMap, workload.value());
      numTiles.push_back(nTiles);
    }
  }

  // If there is interchange, first apply interchange on the number of tiles.
  if (!interchange.empty()) {
    SmallVector<Value> interchangedNumTiles = numTiles;
    for (auto interchangedLoop : llvm::enumerate(interchange)) {
      interchangedNumTiles[interchangedLoop.value()] =
          numTiles[interchangedLoop.index()];
    }
    numTiles = interchangedNumTiles;
  }

  // Prune the numtiles for just the partitioned loops. Iterate in reverse since
  // the number of workgroups is specified from fastest varying to slowest
  // varying.
  SmallVector<Value> numWorkgroups;
  for (auto partitionedLoop : llvm::reverse(partitionableLoops)) {
    // If the loop isnt tiled, skip it.
    if (tileSizes[partitionedLoop] == 0) continue;
    numWorkgroups.push_back(numTiles[partitionedLoop]);
  }
  numWorkgroups.resize(kNumMaxParallelDims, one);
  workgroupCountOp->replaceAllUsesWith(numWorkgroups);
  workgroupCountOp.erase();
  return success();
}

//===---------------------------------------------------------------------===//
// Patterns and methods for tile and distribute of Linalg ops to workgroups.
//===---------------------------------------------------------------------===//

namespace {
struct TileAndDistributeToWorkgroupsPass
    : public TileAndDistributeToWorkgroupsBase<
          TileAndDistributeToWorkgroupsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, IREE::Flow::FlowDialect,
                    IREE::HAL::HALDialect, linalg::LinalgDialect,
                    scf::SCFDialect, tensor::TensorDialect>();
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

  for (func::FuncOp funcOp : innerModule.getOps<func::FuncOp>()) {
    auto exportOp = entryPoints.lookup(funcOp.getName());
    if (!exportOp) continue;

    SmallVector<Operation *> computeOps;
    SmallVector<LoopTilingAndDistributionInfo> tiledLoops;
    if (failed(getComputeOps(funcOp, computeOps, tiledLoops))) {
      funcOp.emitOpError("failed to get compute ops in dispatch");
      return signalPassFailure();
    }
    if (!tiledLoops.empty()) {
      // The entry point already has distribution to workgroups. Do nothing.
      continue;
    }

    // Find the `flow.dispatch.workgroup_count_from_dag_root` operation in the
    // `workgroup_count` region of `hal.executable.export`. Lower this to the
    // actual computation that returns the `workgroup_count`.
    // TODO(ravishankarm): Ideally this should be done using a pattern, but the
    // `workload_per_workgroup` usage here makes it hard. That is to be
    // deprecated. Rework this logic into a pattern when that is done.
    Region &workgroupCountRegion = exportOp.getWorkgroupCount();
    if (!workgroupCountRegion.hasOneBlock()) {
      exportOp.emitOpError(
          "expected workgroup_count region to have a single block");
      return signalPassFailure();
    }
    Block &workgroupCountBody = workgroupCountRegion.front();
    auto ops = workgroupCountBody
                   .getOps<IREE::Flow::DispatchWorkgroupCountFromDagRootOp>();
    if (!llvm::hasSingleElement(ops)) {
      // Do not modify the region since the default path expects only a single
      // `flow.dispatch.workgroup_count_from_dag_root` op.
      continue;
    }
    IREE::Flow::DispatchWorkgroupCountFromDagRootOp defaultWorkgroupCountOp =
        *(ops.begin());

    SmallVector<int64_t> tileSizes, interchange;
    if (failed(lowerDispatchWorkgroupCountFromDagRootOp(
            defaultWorkgroupCountOp, computeOps, tileSizes, interchange))) {
      defaultWorkgroupCountOp.emitOpError(
          "failed to lower default number of workgroups");
      return signalPassFailure();
    }

    // If there are no compute ops, nothing more to do.
    if (computeOps.empty()) continue;

    // Add a marker to the last operation in the list.
    auto marker = StringAttr::get(context, "__workgroup_tiling__");
    computeOps.back()->setAttr(linalg::LinalgTransforms::kLinalgTransformMarker,
                               marker);

    // Configure the linalg options.
    // Tile size selection function.
    auto tileSizeFn = [&](OpBuilder &builder,
                          Operation *op) -> SmallVector<Value, 4> {
      // Check if tile sizes are deduced from the configuration. If so use
      // those.
      return llvm::to_vector<4>(
          llvm::map_range(tileSizes, [&](int64_t ts) -> Value {
            return builder.create<arith::ConstantIndexOp>(op->getLoc(), ts);
          }));
    };

    auto linalgTilingOptions =
        linalg::LinalgTilingOptions()
            .setDistributionOptions(getIREELinalgLoopDistributionOptions())
            .setInterchange(llvm::to_vector<4>(
                llvm::map_range(interchange,
                                [](int64_t v) -> unsigned {
                                  return static_cast<unsigned>(v);
                                })))
            .setLoopType(linalg::LinalgTilingLoopType::Loops)
            .setTileSizeComputationFunction(tileSizeFn);

    {
      RewritePatternSet patterns(context);
      populateTileAndDistributeToWorkgroupsPatterns(
          patterns, linalgTilingOptions,
          linalg::LinalgTransformationFilter(marker));
      if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
        funcOp.emitOpError("Tile+Distribute failed");
        return signalPassFailure();
      }
    }

    // If tiling didn't happen because there are no tile sizes we are
    // potentially left with a marker that will confuse the following passes so
    // we remove the intermediate markers.
    funcOp->walk([&](Operation *op) {
      op->removeAttr(linalg::LinalgTransforms::kLinalgTransformMarker);
    });

    LLVM_DEBUG({
      llvm::dbgs() << "--- After Tile + Distribute ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });

    {
      // Apply linalg tiling optimization patterns.
      RewritePatternSet canonicalizationPatterns(context);
      linalg::populateLinalgTilingCanonicalizationPatterns(
          canonicalizationPatterns);
      populateFoldAffineMinInDistributedLoopsPatterns(canonicalizationPatterns);
      if (failed(applyPatternsAndFoldGreedily(
              funcOp, std::move(canonicalizationPatterns)))) {
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
createTileAndDistributeToWorkgroupsPass() {
  return std::make_unique<TileAndDistributeToWorkgroupsPass>();
}

}  // namespace iree_compiler
}  // namespace mlir

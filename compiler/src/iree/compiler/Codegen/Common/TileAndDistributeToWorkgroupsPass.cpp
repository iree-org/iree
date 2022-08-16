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
#include "iree/compiler/Codegen/Common/DestructiveUpdateUtils.h"
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
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
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
  PartitionableLoopsInterface tilingRoot =
      dyn_cast<PartitionableLoopsInterface>(computeOps.back());
  if (!tilingRoot) {
    return tilingRoot->emitOpError(
        "expected the root of tile and fuse operations to implement the "
        "`PartitionableLoopsInterface`");
  }
  return tilingRoot.getNumLoops();
}

/// Fallback lowering of `flow.dispatch.default_workgroup_count` to {1, 1, 1}.
static LogicalResult lowerToUnitWorkgroupCount(
    IREE::Flow::DispatchDefaultWorkgroupCountOp workgroupCountOp) {
  OpBuilder builder(workgroupCountOp.getContext());
  builder.setInsertionPoint(workgroupCountOp);
  Value one =
      builder.create<arith::ConstantIndexOp>(workgroupCountOp->getLoc(), 1);
  SmallVector<Value> replacements(workgroupCountOp->getNumResults(), one);
  workgroupCountOp->replaceAllUsesWith(replacements);
  workgroupCountOp.erase();
  return success();
}

/// Method to lower the `flow.dispatch.default_workgroup_count` op into the
/// actual computation that returns the number of workgroups.
static LogicalResult lowerDispatchDefaultWorkgroupCountOp(
    IREE::Flow::DispatchDefaultWorkgroupCountOp workgroupCountOp,
    ArrayRef<Operation *> computeOps, SmallVectorImpl<int64_t> &tileSizes,
    SmallVector<int64_t> &interchange,
    SmallVectorImpl<int64_t> &workloadPerWorkgroup) {
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
    return rootOp.getValue()->emitOpError(
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
    if (tileSizes[workload.index()] == 0) {
      numTiles.push_back(one);
      continue;
    }
    if (tileSizes[workload.index()] == 1) {
      numTiles.push_back(workload.value());
      continue;
    }
    AffineExpr s0;
    bindSymbols(workgroupCountOp.getContext(), s0);
    AffineMap numTilesMap =
        AffineMap::get(0, 1, s0.ceilDiv(tileSizes[workload.index()]));
    Value nTiles =
        builder.create<AffineApplyOp>(loc, numTilesMap, workload.value());
    numTiles.push_back(nTiles);
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
    workloadPerWorkgroup.push_back(tileSizes[partitionedLoop]);
  }
  numWorkgroups.resize(kNumMaxParallelDims, one);
  workgroupCountOp->replaceAllUsesWith(numWorkgroups);
  workgroupCountOp.erase();
  return success();
}

/// Update the workload_per_wg value on the TranslationInfoAttr.
// TODO(ravishankarm): The workload_per_wg field should be deprecated. This
// is just transition before all dependencies on it can be removed.
static LogicalResult updateTranslationInfoAttr(
    IREE::HAL::ExecutableExportOp exportOp,
    ArrayRef<int64_t> workloadPerWorkgroup) {
  IREE::Codegen::DispatchLoweringPassPipeline passPipeline =
      IREE::Codegen::DispatchLoweringPassPipeline::CPUDefault;
  if (auto translationInfo = getTranslationInfo(exportOp)) {
    // Expect the `workload_per_wg` to be empty.
    if (!translationInfo.getWorkloadPerWorkgroupVals().empty()) {
      return exportOp.emitOpError(
          "expected workload_per_wg to be empty at this stage");
    }
    passPipeline = translationInfo.getDispatchLoweringPassPipeline();
  }
  auto newTranslationInfoAttr = IREE::Codegen::TranslationInfoAttr::get(
      exportOp.getContext(), passPipeline, workloadPerWorkgroup);
  setTranslationInfo(exportOp, newTranslationInfoAttr);
  return success();
}

//===---------------------------------------------------------------------===//
// Patterns and methods for tile and distribute of Linalg ops to workgroups.
//===---------------------------------------------------------------------===//

// Pull in producers into the tiled operation.
static void pullInProducers(linalg::LinalgOp tiledOp,
                            ValueRange untiledOperands,
                            PatternRewriter &rewriter) {
  for (auto en : llvm::enumerate(untiledOperands)) {
    auto producer = en.value().getDefiningOp<linalg::LinalgOp>();
    if (!producer) continue;

    OpResult opResult = en.value().cast<OpResult>();
    auto maybeFusionInfo = linalg::fuseProducerOfTensor(
        rewriter, producer->getResult(opResult.getResultNumber()),
        tiledOp->getOpOperand(en.index()));
    if (failed(maybeFusionInfo)) continue;

    // If the fusion was successfull recurse over the current producers operands
    // and fuse them in as well.
    SmallVector<Value> origProducerOperands =
        producer.getInputAndOutputOperands();
    pullInProducers(maybeFusionInfo->fusedProducer, origProducerOperands,
                    rewriter);
  }
}

namespace {
// Rewrite pattern to ensure only ops with tensor semantics are tiled.
struct TileAndDistributeLinalgOpsPattern : public linalg::LinalgTilingPattern {
  using Base = linalg::LinalgTilingPattern;
  TileAndDistributeLinalgOpsPattern(MLIRContext *context,
                                    linalg::LinalgTilingOptions options,
                                    linalg::LinalgTransformationFilter marker,
                                    PatternBenefit benefit = 1)
      : Base(context, options, marker, benefit) {}

  LogicalResult matchAndRewrite(linalg::LinalgOp linalgOp,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> untiledOperands = linalgOp.getInputAndOutputOperands();
    FailureOr<linalg::TiledLinalgOp> tiledLinalgOpOr =
        Base::returningMatchAndRewrite(linalgOp, rewriter);
    if (failed(tiledLinalgOpOr)) {
      return failure();
    }
    if (tiledLinalgOpOr->loops.empty()) {
      // If there are no loops, there is nothing to do.
      return success();
    }
    pullInProducers(tiledLinalgOpOr->op, untiledOperands, rewriter);
    return success();
  }
};
}  // namespace

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
      return signalPassFailure();
    }
    if (!tiledLoops.empty()) {
      // The entry point already has distribution to workgroups. Do nothing.
      continue;
    }

    // Find the `flow.dispatch.default_workgroup_count` operation in the
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
                   .getOps<IREE::Flow::DispatchDefaultWorkgroupCountOp>();
    if (!llvm::hasSingleElement(ops)) {
      // Do not modify the region since the default path expects only a single
      // `flow.dispatch.default_workgroup_count` op.
      continue;
    }
    IREE::Flow::DispatchDefaultWorkgroupCountOp defaultWorkgroupCountOp =
        *(ops.begin());

    SmallVector<int64_t> tileSizes, interchange, workloadPerWorkgroup;
    if (failed(lowerDispatchDefaultWorkgroupCountOp(
            defaultWorkgroupCountOp, computeOps, tileSizes, interchange,
            workloadPerWorkgroup))) {
      return signalPassFailure();
    }
    if (failed(updateTranslationInfoAttr(exportOp, workloadPerWorkgroup))) {
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
      patterns.insert<TileAndDistributeLinalgOpsPattern,
                      IREE::LinalgExt::TilingInterfaceTilingPattern>(
          context, linalgTilingOptions,
          linalg::LinalgTransformationFilter(marker));
      if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "--- After Tile + Distribute ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });

    // Apply linalg tiling optimization patterns.
    RewritePatternSet canonicalizationPatterns(context);
    linalg::populateLinalgTilingCanonicalizationPatterns(
        canonicalizationPatterns);
    if (failed(applyPatternsAndFoldGreedily(
            funcOp, std::move(canonicalizationPatterns)))) {
      return signalPassFailure();
    }

    LLVM_DEBUG({
      llvm::dbgs() << "--- After Canonicalize ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });

    // Rewrite destructive updates and ensure no remaining store remains to the
    // full output.

    // TODO(#...): Use of the destructive update rewrite is a hack! There needs
    // to be a way to generate loops as we need, and use the tiled op generation
    // implementation. This should be possible after moving everything to use
    // the `TilingInterface`.
    if (failed(rewriteLinalgDestructiveUpdates(funcOp))) {
      funcOp->emitError("Failed to rewrite destructive updates in:\n")
          << *funcOp.getOperation();
      return signalPassFailure();
    }

    LLVM_DEBUG({
      llvm::dbgs() << "--- After Rewriting destructive updates ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });

    // After rewriting destructive updates, there might be uses of compute
    // operations only in `tensor.dim` ops. Resolve these.
    RewritePatternSet resolveDimOps(context);
    memref::populateResolveRankedShapeTypeResultDimsPatterns(resolveDimOps);
    if (failed(
            applyPatternsAndFoldGreedily(funcOp, std::move(resolveDimOps)))) {
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

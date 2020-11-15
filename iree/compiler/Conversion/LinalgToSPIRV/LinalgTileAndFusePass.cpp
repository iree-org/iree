// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//===- LinalgTileAndFusePass.cpp - Tile and fuse Linalg on Buffers --------===//
//
// Implements a pass to tile and fuse linalg operations on buffers.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Conversion/CodegenUtils/FunctionUtils.h"
#include "iree/compiler/Conversion/CodegenUtils/GetNumWorkgroups.h"
#include "iree/compiler/Conversion/CodegenUtils/MarkerUtils.h"
#include "iree/compiler/Conversion/CodegenUtils/MatmulCodegenStrategy.h"
#include "iree/compiler/Conversion/Common/Attributes.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/CodeGenOptionUtils.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/KernelDispatchUtils.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/MemorySpace.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/Passes.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/Utils.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeDialect.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/Linalg/Analysis/DependenceAnalysis.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Identifier.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopUtils.h"

#define DEBUG_TYPE "iree-linalg-tile-and-fuse"

namespace mlir {
namespace iree_compiler {

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

/// Returns a Linalg marker that replaces existing markers.
linalg::LinalgMarker getLinalgReplaceMarker(StringRef maker,
                                            MLIRContext *context) {
  return linalg::LinalgMarker(ArrayRef<Identifier>(),
                              Identifier::get(maker, context));
}

/// Returns a Linalg marker that matches any of the `matchMarkers` and replaces
/// it with `replaceMarker`.
linalg::LinalgMarker getLinalgMatchAndReplaceMarker(
    ArrayRef<StringRef> matchMarkers, StringRef replaceMarker,
    MLIRContext *context) {
  SmallVector<Identifier, 2> markers;
  markers.reserve(matchMarkers.size());
  for (StringRef marker : matchMarkers) {
    markers.emplace_back(Identifier::get(marker, context));
  }
  return linalg::LinalgMarker(markers, Identifier::get(replaceMarker, context));
}

/// Apply canonicalizations related to tiling to make promotion/vectorization
/// easier.
static void applyCanonicalizationPatterns(MLIRContext *context, Operation *op) {
  OwningRewritePatternList canonicalizationPatterns;
  canonicalizationPatterns.insert<AffineMinCanonicalizationPattern>(context);
  AffineApplyOp::getCanonicalizationPatterns(canonicalizationPatterns, context);
  AffineMinOp::getCanonicalizationPatterns(canonicalizationPatterns, context);
  SubViewOp::getCanonicalizationPatterns(canonicalizationPatterns, context);
  applyPatternsAndFoldGreedily(op, std::move(canonicalizationPatterns));
}

//===----------------------------------------------------------------------===//
// Main pass
//===----------------------------------------------------------------------===//

namespace {
/// Function pass that implements tiling and fusion in Linalg on buffers.
class LinalgTileAndFusePass
    : public PassWrapper<LinalgTileAndFusePass, OperationPass<ModuleOp>> {
 public:
  LinalgTileAndFusePass(const SPIRVCodegenOptions &passOptions)
      : options(passOptions) {}
  LinalgTileAndFusePass(const LinalgTileAndFusePass &pass)
      : options(pass.options) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, gpu::GPUDialect, linalg::LinalgDialect,
                    scf::SCFDialect, ShapeDialect, vector::VectorDialect>();
  }

  void runOnOperation() override;

 private:
  SPIRVCodegenOptions options;
};
}  // namespace

//===----------------------------------------------------------------------===//
// Patterns to tile computation to map to workgroups
//===----------------------------------------------------------------------===//

/// Returns the distribution options for operations when targeting workgroups.
static linalg::LinalgLoopDistributionOptions getWorkgroupDistributionOptions() {
  linalg::LinalgLoopDistributionOptions options;

  options.procInfo = [](OpBuilder &builder, Location loc,
                        ArrayRef<Range> parallelLoopRanges) {
    return getGPUProcessorIdsAndCounts<gpu::BlockIdOp, gpu::GridDimOp>(
        builder, loc, parallelLoopRanges.size());
  };
  options.distributionMethod = {
      linalg::DistributionMethod::CyclicNumProcsEqNumIters,
      linalg::DistributionMethod::CyclicNumProcsEqNumIters,
      linalg::DistributionMethod::CyclicNumProcsEqNumIters};

  return options;
}

/// Once fused the fused views that are due to a RAW dependence can be promoted
/// to workgroup memory. This will make the intermediate storage dead.
static LogicalResult promoteFusedViews(OpBuilder &builder,
                                       ArrayRef<linalg::LinalgOp> fusedOps) {
  linalg::Aliases aliases;
  linalg::LinalgDependenceGraph dependenceGraph(aliases, fusedOps);
  auto fusableDependences =
      linalg::findAllFusableDependences(fusedOps, dependenceGraph);

  DenseSet<Value> promotedViews;
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(*fusedOps.begin());
  for (linalg::LinalgOp op : llvm::reverse(fusedOps)) {
    auto dependences = fusableDependences.lookup(op);
    if (dependences.empty()) continue;
    if (!llvm::hasSingleElement(dependences)) {
      return op.emitError(
          "unable to promote ops with multiple fusable dependences");
    }
    auto dependence = dependences.front();
    unsigned producerIdx = dependence.dependentOpView.operandIndex;
    linalg::LinalgOp consumer =
        cast<linalg::LinalgOp>(dependence.indexingOpView.op);
    unsigned consumerIdx = dependence.indexingOpView.operandIndex;
    Value consumerView = consumer.getShapedOperand(consumerIdx);
    Value promotedView = nullptr;

    if (promotedViews.count(consumerView)) {
      promotedView = consumerView;
    } else if (dependence.dependenceType ==
               linalg::LinalgDependenceGraph::RAW) {
      Optional<linalg::PromotionInfo> promotionInfo =
          linalg::promoteSubviewAsNewBuffer(
              builder, op.getLoc(),
              op.getShapedOperand(producerIdx).getDefiningOp<SubViewOp>(),
              allocateWorkgroupMemory);
      if (!promotionInfo) {
        return op.emitError("unable to promote RAW dependence");
      }
      promotedView = promotionInfo->partialLocalView;
      consumer.getOperation()->setOperand(consumerIdx, promotedView);
      promotedViews.insert(promotedView);
    }
    if (!promotedView) continue;
    op.getOperation()->setOperand(producerIdx, promotedView);
  }
  return success();
}

/// Tile+Fuse only tiles the loops that can be fused. Tile any of the unfused
/// loops in the operation based on the configuration.
static linalg::LinalgOp tileUnfusedLoops(
    OpBuilder &builder, linalg::LinalgOp linalgOp,
    const std::set<unsigned> &fusedLoopDims, const LaunchConfig &launchConfig) {
  SmallVector<int64_t, 4> tileSizes =
      llvm::to_vector<4>(launchConfig.getTileSizes(linalgOp, 0));
  tileSizes.resize(linalgOp.getNumLoops(), 0);
  for (unsigned loopNum : fusedLoopDims) {
    tileSizes[loopNum] = 0;
  }
  if (llvm::all_of(tileSizes, [](int64_t v) { return !v; })) return linalgOp;
  Optional<linalg::TiledLinalgOp> tiledOp = tileLinalgOp(
      builder, linalgOp,
      linalg::LinalgTilingOptions().setTileSizes(tileSizes).setLoopType(
          linalg::LinalgTilingLoopType::ParallelLoops));
  if (!tiledOp) return nullptr;
  linalgOp.erase();
  return tiledOp->op;
}

/// Main utility function that implements the tile and fuse.
static Optional<linalg::TiledAndFusedLinalgOps> tileAndFuseLinalgOps(
    OpBuilder &builder, FuncOp funcOp, ArrayRef<linalg::LinalgOp> fusableOps,
    const linalg::LinalgDependenceGraph &dependenceGraph,
    const LaunchConfig &launchConfig) {
  // Get the tile sizes to use from the last fusable op and the tile+fuse all
  // ops.
  SmallVector<int64_t, 4> tileSizes =
      llvm::to_vector<4>(launchConfig.getTileSizes(fusableOps.back(), 0));
  linalg::LinalgTilingOptions tilingOptions;
  tilingOptions.setDistributionOptions(getWorkgroupDistributionOptions())
      .setTileSizes(tileSizes)
      .setLoopType(linalg::LinalgTilingLoopType::ParallelLoops);

  Optional<linalg::TiledAndFusedLinalgOps> tiledAndFusedOps = llvm::None;
  if (fusableOps.size() == 1) {
    linalg::LinalgOp linalgOp = fusableOps.front();
    Optional<linalg::TiledLinalgOp> tiledOp =
        tileLinalgOp(builder, linalgOp, tilingOptions);
    if (!tiledOp) {
      linalgOp.emitError("unable to tile operation");
      return llvm::None;
    }
    tiledAndFusedOps = linalg::TiledAndFusedLinalgOps{tiledOp->op, {}, {}, {}};
    auto seq = llvm::seq<unsigned>(0, tileSizes.size());
    tiledAndFusedOps->fusedLoopDims.insert(seq.begin(), seq.end());
    tiledAndFusedOps->fusedLoops.assign(tiledOp->loops.begin(),
                                        tiledOp->loops.end());
  } else {
    tiledAndFusedOps = tileAndFuseLinalgOps(builder, fusableOps,
                                            dependenceGraph, tilingOptions);
  }
  if (!tiledAndFusedOps) {
    funcOp.emitError("tile and fuse of linalg operations failed");
    return llvm::None;
  }

  // Update the launch configuration.
  SmallVector<unsigned, 2> distributedLoops =
      llvm::to_vector<2>(tiledAndFusedOps->fusedLoopDims);
  if (failed(updateWorkGroupSize(funcOp, launchConfig.getWorkgroupSize())) ||
      (funcOp.getAttr(getNumWorkgroupsFnAttrName()) &&
       failed(createNumWorkgroupsFromResultShape(
           builder, fusableOps.back(), funcOp, getNumWorkgroupsFnAttrName(),
           tileSizes, distributedLoops)))) {
    funcOp.emitError("failed to update launch configuration");
    return llvm::None;
  }

  // Delete all the original operations.
  for (auto linalgOp : fusableOps) linalgOp.erase();

  // Add workgroup markers to all the tiled and fused operations.
  for (auto fusedProducer : tiledAndFusedOps->fusedProducers) {
    setMarker(fusedProducer, getWorkgroupMarker());
  }
  setMarker(tiledAndFusedOps->op, getWorkgroupMarker());

  return tiledAndFusedOps;
}

static LogicalResult tileAndFuseLinalgBufferOps(
    FuncOp funcOp, ArrayRef<linalg::LinalgOp> linalgOps,
    const linalg::LinalgDependenceGraph &dependenceGraph,
    const LaunchConfig &launchConfig) {
  // Collect all operations that are to be tiled-and-fused.
  MLIRContext *context = funcOp.getContext();
  SmallVector<linalg::LinalgOp, 4> fusableOps;
  for (Operation *operation : linalgOps) {
    if (!launchConfig.hasTileSizes(operation)) continue;
    fusableOps.push_back(cast<linalg::LinalgOp>(operation));
  }
  if (fusableOps.empty()) return success();

  OpBuilder builder(context);
  Optional<linalg::TiledAndFusedLinalgOps> tiledAndFusedOps =
      tileAndFuseLinalgOps(builder, funcOp, fusableOps, dependenceGraph,
                           launchConfig);
  if (!tiledAndFusedOps) {
    return funcOp.emitError("failed to tile and fuse operations");
  }

  LLVM_DEBUG({
    llvm::dbgs() << "--- After Fusion on buffers ---\n";
    funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });

  applyCanonicalizationPatterns(context, funcOp);

  LLVM_DEBUG({
    llvm::dbgs() << "--- After Canonicalization ---\n";
    funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });

  SmallVector<linalg::LinalgOp, 4> promoteFusedViewOps(
      tiledAndFusedOps->fusedProducers.begin(),
      tiledAndFusedOps->fusedProducers.end());
  promoteFusedViewOps.push_back(tiledAndFusedOps->op);

  if (failed(promoteFusedViews(builder, promoteFusedViewOps))) {
    return failure();
  }

  LLVM_DEBUG({
    llvm::dbgs() << "--- After Promotion ---\n";
    funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });

  // Tile the unfused loops. Set the tile sizes for the fused loops to be zero
  // to avoid tiling them again.
  for (linalg::LinalgOp &fusedOp : tiledAndFusedOps->fusedProducers) {
    linalg::LinalgOp tiledOp = tileUnfusedLoops(
        builder, fusedOp, tiledAndFusedOps->fusedLoopDims, launchConfig);
    if (!tiledOp) {
      return fusedOp.emitError("unable to tile unfused loops");
    }
  }
  linalg::LinalgOp tiledOp =
      tileUnfusedLoops(builder, tiledAndFusedOps->op,
                       tiledAndFusedOps->fusedLoopDims, launchConfig);
  if (!tiledOp) {
    return tiledAndFusedOps->op.emitError("unable to tile unfused loops");
  }

  applyCanonicalizationPatterns(context, funcOp);
  return success();
}

//===----------------------------------------------------------------------===//
// Patterns to promote subviews to workgroup memory
//===----------------------------------------------------------------------===//

namespace {
/// Pattern to promote matmul operands to workgroup memory.
struct PromoteMatmulSubviewsPattern
    : public linalg::LinalgPromotionPattern<linalg::MatmulOp> {
  PromoteMatmulSubviewsPattern(MLIRContext *context,
                               linalg::LinalgPromotionOptions options,
                               linalg::LinalgMarker marker,
                               PatternBenefit benefit = 1)
      : linalg::LinalgPromotionPattern<linalg::MatmulOp>(
            context,
            options.setOperandsToPromote({0, 1}).setUseFullTileBuffers(
                {false, false}),
            marker, benefit) {}
};

/// Patterns to promote convolution operands to workgroup memory.
// TODO(ravishankarm): This pattern is only promoting the image subview to
// workgroup memory. In reality we should also be able to promote the filter
// subview to workgroup memory as well. Since none of the loops used to access
// the filter are tiled, this would mean the entire filter is moved to workgroup
// memory. Two reasons this is not done right now:
// 1) Linalg when tiling doesnt create a subview for the filter (since none of
//    its dimensions are tiled. This needs to be relaxed (maybe by using an
//    option).
// 2) Maybe there are better alternatives for handling filter like using
//    different storage classes, since for inference workloads these are model
//    constants. This is TBD.
struct PromoteConvSubviewsPattern
    : public linalg::LinalgPromotionPattern<linalg::ConvOp> {
  PromoteConvSubviewsPattern(MLIRContext *context,
                             linalg::LinalgPromotionOptions options,
                             linalg::LinalgMarker marker,
                             PatternBenefit benefit = 1)
      : linalg::LinalgPromotionPattern<linalg::ConvOp>(
            context,
            options.setOperandsToPromote({1}).setUseFullTileBuffers(
                {false, false}),
            marker, benefit) {}
};
}  // namespace

static void populatePromotionPatterns(MLIRContext *context,
                                      OwningRewritePatternList &patterns) {
  patterns.insert<PromoteMatmulSubviewsPattern, PromoteConvSubviewsPattern>(
      context,
      linalg::LinalgPromotionOptions()
          .setAllocationDeallocationFns(allocateWorkgroupMemory,
                                        deallocateWorkgroupMemory)
          .setCopyInOutFns(copyToWorkgroupMemory, copyToWorkgroupMemory),
      getLinalgMatchAndReplaceMarker(getWorkgroupMarker(),
                                     getWorkgroupMemoryMarker(), context));
}

//===----------------------------------------------------------------------===//
// Patterns to tile computation to map to subgroups
//===----------------------------------------------------------------------===//

/// Computes the Value for subgroupID along each dimension given number of
/// subgroups `numSubGroups` along each dimension (x-first, y-second, z-third).
static SmallVector<linalg::ProcInfo, 2> getSubgroupIdsAndCounts(
    OpBuilder &builder, Location loc, ArrayRef<int64_t> numSubgroups) {
  Type indexType = builder.getIndexType();
  Value subgroupId = builder.create<gpu::SubgroupIdOp>(loc, indexType);
  SmallVector<linalg::ProcInfo, 2> procInfo(numSubgroups.size());

  // subgroupID
  //   = id.z * nsubgroups.y * nsubgroups.x + id.y * nsubgroups.x + id.x
  using edsc::op::operator%;
  for (size_t i = 0, e = numSubgroups.size(); i != e; ++i) {
    Value nprocs = builder.create<ConstantIndexOp>(loc, numSubgroups[i]);
    Value procId = subgroupId % nprocs;
    procInfo[e - i - 1] = linalg::ProcInfo{procId, nprocs};
    subgroupId = builder.create<SignedDivIOp>(loc, subgroupId, nprocs);
  }
  return procInfo;
}

namespace {
/// Pattern to tile linalg.matmul for subgroups.
struct TileMatmulSubgroupPattern
    : public linalg::LinalgTilingPattern<linalg::MatmulOp> {
  using Base = linalg::LinalgTilingPattern<linalg::MatmulOp>;
  TileMatmulSubgroupPattern(MLIRContext *context,
                            linalg::LinalgTilingOptions options,
                            linalg::LinalgMarker marker,
                            PatternBenefit benefit = 1)
      : Base(context, options, marker, benefit) {}
};
}  // namespace

/// Patterns for second level tiling to target subgroups.
static void populateTilingToSubgroupPatterns(
    MLIRContext *context, const LaunchConfig &launchConfig,
    OwningRewritePatternList &patterns) {
  auto getInnerTileSizeFn = [&launchConfig](
                                OpBuilder &builder,
                                Operation *operation) -> SmallVector<Value, 4> {
    ArrayRef<int64_t> tileSizes = launchConfig.getTileSizes(operation, 1);
    if (tileSizes.empty()) return {};
    SmallVector<Value, 4> tileSizesVal;
    tileSizesVal.reserve(tileSizes.size());
    for (auto val : tileSizes) {
      tileSizesVal.push_back(
          builder.create<ConstantIndexOp>(operation->getLoc(), val));
    }
    return tileSizesVal;
  };

  auto getSubgroupProcInfoFn = [&launchConfig](
                                   OpBuilder &builder, Location loc,
                                   ArrayRef<Range> parallelLoopRanges) {
    ArrayRef<int64_t> numSubgroups =
        launchConfig.getNumSubgroups().take_front(parallelLoopRanges.size());
    return getSubgroupIdsAndCounts(builder, loc, numSubgroups);
  };

  linalg::LinalgLoopDistributionOptions subgroupDistributionOptions = {
      getSubgroupProcInfoFn,
      {linalg::DistributionMethod::CyclicNumProcsEqNumIters,
       linalg::DistributionMethod::CyclicNumProcsEqNumIters}};

  patterns.insert<TileMatmulSubgroupPattern>(
      context,
      linalg::LinalgTilingOptions()
          .setLoopType(linalg::LinalgTilingLoopType::ParallelLoops)
          .setTileSizeComputationFunction(getInnerTileSizeFn)
          .setDistributionOptions(subgroupDistributionOptions),
      getLinalgMatchAndReplaceMarker(
          {getWorkgroupMemoryMarker(), getWorkgroupMarker()},
          getVectorizeMarker(), context));
}

//===----------------------------------------------------------------------===//
// Patterns and methods for thread tiling.
//===----------------------------------------------------------------------===//

/// Patterns for third level tiling to target invocations.
static void populateTilingToInvocationPatterns(
    MLIRContext *context, const LaunchConfig &launchConfig,
    OwningRewritePatternList &patterns) {
  linalg::TileSizeComputationFunction getInnerTileSizeFn =
      [&launchConfig](OpBuilder &builder, Operation *operation) {
        ArrayRef<int64_t> tileSizes = launchConfig.getTileSizes(operation, 2);
        if (tileSizes.empty()) return SmallVector<Value, 4>();
        SmallVector<Value, 4> tileSizesVal;
        tileSizesVal.reserve(tileSizes.size());
        for (auto val : tileSizes) {
          tileSizesVal.push_back(
              builder.create<ConstantIndexOp>(operation->getLoc(), val));
        }
        return tileSizesVal;
      };

  auto getThreadProcInfoFn = [&launchConfig](
                                 OpBuilder &builder, Location loc,
                                 ArrayRef<Range> parallelLoopRanges) {
    Type indexType = builder.getIndexType();
    SmallVector<linalg::ProcInfo, 2> procInfo(2);
    procInfo[1] = {builder.create<gpu::ThreadIdOp>(loc, indexType,
                                                   builder.getStringAttr("x")),
                   builder.create<ConstantIndexOp>(
                       loc, launchConfig.getWorkgroupSize()[0])};
    procInfo[0] = {builder.create<gpu::ThreadIdOp>(loc, indexType,
                                                   builder.getStringAttr("y")),
                   builder.create<ConstantIndexOp>(
                       loc, launchConfig.getWorkgroupSize()[1])};
    return procInfo;
  };
  linalg::LinalgLoopDistributionOptions subgroupDistributionOptions = {
      getThreadProcInfoFn,
      {linalg::DistributionMethod::CyclicNumProcsEqNumIters,
       linalg::DistributionMethod::CyclicNumProcsEqNumIters}};
  patterns.insert<linalg::LinalgTilingPattern<linalg::MatmulOp>,
                  linalg::LinalgTilingPattern<linalg::FillOp>>(
      context,
      linalg::LinalgTilingOptions()
          .setLoopType(linalg::LinalgTilingLoopType::ParallelLoops)
          .setTileSizeComputationFunction(getInnerTileSizeFn)
          .setDistributionOptions(subgroupDistributionOptions),
      getLinalgMatchAndReplaceMarker(
          {getWorkgroupMemoryMarker(), getWorkgroupMarker()},
          getVectorizeMarker(), context));
}

//====---------------------------------------------------------------------===//
// Patterns for vectorization
//====---------------------------------------------------------------------===//

static void populateVectorizationPatterns(MLIRContext *context,
                                          const LaunchConfig &launchConfig,
                                          OwningRewritePatternList &patterns) {
  patterns.insert<linalg::LinalgVectorizationPattern<linalg::MatmulOp>,
                  linalg::LinalgVectorizationPattern<linalg::FillOp>>(
      context,
      linalg::LinalgMarker(Identifier::get(getVectorizeMarker(), context)));
}

//====---------------------------------------------------------------------===//
// Patterns for unrolling vectors
//====---------------------------------------------------------------------===//

static void populateVectorUnrollPatterns(MLIRContext *context,
                                         OwningRewritePatternList &patterns) {
  patterns.insert<vector::UnrollVectorPattern<vector::TransferReadOp>,
                  vector::UnrollVectorPattern<vector::ContractionOp>,
                  vector::UnrollVectorPattern<vector::TransferWriteOp>>(
      context,
      vector::UnrollVectorOptions().setNativeShapeFn(getNativeVectorSize));
  vector::populateVectorToVectorCanonicalizationPatterns(patterns, context);
  vector::populateVectorToVectorTransformationPatterns(patterns, context);
}

//====---------------------------------------------------------------------===//
// Vector patterns
//====---------------------------------------------------------------------===//

static void applyVectorTransformation(FuncOp funcOp) {
  {
    OwningRewritePatternList vectorUnrollPatterns;
    populateVectorUnrollPatterns(funcOp.getContext(), vectorUnrollPatterns);
    applyPatternsAndFoldGreedily(funcOp, std::move(vectorUnrollPatterns));

    OwningRewritePatternList canonicalizationPatterns;
    vector::populateVectorSlicesLoweringPatterns(canonicalizationPatterns,
                                                 funcOp.getContext());
    applyPatternsAndFoldGreedily(funcOp, std::move(canonicalizationPatterns));
    LLVM_DEBUG({
      llvm::dbgs() << "--- After Vector Unroll ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });
  }

  {
    linalg::hoistRedundantVectorTransfers(funcOp);

    LLVM_DEBUG({
      llvm::dbgs() << "--- After Hoisting ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });
  }
}

//====---------------------------------------------------------------------===//
// Main pass implementation
//====---------------------------------------------------------------------===//

void LinalgTileAndFusePass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  LLVM_DEBUG(
      llvm::dbgs() << "--- IREE Linalg tile and fuse configuration ---\n";);
  for (FuncOp funcOp : module.getOps<FuncOp>()) {
    if (!isEntryPoint(funcOp)) continue;

    Region &body = funcOp.getBody();
    if (!llvm::hasSingleElement(body.getBlocks())) {
      funcOp.emitError("unhandled dispatch function with multiple blocks");
      return signalPassFailure();
    }
    Block &block = body.front();
    auto linalgOps = block.getOps<linalg::LinalgOp>();
    if (linalgOps.empty()) continue;

    LaunchConfig launchConfig;
    SmallVector<linalg::LinalgOp, 4> linalgOpsVec =
        llvm::to_vector<4>(llvm::map_range(linalgOps, [](Operation *op) {
          return cast<linalg::LinalgOp>(op);
        }));
    linalg::Aliases aliases;
    linalg::LinalgDependenceGraph dependenceGraph(aliases, linalgOpsVec);
    if (failed(launchConfig.init(context, dependenceGraph, options,
                                 linalgOpsVec))) {
      funcOp.emitError("unable to find launch configuration");
      return signalPassFailure();
    }

    LLVM_DEBUG({
      llvm::dbgs() << "@func " << funcOp.getName() << ": # workgroup sizes: [";
      interleaveComma(launchConfig.getWorkgroupSize(), llvm::dbgs());
      llvm::dbgs() << "]\n";
      for (auto op : linalgOps) {
        llvm::dbgs() << "\t" << op.getOperation()->getName() << " : ";
        TileSizesListType const &tileSizes = launchConfig.getTileSizes(op);
        llvm::dbgs() << "{";
        std::string sep = "";
        for (auto &level : enumerate(tileSizes)) {
          llvm::dbgs() << sep << level.index() << " : [";
          sep = ", ";
          interleaveComma(level.value(), llvm::dbgs());
          llvm::dbgs() << "]";
        }
        llvm::dbgs() << "}\n";
      }
    });

    if (failed(tileAndFuseLinalgBufferOps(funcOp, linalgOpsVec, dependenceGraph,
                                          launchConfig))) {
      return signalPassFailure();
    }

    LLVM_DEBUG({
      llvm::dbgs() << "--- After First level of tile+distribute ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });

    if (options.useWorkgroupMemory) {
      // The promotion patterns are put separate from the tiling patterns to
      // make sure that the allocated scratchspace memory is constant sizes
      // which requires some folding to trigger.
      OwningRewritePatternList promotionPatterns;
      populatePromotionPatterns(context, promotionPatterns);
      applyPatternsAndFoldGreedily(funcOp, std::move(promotionPatterns));
      applyCanonicalizationPatterns(context, funcOp);

      LLVM_DEBUG({
        llvm::dbgs() << "--- After Promotion  ---\n";
        funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
        llvm::dbgs() << "\n\n";
      });
    }

    if (launchConfig.useVectorize()) {
      {
        OwningRewritePatternList secondLevelTilingPatterns;
        populateTilingToSubgroupPatterns(context, launchConfig,
                                         secondLevelTilingPatterns);
        applyPatternsAndFoldGreedily(funcOp,
                                     std::move(secondLevelTilingPatterns));
        applyCanonicalizationPatterns(context, funcOp);
        promoteSingleIterationLoops(funcOp);

        LLVM_DEBUG({
          llvm::dbgs() << "--- After Second level Tiling  ---\n";
          funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
          llvm::dbgs() << "\n\n";
        });
      }

      {
        OwningRewritePatternList thirdLevelTilingPatterns;
        populateTilingToInvocationPatterns(context, launchConfig,
                                           thirdLevelTilingPatterns);
        applyPatternsAndFoldGreedily(funcOp,
                                     std::move(thirdLevelTilingPatterns));
        applyCanonicalizationPatterns(context, funcOp);
        promoteSingleIterationLoops(funcOp);

        LLVM_DEBUG({
          llvm::dbgs() << "--- After Third level Tiling  ---\n";
          funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
          llvm::dbgs() << "\n\n";
        });
      }

      {
        OwningRewritePatternList vectorizationPatterns;
        populateVectorizationPatterns(context, launchConfig,
                                      vectorizationPatterns);
        applyPatternsAndFoldGreedily(funcOp, std::move(vectorizationPatterns));
        LLVM_DEBUG({
          llvm::dbgs() << "--- After Vectorization ---\n";
          funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
          llvm::dbgs() << "\n\n";
        });
      }

      applyVectorTransformation(funcOp);
    }

    launchConfig.finalize(funcOp);
    SmallVector<linalg::LinalgOp, 1> toDelete;
    funcOp.walk([&](linalg::LinalgOp linalgOp) {
      if (hasMarker(linalgOp.getOperation(), getDeleteMarker()))
        linalgOp.erase();
    });
  }
}

//===----------------------------------------------------------------------===//
// Pass entry point and registration
//===----------------------------------------------------------------------===//

std::unique_ptr<OperationPass<ModuleOp>> createLinalgTileAndFusePass(
    const SPIRVCodegenOptions &options) {
  return std::make_unique<LinalgTileAndFusePass>(options);
}

static PassRegistration<LinalgTileAndFusePass> pass(
    "iree-codegen-linalg-tile-and-fuse",
    "Tile and fuse Linalg operations on buffers", [] {
      SPIRVCodegenOptions options = getSPIRVCodegenOptionsFromClOptions();
      return std::make_unique<LinalgTileAndFusePass>(options);
    });

}  // namespace iree_compiler
}  // namespace mlir

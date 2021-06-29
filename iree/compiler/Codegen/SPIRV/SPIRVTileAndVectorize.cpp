// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- SPIRVTileAndVectorize.cpp ------------------------------------------===//
//
// This pass tiles and vectorizes Linalg ops on buffers within in a single
// workgroup.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/SPIRV/KernelDispatchUtils.h"
#include "iree/compiler/Codegen/SPIRV/MemorySpace.h"
#include "iree/compiler/Codegen/SPIRV/Utils.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeDialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/Linalg/Analysis/DependenceAnalysis.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorTransforms.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Identifier.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopUtils.h"

#define DEBUG_TYPE "iree-spirv-tile-and-vectorize"

namespace mlir {
namespace iree_compiler {

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

/// Returns a Linalg marker that matches any of the `matchMarkers` and replaces
/// it with `replaceMarker`.
static linalg::LinalgTransformationFilter getLinalgMatchAndReplaceMarker(
    ArrayRef<StringRef> matchMarkers, StringRef replaceMarker,
    MLIRContext *context) {
  SmallVector<Identifier, 2> markers;
  markers.reserve(matchMarkers.size());
  for (StringRef marker : matchMarkers) {
    markers.emplace_back(Identifier::get(marker, context));
  }
  return linalg::LinalgTransformationFilter(
      markers, Identifier::get(replaceMarker, context));
}

/// Converts a symbolic GPU processor dimension to its numeric one.
static unsigned dimToIndex(StringRef dim) {
  return StringSwitch<unsigned>(dim).Case("x", 0).Case("y", 1).Case("z", 2);
}

//===----------------------------------------------------------------------===//
// Main pass
//===----------------------------------------------------------------------===//

namespace {
/// Function pass that implements tiling and fusion in Linalg on buffers.
class SPIRVTileAndVectorizePass
    : public SPIRVTileAndVectorizeBase<SPIRVTileAndVectorizePass> {
 public:
  SPIRVTileAndVectorizePass(const SPIRVCodegenOptions &passOptions)
      : options(passOptions) {}
  SPIRVTileAndVectorizePass(const SPIRVTileAndVectorizePass &pass)
      : options(pass.options) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, IREE::HAL::HALDialect, gpu::GPUDialect,
                    linalg::LinalgDialect, memref::MemRefDialect,
                    scf::SCFDialect, ShapeDialect, vector::VectorDialect>();
  }

  void runOnOperation() override;

 private:
  SPIRVCodegenOptions options;
};
}  // namespace

//===----------------------------------------------------------------------===//
// Patterns to promote subviews to workgroup memory
//===----------------------------------------------------------------------===//

namespace {
/// Pattern to promote matmul operands to workgroup memory.
struct PromoteMatmulSubviewsPattern
    : public linalg::LinalgPromotionPattern<linalg::MatmulOp> {
  PromoteMatmulSubviewsPattern(MLIRContext *context,
                               linalg::LinalgPromotionOptions options,
                               linalg::LinalgTransformationFilter marker,
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
template <typename ConvOpTy>
struct PromoteConvSubviewsPattern
    : public linalg::LinalgPromotionPattern<ConvOpTy> {
  PromoteConvSubviewsPattern(MLIRContext *context,
                             linalg::LinalgPromotionOptions options,
                             linalg::LinalgTransformationFilter marker,
                             PatternBenefit benefit = 1)
      : linalg::LinalgPromotionPattern<ConvOpTy>(
            context,
            options.setOperandsToPromote({0}).setUseFullTileBuffers(
                {false, false}),
            marker, benefit) {}
};
}  // namespace

static void populatePromotionPatterns(MLIRContext *context,
                                      RewritePatternSet &patterns) {
  patterns
      .insert<PromoteMatmulSubviewsPattern,
              PromoteConvSubviewsPattern<linalg::ConvInputNHWCFilterHWCFOp>>(
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
  for (size_t i = 0, e = numSubgroups.size(); i != e; ++i) {
    Value nprocs = builder.create<ConstantIndexOp>(loc, numSubgroups[i]);
    AffineExpr d0 = getAffineDimExpr(0, builder.getContext());
    AffineExpr s0 = getAffineSymbolExpr(0, builder.getContext());
    Value procId =
        makeComposedAffineApply(builder, loc, d0 % s0, {subgroupId, nprocs});
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
                            linalg::LinalgTransformationFilter marker,
                            PatternBenefit benefit = 1)
      : Base(context, options, marker, benefit) {}
};
}  // namespace

/// Patterns for second level tiling to target subgroups.
static void populateTilingToSubgroupPatterns(MLIRContext *context,
                                             const LaunchConfig &launchConfig,
                                             RewritePatternSet &patterns) {
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

  linalg::LinalgLoopDistributionOptions subgroupDistributionOptions;
  subgroupDistributionOptions.procInfo = getSubgroupProcInfoFn;
  subgroupDistributionOptions.distributionMethod = {
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
static void populateTilingToInvocationPatterns(MLIRContext *context,
                                               const LaunchConfig &launchConfig,
                                               RewritePatternSet &patterns) {
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

  auto getThreadProcInfoFn = [](OpBuilder &builder, Location loc,
                                ArrayRef<Range> parallelLoopRanges) {
    return getGPUProcessorIdsAndCounts<gpu::ThreadIdOp, gpu::BlockDimOp>(
        builder, loc, parallelLoopRanges.size());
  };
  linalg::LinalgLoopDistributionOptions invocationDistributionOptions;
  invocationDistributionOptions.procInfo = getThreadProcInfoFn;
  invocationDistributionOptions.distributionMethod = {
      {linalg::DistributionMethod::Cyclic, linalg::DistributionMethod::Cyclic,
       linalg::DistributionMethod::Cyclic}};

  auto tilingOptions =
      linalg::LinalgTilingOptions()
          .setLoopType(linalg::LinalgTilingLoopType::Loops)
          .setTileSizeComputationFunction(getInnerTileSizeFn)
          .setDistributionOptions(invocationDistributionOptions);

  patterns.insert<
      linalg::LinalgTilingPattern<linalg::MatmulOp>,
      linalg::LinalgTilingPattern<linalg::FillOp>,
      linalg::LinalgTilingPattern<linalg::BatchMatmulOp>,
      linalg::LinalgTilingPattern<linalg::ConvInputNWCFilterWCFOp>,
      linalg::LinalgTilingPattern<linalg::ConvInputNDHWCFilterDHWCFOp>,
      linalg::LinalgTilingPattern<linalg::DepthwiseConvInputNHWCFilterHWCFOp>,
      linalg::LinalgTilingPattern<linalg::GenericOp>,
      linalg::LinalgTilingPattern<linalg::PoolingNHWCMaxFOp>,
      linalg::LinalgTilingPattern<linalg::PoolingNHWCMinFOp>,
      linalg::LinalgTilingPattern<linalg::PoolingNHWCSumFOp>>(
      context, tilingOptions,
      getLinalgMatchAndReplaceMarker(
          {getWorkgroupMemoryMarker(), getWorkgroupMarker()},
          getVectorizeMarker(), context));

  patterns.insert<
      linalg::LinalgTilingPattern<linalg::ConvInputNHWCFilterHWCFOp>,
      linalg::LinalgTilingPattern<linalg::DepthwiseConvInputNHWCFilterHWCOp>>(
      context, tilingOptions,
      getLinalgMatchAndReplaceMarker(
          {getWorkgroupMemoryMarker(), getWorkgroupMarker()},
          getConvFilterTileMarker(), context));
}

/// Returns the corresponding range for the given `processorValue` is a GPU
/// thread id or block dim.
static Optional<std::pair<AffineExpr, AffineExpr>> getThreadRange(
    Value processorValue, SmallVectorImpl<Value> & /*dims*/,
    SmallVectorImpl<Value> & /*symbols*/, ArrayRef<int64_t> workgroupSize) {
  if (auto idOp = processorValue.getDefiningOp<gpu::ThreadIdOp>()) {
    OpBuilder builder(processorValue.getContext());
    unsigned index = dimToIndex(idOp.dimension());
    AffineExpr zero = builder.getAffineConstantExpr(0);
    AffineExpr ubExpr = builder.getAffineConstantExpr(workgroupSize[index]);
    return std::make_pair(zero, ubExpr - 1);
  }
  if (auto dimOp = processorValue.getDefiningOp<gpu::BlockDimOp>()) {
    OpBuilder builder(processorValue.getContext());
    unsigned index = dimToIndex(dimOp.dimension());
    AffineExpr bound = builder.getAffineConstantExpr(workgroupSize[index]);
    return std::make_pair(bound, bound);
  }
  return llvm::None;
}

//====---------------------------------------------------------------------===//
// Patterns for vectorization
//====---------------------------------------------------------------------===//

static void populateVectorizationPatterns(MLIRContext *context,
                                          const LaunchConfig &launchConfig,
                                          RewritePatternSet &patterns) {
  linalg::insertVectorizationPatterns<linalg::FillOp, linalg::GenericOp,
                                      linalg::ContractionOpInterface>(
      patterns, linalg::LinalgVectorizationOptions(),
      linalg::LinalgTransformationFilter(
          Identifier::get(getVectorizeMarker(), context)));
}

//====---------------------------------------------------------------------===//
// Patterns for unrolling vectors
//====---------------------------------------------------------------------===//

static void populateVectorUnrollPatterns(MLIRContext *context,
                                         RewritePatternSet &patterns) {
  patterns.insert<vector::UnrollVectorPattern>(
      context,
      vector::UnrollVectorOptions().setNativeShapeFn(getSPIRVNativeVectorSize));
}

namespace {

/// Workaround SPIR-V backend limitations. SPIR-V vetorization pass relies on
/// unrolling to reduce instructions to a vector size we can convert to SPIR-V.
/// When vectorization creates transpose those block unrolling and result in
/// large vector we currently cannot lower. For now we always merge the
/// transpose into the contract op so that it can be unrolled.
// TODO(thomasraoux): Make transpose work with the current unrolling mechanism
// or replace unrolling.
class CombineContractTranspose final
    : public OpRewritePattern<vector::ContractionOp> {
 public:
  using OpRewritePattern<vector::ContractionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ContractionOp op,
                                PatternRewriter &rewriter) const override {
    // Perform lhs + rhs transpositions to conform to matmul row-major
    // semantics. Bail out if the contraction cannot be put in this form.
    MLIRContext *ctx = op.getContext();
    Location loc = op.getLoc();
    bool foundTranspose = false;
    std::array<Value, 3> sources = {op.lhs(), op.rhs(), op.acc()};
    SmallVector<AffineMap> newMaps;
    SmallVector<Value> newSources;
    for (auto source : llvm::enumerate(sources)) {
      auto map =
          op.indexing_maps()[source.index()].cast<AffineMapAttr>().getValue();
      auto tranposeOp = source.value().getDefiningOp<vector::TransposeOp>();
      if (!tranposeOp) {
        newSources.push_back(source.value());
        newMaps.push_back(map);
        continue;
      }
      SmallVector<int64_t, 3> perm;
      tranposeOp.getTransp(perm);
      SmallVector<AffineExpr> exprs(perm.size());
      for (auto remap : llvm::enumerate(perm)) {
        exprs[remap.value()] = map.getResult(remap.index());
      }
      newMaps.push_back(
          AffineMap::get(map.getNumDims(), map.getNumSymbols(), exprs, ctx));
      newSources.push_back(tranposeOp.vector());
      foundTranspose = true;
    }
    if (!foundTranspose) return failure();

    Value res = rewriter.create<vector::ContractionOp>(
        loc, newSources[0], newSources[1], newSources[2],
        rewriter.getAffineMapArrayAttr(newMaps), op.iterator_types());
    rewriter.replaceOp(op, res);
    return success();
  }
};

}  // namespace

//====---------------------------------------------------------------------===//
// Vector patterns
//====---------------------------------------------------------------------===//

static void applyVectorTransformation(FuncOp funcOp) {
  auto targetEnv = spirv::TargetEnv(spirv::lookupTargetEnv(funcOp));
  bool useCooperativeMatrix =
      targetEnv.allows(spirv::Capability::CooperativeMatrixNV) &&
      targetEnv.allows(spirv::Extension::SPV_NV_cooperative_matrix);
  {
    {
      RewritePatternSet vectorUnrollPatterns(funcOp.getContext());
      populateVectorUnrollPatterns(funcOp.getContext(), vectorUnrollPatterns);
      (void)applyPatternsAndFoldGreedily(funcOp,
                                         std::move(vectorUnrollPatterns));
    }
    {
      RewritePatternSet canonicalizationPatterns1(funcOp.getContext());

      vector::populateVectorToVectorTransformationPatterns(
          canonicalizationPatterns1);
      vector::populateVectorToVectorCanonicalizationPatterns(
          canonicalizationPatterns1);
      vector::populateSplitVectorTransferPatterns(canonicalizationPatterns1);
      (void)applyPatternsAndFoldGreedily(funcOp,
                                         std::move(canonicalizationPatterns1));

      RewritePatternSet canonicalizationPatterns2(funcOp.getContext());
      vector::populateVectorSlicesLoweringPatterns(canonicalizationPatterns2);
      vector::populateVectorTransferLoweringPatterns(canonicalizationPatterns2);
      (void)applyPatternsAndFoldGreedily(funcOp,
                                         std::move(canonicalizationPatterns2));

      if (useCooperativeMatrix) {
        // When using cooperative matrix we don't want to lower the contract,
        // instead we want to merge contract and transpose so that they can be
        // converted to cooperative matrix matmul op.
        // TODO(thomasraoux): remove that once we support cooperative matrix
        // lowering in MLIR core.
        RewritePatternSet combineTransposePatterns(funcOp.getContext());
        combineTransposePatterns.add<CombineContractTranspose>(
            funcOp.getContext());
        (void)applyPatternsAndFoldGreedily(funcOp,
                                           std::move(combineTransposePatterns));
      } else {
        RewritePatternSet contractLoweringPatterns(funcOp.getContext());
        vector::populateVectorContractLoweringPatterns(
            contractLoweringPatterns,
            vector::VectorTransformsOptions().setVectorTransformsOptions(
                vector::VectorContractLowering::OuterProduct));
        (void)applyPatternsAndFoldGreedily(funcOp,
                                           std::move(contractLoweringPatterns));
      }
    }
    LLVM_DEBUG({
      llvm::dbgs() << "--- After unrolling vector ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });
  }

  {
    linalg::hoistRedundantVectorTransfers(funcOp);

    LLVM_DEBUG({
      llvm::dbgs() << "--- After hoisting vector transfers ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });
  }
}

//====---------------------------------------------------------------------===//
// Patterns to tile convolution window dimensions
//====---------------------------------------------------------------------===//

static void populateTilingConvFilterPatterns(
    MLIRContext *context, RewritePatternSet &patterns,
    const LaunchConfig &launchConfig,
    linalg::LinalgTransformationFilter marker) {
  auto getTileSizeFn = [&launchConfig](OpBuilder &builder, Operation *op) {
    SmallVector<Value, 4> tileSizes;
    ArrayRef<int64_t> fourthLevel = launchConfig.getTileSizes(op, 3);
    tileSizes.reserve(fourthLevel.size());

    Location loc = op->getLoc();
    for (int64_t size : fourthLevel) {
      tileSizes.push_back(builder.create<ConstantIndexOp>(loc, size));
    }
    return tileSizes;
  };

  auto tilingOptions = linalg::LinalgTilingOptions()
                           .setLoopType(linalg::LinalgTilingLoopType::Loops)
                           .setTileSizeComputationFunction(getTileSizeFn);

  patterns.insert<
      linalg::LinalgTilingPattern<linalg::ConvInputNHWCFilterHWCFOp>,
      linalg::LinalgTilingPattern<linalg::DepthwiseConvInputNHWCFilterHWCFOp>,
      linalg::LinalgTilingPattern<linalg::DepthwiseConvInputNHWCFilterHWCOp>>(
      context, tilingOptions, marker);
}

//====---------------------------------------------------------------------===//
// Patterns to lower linalg ops to loops
//====---------------------------------------------------------------------===//

template <typename OpTy>
struct LowerToLoops final : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    // Only handle the cases where tiling to invocations was done, where tiling
    // convolution filters or vectorization is expected.
    if (!hasMarker(op, {getConvFilterTileMarker(), getVectorizeMarker()}))
      return failure();

    if (linalg::linalgOpToLoops(rewriter, op)) {
      rewriter.eraseOp(op);
      return success();
    }

    return failure();
  }
};

//====---------------------------------------------------------------------===//
// Main pass implementation
//====---------------------------------------------------------------------===//

void SPIRVTileAndVectorizePass::runOnOperation() {
  MLIRContext *context = &getContext();
  IREE::HAL::ExecutableVariantOp variantOp = getOperation();
  ModuleOp module = variantOp.getInnerModule();

  for (FuncOp funcOp : module.getOps<FuncOp>()) {
    if (!isEntryPoint(funcOp)) continue;

    SmallVector<linalg::LinalgOp, 4> linalgOps;
    SmallVector<Operation *, 4> tiledLoops;

    if (failed(getLinalgOps(funcOp, linalgOps, tiledLoops))) {
      // Nothing to do here.
      continue;
    }

    linalg::Aliases aliases;
    linalg::LinalgDependenceGraph dependenceGraph(aliases, linalgOps);
    Optional<LaunchConfig> launchConfigOpt =
        initGPULaunchConfig(context, dependenceGraph, options, linalgOps);
    if (!launchConfigOpt) {
      // No configuration to tile and vectorize. Nothing to do here.
      continue;
    }
    LaunchConfig &launchConfig = *launchConfigOpt;

    LLVM_DEBUG({
      llvm::dbgs() << "\n--- Linalg tile configuration ---\n";
      llvm::dbgs() << "@func " << funcOp.getName() << ": # workgroup sizes: [";
      interleaveComma(launchConfig.getWorkgroupSize(), llvm::dbgs());
      llvm::dbgs() << "]\n";
      for (auto op : linalgOps) {
        llvm::dbgs() << "\t" << op.getOperation()->getName() << " : ";
        TileSizesListTypeRef tileSizes = launchConfig.getTileSizes(op);
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

    if (options.useWorkgroupMemory) {
      // The promotion patterns are put separate from the tiling patterns to
      // make sure that the allocated scratchspace memory is constant sizes
      // which requires some folding to trigger.
      RewritePatternSet promotionPatterns(&getContext());
      populatePromotionPatterns(context, promotionPatterns);
      (void)applyPatternsAndFoldGreedily(funcOp, std::move(promotionPatterns));

      RewritePatternSet promotionCanonicalizationPatterns =
          linalg::getLinalgTilingCanonicalizationPatterns(context);
      populateAffineMinCanonicalizationPattern(
          promotionCanonicalizationPatterns);
      (void)applyPatternsAndFoldGreedily(
          funcOp, std::move(promotionCanonicalizationPatterns));

      LLVM_DEBUG({
        llvm::dbgs() << "--- After workgroup memory promotion  ---\n";
        funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
        llvm::dbgs() << "\n\n";
      });
    }

    // TODO(thomasraoux, antiagainst): Tiling to subgroups shouldn't be
    // controlled by vectorization. This is needed due to historical reasons.
    // Change the second level tiling to cyclic to loops and remove this.
    if (launchConfig.useVectorize()) {
      RewritePatternSet secondLevelTilingPatterns(&getContext());
      populateTilingToSubgroupPatterns(context, launchConfig,
                                       secondLevelTilingPatterns);
      (void)applyPatternsAndFoldGreedily(funcOp,
                                         std::move(secondLevelTilingPatterns));

      RewritePatternSet secondLevelTilingCanonicalizationPatterns =
          linalg::getLinalgTilingCanonicalizationPatterns(context);
      populateAffineMinCanonicalizationPattern(
          secondLevelTilingCanonicalizationPatterns);
      (void)applyPatternsAndFoldGreedily(
          funcOp, std::move(secondLevelTilingCanonicalizationPatterns));
      promoteSingleIterationLoops(funcOp);

      LLVM_DEBUG({
        llvm::dbgs() << "--- After tiling to subgroups ---\n";
        funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
        llvm::dbgs() << "\n\n";
      });
    }

    {
      RewritePatternSet thirdLevelTilingPatterns(&getContext());
      populateTilingToInvocationPatterns(context, launchConfig,
                                         thirdLevelTilingPatterns);
      (void)applyPatternsAndFoldGreedily(funcOp,
                                         std::move(thirdLevelTilingPatterns));

      // Remove trip-one loops created during cyclic loop distribution if we can
      // prove the tiling was perfect.
      RewritePatternSet canoncalizationPatterns(context);
      populateAffineMinSCFCanonicalizationPattern(canoncalizationPatterns);
      ArrayRef<int64_t> workgroupSize = launchConfig.getWorkgroupSize();
      auto getThreadRangeFn = [workgroupSize](Value processorValue,
                                              SmallVectorImpl<Value> &dims,
                                              SmallVectorImpl<Value> &symbols) {
        return getThreadRange(processorValue, dims, symbols, workgroupSize);
      };
      populateRemoveSingleIterationLoopPattern(canoncalizationPatterns,
                                               getThreadRangeFn);
      (void)applyPatternsAndFoldGreedily(funcOp,
                                         std::move(canoncalizationPatterns));

      // Perform generic canonicalization.
      RewritePatternSet threadLevelTilingCanonicalizationPatterns =
          linalg::getLinalgTilingCanonicalizationPatterns(context);
      populateAffineMinCanonicalizationPattern(
          threadLevelTilingCanonicalizationPatterns);
      (void)applyPatternsAndFoldGreedily(
          funcOp, std::move(threadLevelTilingCanonicalizationPatterns));

      LLVM_DEBUG({
        llvm::dbgs() << "--- After tiling to invocations ---\n";
        funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
        llvm::dbgs() << "\n\n";
      });
    }

    {
      RewritePatternSet tilingPatterns(&getContext());
      auto marker = getLinalgMatchAndReplaceMarker(
          getConvFilterTileMarker(), getVectorizeMarker(), context);
      populateTilingConvFilterPatterns(context, tilingPatterns, launchConfig,
                                       marker);
      populateFoldGPUProcessorIDUsesPatterns(context, tilingPatterns);
      tilingPatterns.insert<linalg::AffineMinSCFCanonicalizationPattern>(
          context);
      (void)applyPatternsAndFoldGreedily(funcOp, std::move(tilingPatterns));

      RewritePatternSet convTilingCanonicalizationPatterns =
          linalg::getLinalgTilingCanonicalizationPatterns(context);
      populateAffineMinCanonicalizationPattern(
          convTilingCanonicalizationPatterns);
      (void)applyPatternsAndFoldGreedily(
          funcOp, std::move(convTilingCanonicalizationPatterns));

      LLVM_DEBUG({
        llvm::dbgs() << "--- After tiling convolution filter  ---\n";
        funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
        llvm::dbgs() << "\n\n";
      });
    }

    if (launchConfig.useVectorize()) {
      {
        RewritePatternSet vectorizationPatterns(&getContext());
        populateVectorizationPatterns(context, launchConfig,
                                      vectorizationPatterns);
        populateLinalgToVectorVectorizeConvPatterns(context,
                                                    vectorizationPatterns);
        (void)applyPatternsAndFoldGreedily(funcOp,
                                           std::move(vectorizationPatterns));
        LLVM_DEBUG({
          llvm::dbgs() << "--- After vectorization ---\n";
          funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
          llvm::dbgs() << "\n\n";
        });
      }

      // TODO: This should be a folding of Add into Contract in core but while
      // they live in different dialects, it is not possible without unnatural
      // dependencies.
      funcOp.walk([&](Operation *op) {
        if (auto contract = canonicalizeContractionAdd(op))
          op->replaceAllUsesWith(contract);
      });

      applyVectorTransformation(funcOp);
    }

    // Lower ops that were tiled to invocations but not vectorized to loops.
    // TODO(antiagainst): This is here now to simplify the interaction with
    // ConvertToGPUPass, where we finally lower away all Linalg ops. Once that
    // pass is cleaned up, we can invoke createConvertLinalgToLoopsPass
    // directly.
    {
      RewritePatternSet patterns(context);
      patterns
          .add<LowerToLoops<linalg::BatchMatmulOp>,
               LowerToLoops<linalg::ConvInputNWCFilterWCFOp>,
               LowerToLoops<linalg::ConvInputNHWCFilterHWCFOp>,
               LowerToLoops<linalg::ConvInputNDHWCFilterDHWCFOp>,
               LowerToLoops<linalg::DepthwiseConvInputNHWCFilterHWCFOp>,
               LowerToLoops<linalg::DepthwiseConvInputNHWCFilterHWCOp>,
               LowerToLoops<linalg::FillOp>, LowerToLoops<linalg::GenericOp>,
               LowerToLoops<linalg::MatmulOp>,
               LowerToLoops<linalg::PoolingNHWCMaxFOp>,
               LowerToLoops<linalg::PoolingNHWCMinFOp>,
               LowerToLoops<linalg::PoolingNHWCSumFOp>>(context);
      (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
    }

    launchConfig.finalize(funcOp);
  }
}

//===----------------------------------------------------------------------===//
// Pass entry point and registration
//===----------------------------------------------------------------------===//

std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createSPIRVTileAndVectorizePass(const SPIRVCodegenOptions &options) {
  return std::make_unique<SPIRVTileAndVectorizePass>(options);
}

}  // namespace iree_compiler
}  // namespace mlir

// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- SPIRVTileAndDistribute.cpp -----------------------------------------===//
//
// This pass tiles and distributes Linalg ops with buffer semantics to subgroups
// and invocations.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/SPIRV/Utils.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorTransforms.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopUtils.h"

#define DEBUG_TYPE "iree-spirv-tile-and-distribute"

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
// Subgroup tiling patterns
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
                                             RewritePatternSet &patterns) {
  auto getInnerTileSizeFn = [&](OpBuilder &builder,
                                Operation *operation) -> SmallVector<Value, 4> {
    SmallVector<int64_t> tileSizes = getTileSizes(operation, 1);
    return llvm::to_vector<4>(
        llvm::map_range(tileSizes, [&](int64_t v) -> Value {
          return builder.create<ConstantIndexOp>(operation->getLoc(), v);
        }));
  };

  auto getSubgroupProcInfoFn = [&](OpBuilder &builder, Location loc,
                                   ArrayRef<Range> parallelLoopRanges) {
    // TODO(ravishankarm): For now assume that there is always a single subgroup
    std::array<int64_t, 3> numSubgroups = {1, 1, 1};
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
// Invocation tiling patterns
//===----------------------------------------------------------------------===//

/// Patterns for third level tiling to target invocations.
static void populateTilingToInvocationPatterns(MLIRContext *context,
                                               RewritePatternSet &patterns) {
  linalg::TileSizeComputationFunction getInnerTileSizeFn =
      [&](OpBuilder &builder, Operation *operation) {
        SmallVector<int64_t> tileSizes = getTileSizes(operation, 2);
        return llvm::to_vector<4>(
            llvm::map_range(tileSizes, [&](int64_t v) -> Value {
              return builder.create<ConstantIndexOp>(operation->getLoc(), v);
            }));
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

  patterns.insert<linalg::LinalgTilingPattern<linalg::MatmulOp>,
                  linalg::LinalgTilingPattern<linalg::FillOp>,
                  linalg::LinalgTilingPattern<linalg::BatchMatmulOp>,
                  linalg::LinalgTilingPattern<linalg::Conv1DNwcWcfOp>,
                  linalg::LinalgTilingPattern<linalg::Conv3DNdhwcDhwcfOp>,
                  linalg::LinalgTilingPattern<linalg::DepthwiseConv2DNhwcOp>,
                  linalg::LinalgTilingPattern<linalg::GenericOp>,
                  linalg::LinalgTilingPattern<linalg::PoolingNhwcMaxOp>,
                  linalg::LinalgTilingPattern<linalg::PoolingNhwcMinOp>,
                  linalg::LinalgTilingPattern<linalg::PoolingNhwcSumOp>>(
      context, tilingOptions,
      getLinalgMatchAndReplaceMarker(
          {getWorkgroupMemoryMarker(), getWorkgroupMarker()},
          getVectorizeMarker(), context));

  patterns.insert<linalg::LinalgTilingPattern<linalg::Conv2DNhwcHwcfOp>,
                  linalg::LinalgTilingPattern<linalg::DepthwiseConv2DNhwOp>>(
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
// Convolution filter tiling patterns
//====---------------------------------------------------------------------===//

static void populateTilingConvFilterPatterns(
    MLIRContext *context, RewritePatternSet &patterns,
    linalg::LinalgTransformationFilter marker) {
  auto getTileSizeFn = [&](OpBuilder &builder, Operation *op) {
    SmallVector<Value, 4> tileSizes;
    SmallVector<int64_t, 4> fourthLevel = getTileSizes(op, 3);
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

  patterns.insert<linalg::LinalgTilingPattern<linalg::Conv2DNhwcHwcfOp>,
                  linalg::LinalgTilingPattern<linalg::DepthwiseConv2DNhwOp>,
                  linalg::LinalgTilingPattern<linalg::DepthwiseConv2DNhwcOp>>(
      context, tilingOptions, marker);
}

//===----------------------------------------------------------------------===//
// Main pass
//===----------------------------------------------------------------------===//

namespace {
/// Function pass that implements tiling and distributing Linalg ops with
/// buffer semantics.
class SPIRVTileAndDistributePass
    : public SPIRVTileAndDistributeBase<SPIRVTileAndDistributePass> {
 public:
  SPIRVTileAndDistributePass() = default;
  SPIRVTileAndDistributePass(const SPIRVTileAndDistributePass &pass) = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, gpu::GPUDialect, linalg::LinalgDialect,
                    memref::MemRefDialect, scf::SCFDialect,
                    vector::VectorDialect>();
  }

  void runOnOperation() override;
};
}  // namespace

//====---------------------------------------------------------------------===//
// Main pass implementation
//====---------------------------------------------------------------------===//

void SPIRVTileAndDistributePass::runOnOperation() {
  MLIRContext *context = &getContext();
  FuncOp funcOp = getOperation();
  auto entryPointOp = getEntryPoint(funcOp);
  if (!entryPointOp) return;

  {
    RewritePatternSet subgroupTilingPatterns(&getContext());
    populateTilingToSubgroupPatterns(context, subgroupTilingPatterns);
    (void)applyPatternsAndFoldGreedily(funcOp,
                                       std::move(subgroupTilingPatterns));

    RewritePatternSet canonicalizationPatterns =
        linalg::getLinalgTilingCanonicalizationPatterns(context);
    populateAffineMinCanonicalizationPattern(canonicalizationPatterns);
    (void)applyPatternsAndFoldGreedily(funcOp,
                                       std::move(canonicalizationPatterns));
    promoteSingleIterationLoops(funcOp);

    LLVM_DEBUG({
      llvm::dbgs() << "--- After tiling to subgroups ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });
  }

  {
    RewritePatternSet invocationTilingPatterns(&getContext());
    populateTilingToInvocationPatterns(context, invocationTilingPatterns);
    (void)applyPatternsAndFoldGreedily(funcOp,
                                       std::move(invocationTilingPatterns));

    // Remove trip-one loops created during cyclic loop distribution if we can
    // prove the tiling was perfect.
    RewritePatternSet canoncalizationPatterns(context);
    populateAffineMinSCFCanonicalizationPattern(canoncalizationPatterns);
    SmallVector<int64_t> workgroupSize = getWorkgroupSize(entryPointOp);
    if (workgroupSize.empty()) {
      entryPointOp.emitError("expected to have workgroup_size attribute");
      return signalPassFailure();
    }
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
    RewritePatternSet convFilterTilingPatterns(&getContext());
    auto marker = getLinalgMatchAndReplaceMarker(getConvFilterTileMarker(),
                                                 getVectorizeMarker(), context);
    populateTilingConvFilterPatterns(context, convFilterTilingPatterns, marker);
    populateFoldGPUProcessorIDUsesPatterns(context, convFilterTilingPatterns);
    convFilterTilingPatterns
        .insert<linalg::AffineMinSCFCanonicalizationPattern>(context);
    (void)applyPatternsAndFoldGreedily(funcOp,
                                       std::move(convFilterTilingPatterns));

    RewritePatternSet canonicalizationPatterns =
        linalg::getLinalgTilingCanonicalizationPatterns(context);
    populateAffineMinCanonicalizationPattern(canonicalizationPatterns);
    (void)applyPatternsAndFoldGreedily(funcOp,
                                       std::move(canonicalizationPatterns));

    LLVM_DEBUG({
      llvm::dbgs() << "--- After tiling convolution filter  ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });
  }
}

//===----------------------------------------------------------------------===//
// Pass entry point and registration
//===----------------------------------------------------------------------===//

std::unique_ptr<OperationPass<FuncOp>> createSPIRVTileAndDistributePass() {
  return std::make_unique<SPIRVTileAndDistributePass>();
}

}  // namespace iree_compiler
}  // namespace mlir

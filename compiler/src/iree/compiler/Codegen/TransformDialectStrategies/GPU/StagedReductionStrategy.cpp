// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/TransformDialectStrategies/GPU/StagedReductionStrategy.h"

#include "iree-dialects/Dialect/LinalgTransform/StructuredTransformOpsExt.h"
#include "iree/compiler/Codegen/Common/TransformExtensions/CommonExtensions.h"
#include "iree/compiler/Codegen/LLVMGPU/TransformExtensions/LLVMGPUExtensions.h"
#include "iree/compiler/Codegen/TransformDialectStrategies/Common/Common.h"
#include "iree/compiler/Codegen/TransformDialectStrategies/GPU/Common.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

using namespace mlir;

#define DEBUG_TYPE "iree-transform-builder"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

// TODO: significantly better namespacing.
using iree_compiler::IREE::transform_dialect::ApplyPatternsOp;
using iree_compiler::IREE::transform_dialect::ApplyPatternsOpPatterns;
using iree_compiler::IREE::transform_dialect::ForeachThreadToWorkgroupOp;
using iree_compiler::IREE::transform_dialect::
    MapNestedForeachThreadToGpuThreadsOp;
using iree_compiler::IREE::transform_dialect::VectorToWarpExecuteOnLane0Op;
using iree_compiler::IREE::transform_dialect::VectorWarpDistributionOp;
using transform::FuseIntoContainingOp;
using transform::MatchOp;
using transform::ScalarizeOp;
using transform::SequenceOp;
using transform_ext::MatchCallbackOp;
using transform_ext::RegisterMatchCallbacksOp;
using transform_ext::StructuredOpMatcher;

using iree_compiler::gpu::AbstractReductionStrategy;
using iree_compiler::gpu::adjustNumberOfWarpsForBlockShuffle;
using iree_compiler::gpu::build1DSplittingStrategyWithOptionalThreadMapping;
using iree_compiler::gpu::buildCommonTrailingStrategy;
using iree_compiler::gpu::buildDistributeVectors;
using iree_compiler::gpu::kCudaMaxNumThreads;
using iree_compiler::gpu::kCudaMaxVectorLoadBitWidth;
using iree_compiler::gpu::kCudaWarpSize;
using iree_compiler::gpu::ReductionConfig;
using iree_compiler::gpu::scaleUpByBitWidth;
using iree_compiler::gpu::StagedReductionStrategy;

StagedReductionStrategy
mlir::iree_compiler::gpu::StagedReductionStrategy::create(
    MLIRContext *context,
    const transform_ext::MatchedReductionCaptures &captures) {
  ReductionConfig gpuReductionConfig = getStagedReductionConfig(captures);
  StagedReductionStrategy strategy(context, captures,
                                   gpuReductionConfig.maxNumThreads,
                                   gpuReductionConfig.vectorSize);
  LLVM_DEBUG(DBGS() << "use staged reduction strategy\n");
  return strategy;
}

void mlir::iree_compiler::gpu::StagedReductionStrategy::compute(
    int64_t maxNumThreadsToUse, int64_t maxVectorSize) {
  assert(maxNumThreadsToUse > 0 && "maxNumThreadsToUse must be > 0");
  assert(maxNumThreadsToUse >= kCudaWarpSize && "need at least a warp?");
  assert(maxVectorSize > 0 && "maxVectorSize must be > 0");

  // Block-level
  // ===========
  // Tile all the parallel dimensions to 1 and create many blocks.
  // TODO: Investigate taking some sizes that divide the dimensions and make
  // the kernel meatier.
  int64_t numParallelLoops = captures.reductionRank - 1;
  workgroupTileSizes.append(numParallelLoops, 1);

  // Thread-level
  // ============
  // Stage 1
  // -------
  // Maximal vector size that divides the problem size.
  // TODO: Split to ensure 4 on most of the problem and use a 1-epilogue.
  int64_t reductionDimensionSize = captures.reductionOpSizes.back();
  // Tile reduction to the maximal multiple `vectorSize` allowed.
  // This locally reduces the large unknown reduction into a guaranteed
  // multiple of `vectorSize`.
  if (ShapedType::isDynamic(reductionDimensionSize)) {
    // In the dynamic case, always run vector size of 1 and pad to the maximal
    // warp size below the `maxNumThreadsToUse` limit.
    vectorSize = 1;
    numThreadsXInBlock =
        iree_compiler::previousMultipleOf(maxNumThreadsToUse, kCudaWarpSize);
  } else {
    // Adjust the vector size to the max power of 2 that divides the reduction,
    // this dimensions the vector properly, whatever the elemental type.
    assert((maxVectorSize & maxVectorSize - 1) == 0 &&
           "maxVectorSize must be a power of 2");
    // TODO: we could also split out the first multiple of vectorSize instead
    // of reducing the vectorSize. This is better done with future stride /
    // alignment in mind.
    // TODO: splitting here also requires the post-bufferization privatization
    // analysis (see #11715).
    for (vectorSize = maxVectorSize; vectorSize > 1; vectorSize >>= 1)
      if (reductionDimensionSize % vectorSize == 0) break;
    // Pad to the next multiple of the warp size above
    // `reductionDimensionSize / vectorSize` but below `maxNumThreadsToUse`.
    numThreadsXInBlock = std::min(
        iree_compiler::nextMultipleOf(reductionDimensionSize / vectorSize,
                                      kCudaWarpSize),
        iree_compiler::previousMultipleOf(maxNumThreadsToUse, kCudaWarpSize));
  }
}

/// Build transform IR to split the reduction into a parallel and combiner part.
/// Then tile the parallel part and map it to `tileSize` threads, each reducing
/// on `vectorSize` elements.
/// Lastly, fuse the newly created fill and elementwise operations into the
/// resulting containing foreach_thread op.
/// Return a triple of handles to (foreach_thread, fill, combiner)
static std::tuple<Value, Value, Value> buildTileReductionUsingScfForeach(
    ImplicitLocOpBuilder &b, Value gridReductionH, int64_t reductionRank,
    int64_t tileSize, int64_t reductionVectorSize) {
  auto threadX = mlir::gpu::GPUThreadMappingAttr::get(b.getContext(),
                                                      mlir::gpu::Threads::DimX);

  SmallVector<int64_t> leadingParallelDims(reductionRank - 1, 0);
  SmallVector<int64_t> numThreads = leadingParallelDims;
  numThreads.push_back(tileSize);
  SmallVector<int64_t> tileSizes = leadingParallelDims;
  tileSizes.push_back(reductionVectorSize);
  auto tileReduction = b.create<transform::TileReductionUsingForeachThreadOp>(
      /*target=*/gridReductionH,
      /*numThreads=*/numThreads,
      /*tileSizes=*/tileSizes,
      /*threadDimMapping=*/b.getArrayAttr(threadX));
  Value blockParallelForeachThreadOp = tileReduction.getForeachThreadOp();
  Value blockParallelFillH = tileReduction.getFillOp();
  Value blockCombinerOpH = tileReduction.getCombiningLinalgOp();
  // Fuse the fill and elementwise to privatize them.
  blockParallelFillH = b.create<FuseIntoContainingOp>(
      blockParallelFillH, blockParallelForeachThreadOp);
  return std::make_tuple(blockParallelForeachThreadOp, blockParallelFillH,
                         blockCombinerOpH);
}

static void buildStagedReductionStrategyFindBetterName(
    ImplicitLocOpBuilder &b, Value gridReductionH, Value maybeTiledLeadingH,
    Value maybeTiledTrailingH, const StagedReductionStrategy &strategy) {
  // Map the potential maybeTiledLeadingH.
  // TODO: Consider fusing leading elementwise into threads.
  if (strategy.captures.maybeLeadingRank > 0) {
    int64_t vectorSize =
        kCudaMaxVectorLoadBitWidth /
        strategy.captures.maybeLeadingOutputElementalTypeBitWidth;
    assert((vectorSize & (vectorSize - 1)) == 0 && "size must be power of 2");
    build1DSplittingStrategyWithOptionalThreadMapping(
        b, maybeTiledLeadingH, strategy.captures.maybeLeadingRank,
        strategy.captures.leadingOpSizes, strategy.getNumThreadsXInBlock(),
        vectorSize);
  }

  // Staged reduction step 1: break gridReductionH apart.
  auto [blockParallelForeachThreadOp, blockParallelFillH, blockCombinerOpH] =
      buildTileReductionUsingScfForeach(
          b, gridReductionH, strategy.captures.reductionRank,
          strategy.getNumThreadsXInBlock(), strategy.getVectorSize());

  // Staged reduction step 2: multi-warp shuffle reduce.
  // Map the combiner reduction to one thread along y so it can be mapped
  // further via predication.
  auto threadY = mlir::gpu::GPUThreadMappingAttr::get(b.getContext(),
                                                      mlir::gpu::Threads::DimY);
  iree_compiler::buildTileFuseDistToForeachThreadWithTileSizes(
      b, blockCombinerOpH, {}, getAsOpFoldResult(b.getI64ArrayAttr({1})),
      b.getArrayAttr(threadY));

  // Map the potential maybeTiledTrailingH.
  if (strategy.captures.maybeTrailingRank > 0) {
    int64_t vectorSize =
        (4 * 32) / strategy.captures.maybeTrailingOutputElementalTypeBitWidth;
    build1DSplittingStrategyWithOptionalThreadMapping(
        b, maybeTiledTrailingH, strategy.captures.maybeTrailingRank,
        strategy.captures.trailingOpSizes, strategy.getNumThreadsXInBlock(),
        vectorSize);
  }
}

/// Builds the transform IR tiling reductions for CUDA targets. Supports
/// reductions in the last dimension, with optional leading and trailing
/// elementwise operations.
void mlir::iree_compiler::gpu::buildStagedReductionStrategy(
    ImplicitLocOpBuilder &b, Value variantH,
    const StagedReductionStrategy &strategy) {
  // Step 1. Call the matcher. Note that this is the same matcher as used to
  // trigger this compilation path, so it must always apply.
  b.create<RegisterMatchCallbacksOp>();
  auto [maybeLeadingH, fillH, reductionH, maybeTrailingH] =
      unpackRegisteredMatchCallback<4>(
          b, "reduction", transform::FailurePropagationMode::Propagate,
          variantH);

  // Step 2. Use tiling to introduce a single-iteration loop mapped to a
  // single block/workgroup. Keep everything fused.
  auto [maybeLeadingHBlock, gridFillH, gridReductionH,
        maybeTiledTrailingHBlock] =
      buildReductionStrategyBlockDistribution(
          b, maybeLeadingH, fillH, reductionH, maybeTrailingH, strategy);

  // Step 3. Split the reduction and tile the pieces to ensure vector
  // load/stores and mapping to a single warp with shuffles.
  // TODO: consider fusing gridFillH.
  buildStagedReductionStrategyFindBetterName(
      b, gridReductionH, maybeLeadingHBlock, maybeTiledTrailingHBlock,
      strategy);

  // Step 4-5. Common trailing steps.
  auto [variantH2, funcH] = buildCommonTrailingStrategy(b, variantH, strategy);

  // Step 6. The staged strategy has a post-bufferization vector distribution
  // with rank-reduction. The vector distribution occurs on multiple warps and
  // is itself internally staged in 2 steps.
  assert(strategy.getNumThreadsXInBlock() % kCudaWarpSize == 0 &&
         "strategy requires full warps");
  int64_t numWarpsToUse = strategy.getNumThreadsXInBlock() / kCudaWarpSize;
  int64_t bitWidth = strategy.captures.reductionOutputElementalTypeBitWidth;
  numWarpsToUse = adjustNumberOfWarpsForBlockShuffle(numWarpsToUse, bitWidth);
  buildDistributeVectors(b, variantH2, funcH, numWarpsToUse * kCudaWarpSize);
}

/// The configuration below has been determined empirically by performing a
/// manual tradeoff between problem size, amount of parallelism and vector size
/// on a particular NVIDIA RTX2080Ti 12GB card.
/// This is a coarse tradeoff that should generally give reasonably good results
/// but that begs to be complemented by hardcoded known good configurations and
/// ultimately a database and/or a random forest compression of configurations
/// with guaranteed performance.
// TODO: Lift some of the strategy sizing logic as hints and/or heuristics to
// also work properly in the dynamic case.
// TODO: Support more HW configs and make it more pluggable.
ReductionConfig mlir::iree_compiler::gpu::getStagedReductionConfig(
    const transform_ext::MatchedReductionCaptures &captures) {
  int64_t bitWidth = captures.reductionOutputElementalTypeBitWidth;
  int64_t vectorSize = scaleUpByBitWidth(4, bitWidth);
  int64_t maxNumThreads = 8 * kCudaWarpSize;
  // No adjustments in the dynamic case, we need extra information to make a
  // good decision.
  int64_t redSize = captures.reductionOpSizes.back();
  if (ShapedType::isDynamic(redSize))
    return ReductionConfig{maxNumThreads, vectorSize};
  // Scale down to smaller sizes (4, 8, 16)-warps.
  if (scaleUpByBitWidth(redSize, bitWidth) <= 4 * kCudaWarpSize) {
    vectorSize = scaleUpByBitWidth(1, bitWidth);
    maxNumThreads = 4 * kCudaWarpSize;
  } else if (scaleUpByBitWidth(redSize, bitWidth) <= 8 * kCudaWarpSize) {
    vectorSize = scaleUpByBitWidth(2, bitWidth);
    maxNumThreads = 4 * kCudaWarpSize;
  } else if (scaleUpByBitWidth(redSize, bitWidth) <= 8 * 2 * kCudaWarpSize) {
    vectorSize = scaleUpByBitWidth(4, bitWidth);
    maxNumThreads = 4 * kCudaWarpSize;
  }
  // Scale up to larger sizes (32, 64, 128+)-warps, using vector-4.
  if (!captures.trailingOpSizes.empty()) {
    if (scaleUpByBitWidth(redSize, bitWidth) >= 128 * 4 * kCudaWarpSize) {
      vectorSize = scaleUpByBitWidth(4, bitWidth);
      maxNumThreads = 32 * kCudaWarpSize;
    } else if (scaleUpByBitWidth(redSize, bitWidth) >= 64 * 4 * kCudaWarpSize) {
      vectorSize = scaleUpByBitWidth(4, bitWidth);
      maxNumThreads = 16 * kCudaWarpSize;
    } else if (scaleUpByBitWidth(redSize, bitWidth) >= 32 * 4 * kCudaWarpSize) {
      vectorSize = scaleUpByBitWidth(4, bitWidth);
      maxNumThreads = 8 * kCudaWarpSize;
    } else if (scaleUpByBitWidth(redSize, bitWidth) >= 16 * 4 * kCudaWarpSize) {
      vectorSize = scaleUpByBitWidth(4, bitWidth);
      maxNumThreads = 4 * kCudaWarpSize;
    }
  }
  return ReductionConfig{maxNumThreads, vectorSize};
}

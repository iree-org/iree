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

using iree_compiler::buildTileReductionUsingScfForeach;
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
    const transform_ext::MatchedReductionCaptures &captures,
    const ReductionConfig &reductionConfig) {
  StagedReductionStrategy strategy(context, captures);
  strategy.configure(reductionConfig);
  LLVM_DEBUG(DBGS() << "use GPU staged reduction strategy\n");
  return strategy;
}

void mlir::iree_compiler::gpu::StagedReductionStrategy::configure(
    const ReductionConfig &reductionConfig) {
  int64_t maxNumThreadsToUse = reductionConfig.maxNumThreads;
  int64_t maxVectorSize = reductionConfig.vectorSize;
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
  auto threadX = mlir::gpu::GPUThreadMappingAttr::get(b.getContext(),
                                                      mlir::gpu::Threads::DimX);
  auto [blockParallelForeachThreadOp, blockParallelFillH, blockCombinerOpH] =
      buildTileReductionUsingScfForeach(
          b, gridReductionH, strategy.captures.reductionRank,
          strategy.getNumThreadsXInBlock(), strategy.getVectorSize(), threadX);

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

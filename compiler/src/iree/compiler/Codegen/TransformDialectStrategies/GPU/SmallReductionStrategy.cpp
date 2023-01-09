// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/TransformDialectStrategies/GPU/SmallReductionStrategy.h"

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
using iree_compiler::gpu::SmallReductionStrategy;

FailureOr<SmallReductionStrategy>
mlir::iree_compiler::gpu::SmallReductionStrategy::create(
    MLIRContext *context,
    const transform_ext::MatchedReductionCaptures &captures) {
  ReductionConfig gpuReductionConfig = getSmallReductionConfig(captures);
  SmallReductionStrategy strategy(context, captures,
                                  gpuReductionConfig.maxNumThreads);
  if (!strategy.isProfitable()) return failure();
  LLVM_DEBUG(DBGS() << "use small reduction strategy\n");
  return strategy;
}

void mlir::iree_compiler::gpu::SmallReductionStrategy::compute(
    int64_t maxNumThreadsToUse, bool hasTrailingElementwise) {
  assert(maxNumThreadsToUse > 0 && "maxNumThreadsToUse must be > 0");
  assert(maxNumThreadsToUse >= kCudaWarpSize && "not even a warp?");

  // Block-level
  // ===========
  // TODO: capture more dims than just the most minor parallel and have a more
  // powerful `maybeDivisor` evaluation.
  int64_t mostMinorParallelDimensionSize =
      ArrayRef<int64_t>(captures.reductionOpSizes).drop_back().back();
  FailureOr<int64_t> maybeDivisor = maxDivisorOfValueBelowLimit(
      mostMinorParallelDimensionSize, maxNumThreadsToUse);

  // Trailing elementwise unaligned tiling created bounded local buffers that
  // are dynamic. Attempting to bound them in Common/PadDynamicAlloc.cpp results
  // in a crash in the associated upstream util.
  // TODO: Capture static parallel dimensions and allow if workgroupTileSizes
  // divides the parallel dimension evenly.
  // TODO: More generally fix PadDynamicAlloc and the associated upstream util.
  if (failed(maybeDivisor) && hasTrailingElementwise) return;

  // Dynamic reductions are never supported by default because we can never
  // know offhand whether we are in a small-reduction regime mode. Since this
  // mode does not coalesce reads, perf will suffer catastrophically on larger
  // runtime reduction.
  // TODO: explicit hint from above that we really want to do that.
  // TODO: evolve towards expressing this constraint with a perf-directed
  // matcher that composes with the existing structural matchers.
  int64_t reductionDimensionSize = captures.reductionOpSizes.back();
  if (ShapedType::isDynamic(reductionDimensionSize)) return;

  // Otherwise, still only support the small cases for now and fall back to
  // other strategies otherwise.
  // TODO: evolve towards expressing this constraint with a perf-directed
  // matcher that composes with the existing structural matchers.
  if (reductionDimensionSize >= 2 * kCudaWarpSize) return;

  // If the captured dimension has no satisfactory divisor, just tile the last
  // parallel dimension by 2 * kCudaWarpSize.
  int64_t numParallelLoops = captures.reductionRank - 1;
  workgroupTileSizes.append(numParallelLoops, 1);
  workgroupTileSizes.back() =
      hasTrailingElementwise
          ? *maybeDivisor
          : std::min((int64_t)maxNumThreadsToUse, (int64_t)(2 * kCudaWarpSize));

  // Thread-level
  // ============
  // Just running sequentially on each thread and relying on cache for
  // locality.
  // TODO: evolve towards expressing constraints with perf-directed matchers.
  profitable = true;
}

static void buildSmallReductionStrategyThreadDistribution(
    ImplicitLocOpBuilder &b, Value maybeLeadingH, Value fillH, Value reductionH,
    Value maybeTrailingH, const AbstractReductionStrategy &strategy) {
  auto [fusionTargetH, fusionGroupH] =
      iree_compiler::buildSelectFirstNonEmpty(b, maybeTrailingH, reductionH);
  ArrayRef<Attribute> allThreadsRef(strategy.allThreadAttrs);
  iree_compiler::TileToForeachThreadAndFuseAndDistributeResult tileResult =
      iree_compiler::buildTileFuseDistToForeachThreadWithNumThreads(
          /*builder=*/b,
          /*rootH=*/fusionTargetH,
          /*opsToFuseH=*/fusionGroupH,
          /*numThreads=*/
          getAsOpFoldResult(b.getI64ArrayAttr(strategy.workgroupTileSizes)),
          /*threadDimMapping=*/
          b.getArrayAttr(
              allThreadsRef.take_front(strategy.captures.reductionRank - 1)));
  fillH = b.create<FuseIntoContainingOp>(fillH, tileResult.foreachThreadH);
  maybeLeadingH =
      b.create<FuseIntoContainingOp>(maybeLeadingH, tileResult.foreachThreadH);

  // 1. Scalarize all ops to ensure vectorization.
  auto pdlOperation = pdl::OperationType::get(b.getContext());
  fillH = b.create<ScalarizeOp>(pdlOperation, fillH);
  maybeLeadingH = b.create<ScalarizeOp>(pdlOperation, maybeLeadingH);
  Value tiledH = b.create<ScalarizeOp>(pdlOperation, tileResult.tiledOpH);
  Value fusedH = b.create<ScalarizeOp>(
      pdlOperation, tileResult.resultingFusedOpsHandles.front());
  auto [blockReductionH, maybeBlockTrailingH] =
      iree_compiler::buildSelectFirstNonEmpty(b, fusedH, tiledH);

  // 2. Apply the 1d splitting strategy to the reduction part while specifying
  // a single thread. This triggers the splitting but not the thread mapping
  // part.
  build1DSplittingStrategyWithOptionalThreadMapping(
      b, blockReductionH, strategy.captures.reductionRank,
      strategy.captures.reductionOpSizes,
      /*numThreads=*/1);

  // 3. apply the 1d splitting strategy to the trailing elementwise.
  build1DSplittingStrategyWithOptionalThreadMapping(
      b, maybeBlockTrailingH, strategy.captures.maybeTrailingRank,
      strategy.captures.trailingOpSizes,
      strategy.getNumThreadsInBlock().back());
}

void mlir::iree_compiler::gpu::buildGpuSmallReductionStrategy(
    ImplicitLocOpBuilder &b, Value variantH,
    const SmallReductionStrategy &strategy) {
  // Step 1. Call the matcher. Note that this is the same matcher as used to
  // trigger this compilation path, so it must always apply.
  b.create<RegisterMatchCallbacksOp>();
  auto [maybeLeadingH, fillH, reductionH, maybeTrailingH] =
      unpackRegisteredMatchCallback<4>(
          b, "reduction", transform::FailurePropagationMode::Propagate,
          variantH);

  // Step 2. Apply block-level part of the strategy, keeps everything fused.
  auto [maybeLeadingHBlock, gridFillH, gridReductionH,
        maybeTiledTrailingHBlock] =
      buildReductionStrategyBlockDistribution(
          b, maybeLeadingH, fillH, reductionH, maybeTrailingH, strategy);

  // Step 3. Apply thread-level part of the strategy, keeps everything fused.
  buildSmallReductionStrategyThreadDistribution(
      b, maybeLeadingHBlock, gridFillH, gridReductionH,
      maybeTiledTrailingHBlock, strategy);

  // Step 4-5. Common trailing steps.
  buildCommonTrailingStrategy(b, variantH, strategy);
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
ReductionConfig mlir::iree_compiler::gpu::getSmallReductionConfig(
    const transform_ext::MatchedReductionCaptures &captures) {
  int64_t maxNumThreads = 4 * kCudaWarpSize;
  return ReductionConfig{maxNumThreads, 0};
}

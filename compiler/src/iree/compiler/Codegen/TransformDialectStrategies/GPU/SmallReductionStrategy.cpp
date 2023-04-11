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
using iree_compiler::IREE::transform_dialect::ForallToWorkgroupOp;
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

SmallReductionStrategy mlir::iree_compiler::gpu::SmallReductionStrategy::create(
    MLIRContext *context,
    const transform_ext::MatchedReductionCaptures &captures,
    const ReductionConfig &reductionConfig) {
  SmallReductionStrategy strategy(context, captures);
  strategy.configure(reductionConfig);
  LLVM_DEBUG(DBGS() << "use GPU small reduction strategy\n");
  return strategy;
}

void mlir::iree_compiler::gpu::SmallReductionStrategy::configure(
    const ReductionConfig &reductionConfig) {
  int64_t maxNumThreadsToUse = reductionConfig.maxNumThreads;
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
  // TODO: More generally fix PadDynamicAlloc and the associated upstream util.
  bool hasTrailingElementwise = (captures.maybeTrailingRank > 0);
  if (failed(maybeDivisor) && hasTrailingElementwise) maybeDivisor = 1;

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
}

static void buildSmallReductionStrategyThreadDistribution(
    ImplicitLocOpBuilder &b, Value variantH, Value maybeLeadingH, Value fillH,
    Value reductionH, Value maybeTrailingH,
    const AbstractReductionStrategy &strategy) {
  auto [fusionTargetH, fusionGroupH] =
      iree_compiler::buildSelectFirstNonEmpty(b, maybeTrailingH, reductionH);
  ArrayRef<Attribute> allThreadsRef(strategy.allThreadAttrs);
  iree_compiler::TileToForallAndFuseAndDistributeResult tileResult =
      iree_compiler::buildTileFuseDistToForallWithNumThreads(
          /*builder=*/b,
          /*isolatedParentOpH=*/variantH,
          /*rootH=*/fusionTargetH,
          /*opsToFuseH=*/fusionGroupH,
          /*numThreads=*/
          getAsOpFoldResult(b.getI64ArrayAttr(strategy.workgroupTileSizes)),
          /*threadDimMapping=*/
          b.getArrayAttr(
              allThreadsRef.take_front(strategy.captures.reductionRank - 1)));
  fillH = b.create<FuseIntoContainingOp>(fillH, tileResult.forallH);
  maybeLeadingH =
      b.create<FuseIntoContainingOp>(maybeLeadingH, tileResult.forallH);

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
      /*b=*/b,
      /*isolatedParentOpH=*/variantH,
      /*opH=*/blockReductionH,
      /*rank=*/strategy.captures.reductionRank,
      // TODO: capture and generalize mostMinorDim.
      /*mostMinorDim=*/strategy.captures.reductionRank - 1,
      /*opSizes=*/strategy.captures.reductionOpSizes,
      /*numThreads=*/1);

  // 3. apply the 1d splitting strategy to the trailing elementwise.
  build1DSplittingStrategyWithOptionalThreadMapping(
      /*b=*/b,
      /*isolatedParentOpH=*/variantH,
      /*opH=*/maybeBlockTrailingH,
      /*rank=*/strategy.captures.maybeTrailingRank,
      // TODO: capture and generalize mostMinorDim.
      /*mostMinorDim=*/strategy.captures.maybeTrailingRank - 1,
      /*opSizes=*/strategy.captures.trailingOpSizes,
      /*numThreads=*/strategy.getNumThreadsXInBlock(),
      /*mappingAttr=*/strategy.allThreadAttrs.front());
}

void mlir::iree_compiler::gpu::buildSmallReductionStrategy(
    ImplicitLocOpBuilder &b, Value variantH,
    const SmallReductionStrategy &strategy) {
  // Step 1. Apply block-level part of the strategy, keeps everything fused.
  auto [maybeLeadingHBlock, gridFillH, gridReductionH, maybeTiledTrailingHBlock,
        forall] =
      buildReductionStrategyBlockDistribution(b, variantH, strategy);

  // Step 2. Apply thread-level part of the strategy, keeps everything fused.
  buildSmallReductionStrategyThreadDistribution(
      b, variantH, maybeLeadingHBlock, gridFillH, gridReductionH,
      maybeTiledTrailingHBlock, strategy);

  // Step 3-4. Common trailing steps.
  buildCommonTrailingStrategy(b, variantH, strategy);
}

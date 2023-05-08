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
#include "mlir/Dialect/SCF/IR/SCF.h"
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
using iree_compiler::IREE::transform_dialect::ShareForallOperandsOp;
using iree_compiler::IREE::transform_dialect::VectorToWarpExecuteOnLane0Op;
using iree_compiler::IREE::transform_dialect::VectorWarpDistributionOp;
using transform::FuseIntoContainingOp;
using transform::MatchOp;
using transform::ScalarizeOp;
using transform::SequenceOp;
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

static Value shareForeachArgument(ImplicitLocOpBuilder &b, Value Forall,
                                  ArrayRef<int64_t> indices) {
  auto foreachType = transform::OperationType::get(
      b.getContext(), scf::ForallOp::getOperationName());
  Forall = b.create<transform::CastOp>(foreachType, Forall);
  return b
      .create<iree_compiler::IREE::transform_dialect::ShareForallOperandsOp>(
          foreachType, Forall, indices);
}

static void buildStagedReductionStrategyThreadLevel(
    ImplicitLocOpBuilder &b, Value isolatedParentOpH, Value gridReductionH,
    Value gridFillH, Value maybeTiledLeadingH, Value maybeTiledTrailingH,
    const StagedReductionStrategy &strategy) {
  // Map the potential maybeTiledLeadingH.
  // TODO: Consider fusing leading elementwise into threads.
  if (strategy.captures.maybeLeadingRank > 0) {
    int64_t vectorSize =
        kCudaMaxVectorLoadBitWidth /
        strategy.captures.maybeLeadingOutputElementalTypeBitWidth;
    assert((vectorSize & (vectorSize - 1)) == 0 && "size must be power of 2");
    build1DSplittingStrategyWithOptionalThreadMapping(
        /*b=*/b,
        /*isolatedParentOpH=*/isolatedParentOpH,
        /*opH=*/maybeTiledLeadingH,
        /*rank=*/strategy.captures.maybeLeadingRank,
        // TODO: capture and generalize mostMinorDim.
        /*mostMinorDim=*/strategy.captures.maybeLeadingRank - 1,
        /*opSizes=*/strategy.captures.leadingOpSizes,
        /*numThreads=*/strategy.getNumThreadsXInBlock(),
        /*mappingAttr=*/strategy.allThreadAttrs.front(),
        /*maxVectorSize=*/vectorSize);
  }

  // Staged reduction step 1: break gridReductionH apart.
  auto [blockParallelForallOp, blockParallelFillH, blockCombinerOpH] =
      buildTileReductionUsingScfForeach(
          /*b=*/b,
          /*isolatedParentOpH=*/isolatedParentOpH,
          /*reductionH=*/gridReductionH,
          /*reductionRank=*/strategy.captures.reductionRank,
          /*tileSize=*/strategy.getNumThreadsXInBlock(),
          /*reductionVectorSize=*/strategy.getVectorSize(),
          /*mappingAttr=*/strategy.allThreadAttrs[0]);

  // Staged reduction step 2: multi-warp shuffle reduce.
  // Map the combiner reduction to one thread along y. Mapping this part along
  // y only will trigger the insertion of an `scf.if (threadIdx.x == 0)`
  // predicate after `scf.forall` is lowered.
  // This predicate allows further vector distribution to kick in.
  Value root = blockCombinerOpH;
  SmallVector<Value> opsToFuse = {gridFillH};

  // By the properties matching, we know the optional trailing op takes the
  // result of the reduction as an input argument.
  // It necessarily follows that maybeTrailingRank >= reductionRank - 1.
  // When maybeTrailingRank == reductionRank - 1, by the properties of the
  // transformations we have applied until now, we know that the elementwise is
  // a simple scalar operation and it can be fused in the producing reduction
  // without creating recomputations.
  // TODO: Some `transform.assert` op that the shape of the op is indeed 1s only
  // as a safety measure.
  // TODO: More composable transform strategy parts require more matching after
  // part of the strategy has been applied. See the discussion in #11951 for
  // more context.
  if (strategy.captures.maybeTrailingRank ==
      strategy.captures.reductionRank - 1) {
    root = maybeTiledTrailingH;
    opsToFuse.push_back(blockCombinerOpH);
  }
  iree_compiler::buildTileFuseDistToForallWithTileSizes(
      /*b=*/b,
      /*isolatedParentOpH=*/isolatedParentOpH,
      /*rootH=*/root,
      /*opsToFuse=*/opsToFuse,
      /*tileSizes=*/getAsOpFoldResult(b.getI64ArrayAttr({1})),
      /*mappingAttr=*/b.getArrayAttr(strategy.allThreadAttrs[1]));

  // Map the potential maybeTiledTrailingH if it hasn't been fused with the
  // reduction.
  if (root != maybeTiledTrailingH && strategy.captures.maybeTrailingRank > 0) {
    int64_t vectorSize =
        iree_compiler::gpu::kCudaMaxVectorLoadBitWidth /
        strategy.captures.maybeTrailingOutputElementalTypeBitWidth;
    build1DSplittingStrategyWithOptionalThreadMapping(
        /*b=*/b,
        /*isolatedParentOpH=*/isolatedParentOpH,
        /*opH=*/maybeTiledTrailingH,
        /*rank=*/strategy.captures.maybeTrailingRank,
        // TODO: capture and generalize mostMinorDim.
        /*mostMinorDim=*/strategy.captures.maybeTrailingRank - 1,
        /*opSizes=*/strategy.captures.trailingOpSizes,
        /*numThreads=*/strategy.getNumThreadsXInBlock(),
        /*mappingAttr=*/strategy.allThreadAttrs.front(),
        /*maxVectorSize=*/vectorSize);
  }
}

/// Builds the transform IR tiling reductions for CUDA targets. Supports
/// reductions in the last dimension, with optional leading and trailing
/// elementwise operations.
void mlir::iree_compiler::gpu::buildStagedReductionStrategy(
    ImplicitLocOpBuilder &b, Value variantH,
    const StagedReductionStrategy &strategy) {
  // Step 1. Match and tile to introduce the top-level scf.forall for
  // the block/workgroup level. Keep everything fused.
  auto [maybeLeadingHBlock, gridFillH, gridReductionH, maybeTiledTrailingHBlock,
        commonEnclosingForallH] =
      buildReductionStrategyBlockDistribution(b, variantH, strategy);

  // Step 2. Split the reduction and tile the pieces to ensure vector
  // load/stores and mapping to a single warp with shuffles.
  buildStagedReductionStrategyThreadLevel(
      b,
      /*isolatedParentOpH=*/variantH, gridReductionH, gridFillH,
      maybeLeadingHBlock, maybeTiledTrailingHBlock, strategy);

  // Step 3. Make sure we don't create allocation by sharing forall
  // output. This amounts to injecting user-defined static information that each
  // thread accesses only a private slice. This needs to be added late, once we
  // don't need handles anymore, because contained handles are currently always
  // invalidated, even when modified inplace.
  // TODO: Relax nested invalidation for transforms that only move or modify
  // contained ops inplace.
  shareForeachArgument(b, commonEnclosingForallH, ArrayRef<int64_t>({0}));

  // Step 4-5. Common trailing steps.
  auto [variantH2, funcH] = buildCommonTrailingStrategy(b, variantH, strategy);

  // Step 6. The staged strategy has a post-bufferization vector distribution
  // with rank-reduction. The vector distribution occurs on multiple warps and
  // is itself internally staged in 2 stages.
  assert(strategy.getNumThreadsXInBlock() % kCudaWarpSize == 0 &&
         "strategy requires full warps");
  int64_t numWarpsToUse = strategy.getNumThreadsXInBlock() / kCudaWarpSize;
  // Distribute the reduction on all the threads of the group. This allows us
  // to have the same data layout for the partial reduction and the merge and
  // therefore we can optimize away the temporary memory usage.
  buildDistributeVectors(b, variantH2, funcH, numWarpsToUse * kCudaWarpSize);

  // Step 7. Apply clean up of memory operations.
  funcH = b.create<MatchOp>(variantH2, func::FuncOp::getOperationName());
  iree_compiler::buildMemoryOptimizations(b, funcH);
}

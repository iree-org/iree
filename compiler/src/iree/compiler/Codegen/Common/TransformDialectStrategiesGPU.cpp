// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/TransformDialectStrategiesGPU.h"

#include "iree-dialects/Dialect/LinalgTransform/StructuredTransformOpsExt.h"
#include "iree-dialects/Transforms/TransformMatchers.h"
#include "iree/compiler/Codegen/Common/TransformDialectStrategies.h"
#include "iree/compiler/Codegen/Common/TransformExtensions/CommonExtensions.h"
#include "iree/compiler/Codegen/LLVMGPU/TransformExtensions/LLVMGPUExtensions.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
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
using transform::SequenceOp;
using transform_ext::MatchCallbackOp;
using transform_ext::RegisterMatchCallbacksOp;
using transform_ext::StructuredOpMatcher;
using transform_ext::TakeFirstOp;

/// Matches a C++ callback previously registered under `callbackName` and
/// taking arguments `args`.
/// Unpacks a number of handles `N` (asserts there are exactly `N` matched ops
/// but this could be relaxed if needed).
/// Returns the tuple of handles.
template <int N, typename... MatchingArgs>
auto unpackRegisteredMatchCallback(ImplicitLocOpBuilder &b,
                                   StringRef callbackName,
                                   MatchingArgs... args) {
  SmallVector<Type> matchedTypes(N, pdl::OperationType::get(b.getContext()));
  auto matchOp = b.create<MatchCallbackOp>(
      matchedTypes, callbackName, std::forward<decltype(args)>(args)...);
  assert(matchOp->getNumResults() == N && "Unexpected number of results");
  std::array<Value, N> a;
  for (int64_t i = 0; i < N; ++i) a[i] = matchOp->getResult(i);
  return std::tuple_cat(a);
}

/// Post-bufferization mapping to blocks and threads.
/// Takes a handle to a func.func and returns an updated handle to a
/// func.func.
Value mlir::iree_compiler::buildMapToBlockAndThreads(
    ImplicitLocOpBuilder &b, Value funcH, ArrayRef<int64_t> blockSize) {
  funcH = b.create<ForeachThreadToWorkgroupOp>(funcH);
  return b.create<MapNestedForeachThreadToGpuThreadsOp>(funcH, blockSize);
}

/// Post-bufferization vector distribution with rank-reduction.
/// Takes a handle to a func.func and returns an updated handle to a
/// func.func.
Value mlir::iree_compiler::buildDistributeVectors(ImplicitLocOpBuilder &b,
                                                  Value variantH, Value funcH,
                                                  int64_t warpSize) {
  ApplyPatternsOpPatterns patterns;
  patterns.foldMemrefAliases = true;
  patterns.rankReducing = true;
  funcH = b.create<ApplyPatternsOp>(funcH, patterns);
  Value ifH = b.create<MatchOp>(funcH, scf::IfOp::getOperationName());
  // Locally suppress failures for this op only because it doesn't cover the
  // `threadIdx.x == 0 && threadIdx.y == 0` case at the moment.
  auto sequence = b.create<SequenceOp>(
      TypeRange(), transform::FailurePropagationMode::Suppress, variantH);
  {
    OpBuilder::InsertionGuard guard(b);
    b.createBlock(&sequence.getBody(), sequence.getBody().begin(),
                  pdl::OperationType::get(b.getContext()), b.getLoc());
    ifH = b.create<VectorToWarpExecuteOnLane0Op>(ifH, warpSize);
    b.create<transform::YieldOp>();
  }
  b.create<VectorWarpDistributionOp>(funcH);
  return funcH;
}

//===----------------------------------------------------------------------===//
// Higher-level problem-specific strategy creation APIs, these should favor
// user-friendliness.
//===----------------------------------------------------------------------===//

namespace {

/// Encodes a strategy for a 1-d reduction mapped to a block.
///
/// This happens in a staged fashion to encode good tradeoffs between amount
/// of parallelism, occupancy and granularity of the load/store operations.
/// The tradeoff is controlled at a distance by specifying a
/// `maxNumThreadsToUse` upper bound.
///
/// Bottom-up perspective:
/// ======================
/// Stage 3: the warp shuffle step is normalized to run on a single warp using
/// a vector<warp_shufle_vector_size x T> element. The size of this vector is
/// controlled by a `warpShuffleSize` parameter passed to the constructor which
/// must be a 1,2 or 4-multiple of the machine warp size.
///
/// Stage 2: the second stage of the reduction is normalized to reduce from a
/// "k-warps" abstraction (k determined in Step 1) to a single `warpShuffleSize`
/// that can be reduced unambiguously well on a single warp during Stage 3.
/// This second/ stage is optional and only occurs when k > 1.
///
/// Stage 1: the first stage of the reduction is normalized to run on "k-warps"
/// of maximal vector size for both the hardware and the problem sizes.
/// The overprovisioning to "k-warps" allows multiple warps to run in parallel.
/// The `reductionTileSizeStage1` is this "k-warps" quantity and is also the
/// number of threads (i.e. blockDim.x) used to parallelize the problem.
/// This also results in `reductionTileSizeStage1` live values that are
/// allocated in shared memory and creates a tradeoff between parallelism and
/// occupancy.
/// The normalization guarantees that whatever the problem size P, we reduce
/// from `tensor<P x T>` to `tensor<reductionTileSizeStage1 x T>` by using the
/// largest possible `vector.transfer` operations. The vector size is chosen as
/// follows: when the `reductionDimensionSize` is a multiple of 4, choose 4;
/// otherwise try with 2; otherwise just use 1.
///
// TODO: Support various elemental types.
// TODO: Split to ensure 4 on most of the problem and use a 1-epilogue.
class ReductionStrategy3StageThreadDistribution {
 public:
  ReductionStrategy3StageThreadDistribution() = default;
  ReductionStrategy3StageThreadDistribution(int64_t reductionDimensionSize,
                                            int64_t maxNumThreadsToUse,
                                            int64_t warpShuffleSize) {
    compute(reductionDimensionSize, maxNumThreadsToUse, warpShuffleSize);
  }
  ReductionStrategy3StageThreadDistribution(
      const ReductionStrategy3StageThreadDistribution &) = default;

  ReductionStrategy3StageThreadDistribution &operator=(
      const ReductionStrategy3StageThreadDistribution &) = default;

  int64_t getVectorSizeStage1() { return vectorSizeStage1; }

  int64_t getNumThreadsXInBlock() { return reductionTileSizeStage1; }
  int64_t getNumThreadsYInBlock() { return 1; }
  int64_t getNumThreadsZInBlock() { return 1; }
  std::array<int64_t, 3> getNumThreadsInBlock() {
    return {getNumThreadsXInBlock(), getNumThreadsYInBlock(),
            getNumThreadsZInBlock()};
  }

  bool hasStage2() { return reductionTileSizeStage2.has_value(); }
  int64_t getWarpShuffleSize() { return reductionTileSizeStage2.value(); }
  int64_t getVectorSizeStage2() { return vectorSizeStage2.value(); }

 private:
  /// Maximal vector size (among {1, 2, 4}) that divides the
  /// `reductionDimensionSize` and is used for vector transfers in Stage 1.
  int64_t vectorSizeStage1;
  /// Maximal "k-warp" size within the limits of the `maxNumThreadsToUse` and
  /// `reductionDimensionSize` parameters.
  /// This is also the blockDim.x of the kernel.
  int64_t reductionTileSizeStage1;
  /// Maximal vector size allowing to reduce from "k-warp" to single warp: when
  /// `k` is a multiple of 4, choose 4; otherwise try with 2; otherwise just
  /// use 1.
  std::optional<int64_t> vectorSizeStage2;
  /// `reductionTileSizeStage2` is exactly set to `warpShuffleSize` passed to
  /// the constructor, only when reductionTileSizeStage1 > warpShuffleSize (i.e.
  /// k > 1).
  std::optional<int64_t> reductionTileSizeStage2;

  /// Compute the staged strategy based on the `reductionDimensionSize`, the
  /// `maxNumThreadsToUse` and the `warpShuffleSize`.
  /// The latter 2 numbers control the tradeoff between parallelism and shared
  /// memory consumption.
  // TODO: Characterize shared memory consumption and limit for good occupancy.
  // TODO: Support various elemental types.
  void compute(int64_t reductionDimensionSize, int64_t maxNumThreadsToUse,
               int64_t warpShuffleSize);
};

void ReductionStrategy3StageThreadDistribution::compute(
    int64_t reductionDimensionSize, int64_t maxNumThreadsToUse,
    int64_t warpShuffleSize) {
  assert(warpShuffleSize > 0 && "warpShuffleSize must > 0");
  assert(warpShuffleSize % iree_compiler::kCudaWarpSize == 0 &&
         "warpShuffleSize must be a multiple of warpSize");
  assert(warpShuffleSize <= 4 * iree_compiler::kCudaWarpSize &&
         "must be smaller or equal to 4 * warp_size");

  // Stage 1.
  // Maximal vector size that divides the problem size.
  // TODO: Split to ensure 4 on most of the problem and use a 1-epilogue.
  if (reductionDimensionSize > 0 && reductionDimensionSize % 4 == 0)
    vectorSizeStage1 = 4;
  else if (reductionDimensionSize > 0 && reductionDimensionSize % 2 == 0)
    vectorSizeStage1 = 2;
  else
    vectorSizeStage1 = 1;

  // Tile reduction to the maximal multiple `warpShuffleSize` allowed.
  // This locally reduces the large unknown reduction into a guaranteed
  // multiple of `warpShuffleSize`.
  if (reductionDimensionSize > 0) {
    reductionTileSizeStage1 = std::min(
        iree_compiler::nextMultipleOf(reductionDimensionSize / vectorSizeStage1,
                                      warpShuffleSize),
        iree_compiler::previousMultipleOf(maxNumThreadsToUse, warpShuffleSize));
  } else {
    reductionTileSizeStage1 =
        iree_compiler::previousMultipleOf(maxNumThreadsToUse, warpShuffleSize);
  }
  // Stage 2 is only needed if `reductionTileSizeStage1` consists of multiple
  // `warpShuffleSize`; otherwise, we just skip this step.
  if (reductionTileSizeStage1 > warpShuffleSize) {
    // Tile reduction exactly to `warpShuffleSize` which will be used in the 3rd
    // stage to distribute to warp shuffles.
    reductionTileSizeStage2 = warpShuffleSize;
    // The vector size we use depends on the number of `warpShuffleSize`s in
    // `reductionTileSizeStage1`.
    int64_t factor = reductionTileSizeStage1 / warpShuffleSize;
    if (factor % 4 == 0)
      vectorSizeStage2 = 4;
    else if (factor % 2 == 0)
      vectorSizeStage2 = 2;
    else
      vectorSizeStage2 = 1;
  }
}

/// Structure to hold the parameters related to GPU reduction strategy.
struct GPUReductionStrategyInfos {
  explicit GPUReductionStrategyInfos(
      MLIRContext *context, transform_ext::MatchedReductionCaptures captures)
      : context(context),
        reductionRank(captures.reductionRank),
        reductionDimensionSize(captures.reductionDimensionSize),
        maybeLeadingRank(captures.maybeLeadingRank),
        maybeTrailingRank(captures.maybeTrailingRank) {
    auto blockX =
        mlir::gpu::GPUBlockMappingAttr::get(context, mlir::gpu::Blocks::DimX);
    auto blockY =
        mlir::gpu::GPUBlockMappingAttr::get(context, mlir::gpu::Blocks::DimY);
    auto blockZ =
        mlir::gpu::GPUBlockMappingAttr::get(context, mlir::gpu::Blocks::DimZ);
    allBlockAttrs = SmallVector<Attribute>{blockX, blockY, blockZ};
  }

  void computeThreadDistribution(int64_t maxNumThreads,
                                 int64_t warpShuffleSize) {
    threadDistribution3Stages =
        std::make_unique<ReductionStrategy3StageThreadDistribution>(
            reductionDimensionSize, maxNumThreads, warpShuffleSize);
  }

  /// Constructor quantities.
  MLIRContext *context;
  int64_t reductionRank;
  int64_t reductionDimensionSize;
  int64_t maybeLeadingRank;
  int64_t maybeTrailingRank;

  /// Derived quantities.
  std::unique_ptr<ReductionStrategy3StageThreadDistribution>
      threadDistribution3Stages;
  SmallVector<Attribute> allBlockAttrs;
  // Tile sizes for the workgroup / determines grid size.
  SmallVector<int64_t> workgroupTileSizes;
  // Launch bounds for the workgroups / block size.
  std::array<int64_t, 3> workgroupSize;
};
}  // namespace

static std::tuple<Value, Value, Value> createReductionStrategyBlockDistribution(
    ImplicitLocOpBuilder &b, Value maybeLeadingH, Value fillH, Value reductionH,
    Value maybeTrailingH, const GPUReductionStrategyInfos &infos) {
  auto pdlOperation = pdl::OperationType::get(b.getContext());
  auto fusionTargetSelector = b.create<TakeFirstOp>(
      pdlOperation, pdlOperation, ArrayRef<Value>{maybeTrailingH, reductionH});
  Value fusionTargetH = fusionTargetSelector.getFirst();
  Value fusionGroupH = fusionTargetSelector.getRest();
  ArrayRef<Attribute> allBlocksRef(infos.allBlockAttrs);
  iree_compiler::TileToForeachThreadAndFuseAndDistributeResult tileResult =
      iree_compiler::
          buildTileFuseDistToForeachThreadAndWorgroupCountWithTileSizes(
              /*builder=*/b,
              /*rootH=*/fusionTargetH,
              /*opsToFuseH=*/fusionGroupH,
              /*tileSizes=*/
              getAsOpFoldResult(b.getI64ArrayAttr(infos.workgroupTileSizes)),
              /*threadDimMapping=*/
              b.getArrayAttr(allBlocksRef.take_front(infos.reductionRank - 1)));
  fillH = b.create<FuseIntoContainingOp>(fillH, tileResult.foreachThreadH);
  if (infos.maybeLeadingRank > 0) {
    maybeLeadingH = b.create<FuseIntoContainingOp>(maybeLeadingH,
                                                   tileResult.foreachThreadH);
  }
  // Here, TakeFirstOp acts as a normalizer:
  //   1. if fusionTargetH is maybeTrailingH then getFirst() is reductionH and
  //      getRest() is maybeTrailingH.
  //   2. if fusionTargetH is reductionH then getFirst() is reductionH and
  //      getRest() is empty.
  auto gridReductionSelector = b.create<TakeFirstOp>(
      pdlOperation, pdlOperation,
      ValueRange{tileResult.resultingFusedOpsHandles.front(),
                 tileResult.tiledOpH});
  Value gridReductionH = gridReductionSelector.getFirst();
  maybeTrailingH = gridReductionSelector.getRest();
  return std::make_tuple(maybeLeadingH, gridReductionH, maybeTrailingH);
}

static std::tuple<Value, Value, Value>
createReductionStrategy3StageThreadDistributionStep(
    ImplicitLocOpBuilder &b, Value gridReductionH, int64_t reductionRank,
    int64_t reductionTileSizeStage, int64_t reductionVectorSize) {
  auto threadX = mlir::gpu::GPUThreadMappingAttr::get(b.getContext(),
                                                      mlir::gpu::Threads::DimX);
  // Split the reduction into a parallel and combiner part, then tile the
  // parallel part and map it to `reductionTileSizeStage` threads, each working
  // on `reductionVectorSize`.
  SmallVector<int64_t> leadingParallelDims(reductionRank - 1, 0);
  SmallVector<int64_t> numThreads = leadingParallelDims;
  numThreads.push_back(reductionTileSizeStage);
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
  // Fuse the fill and pointwise to privatize them.
  blockParallelFillH = b.create<FuseIntoContainingOp>(
      blockParallelFillH, blockParallelForeachThreadOp);
  return std::make_tuple(blockParallelForeachThreadOp, blockParallelFillH,
                         blockCombinerOpH);
}

static void createElementwiseStrategyThreadStep(ImplicitLocOpBuilder &b,
                                                Value elementwiseH,
                                                int64_t rank,
                                                int64_t numThreadsXInBlock) {
  assert(rank > 0 && "nonnegative rank expected");
  SmallVector<int64_t> trailingTileSizes(rank, 0);
  SmallVector<int64_t> scfForTileSizes = trailingTileSizes,
                       foreachTileSizes = trailingTileSizes;
  // The following assumes we only want to tile the most-minor dimension of the
  // trailing operation. This may be a completely wrong choice.
  // TODO: More robustness to permutations of most-minor dimensions.
  // TODO: Capture most minor size and compute tile size based on it.
  scfForTileSizes.back() = numThreadsXInBlock;
  foreachTileSizes.back() = numThreadsXInBlock;
  // TODO: Only tile by scf.for if we wish to normalize the tiling (i.e. if
  // the result has a lot more values than the number of threads in block).
  // This allows us to avoid uncoalesced accesses.
  // TODO: Refine elementwise strategy to allow vector<2/4>.
  // TODO: Consider splitting the elementwise strategy to ensure vector<2/4>.
  auto res = iree_compiler::buildTileFuseToScfFor(
      b, elementwiseH, {},
      getAsOpFoldResult(b.getI64ArrayAttr({scfForTileSizes})));
  auto threadX = mlir::gpu::GPUThreadMappingAttr::get(b.getContext(),
                                                      mlir::gpu::Threads::DimX);
  iree_compiler::buildTileFuseDistToForeachThreadWithNumThreads(
      b, res.tiledOpH, {},
      getAsOpFoldResult(b.getI64ArrayAttr(foreachTileSizes)),
      b.getArrayAttr({threadX}));
}

static void createReductionStrategy3StageThreadDistribution(
    ImplicitLocOpBuilder &b, Value gridReductionH, Value maybeTiledLeadingH,
    Value maybeTiledTrailingH, const GPUReductionStrategyInfos &infos) {
  // Map the potential maybeTiledLeadingH.
  // TODO: Consider fusing leading elementwise into threads.
  if (infos.maybeLeadingRank > 0) {
    createElementwiseStrategyThreadStep(
        b, maybeTiledLeadingH, infos.maybeLeadingRank,
        infos.threadDistribution3Stages->getNumThreadsXInBlock());
  }

  // Staged reduction step 1: break gridReductionH apart.
  auto [blockParallelForeachThreadOp, blockParallelFillH, blockCombinerOpH] =
      createReductionStrategy3StageThreadDistributionStep(
          b, gridReductionH, infos.reductionRank,
          infos.threadDistribution3Stages->getNumThreadsXInBlock(),
          infos.threadDistribution3Stages->getVectorSizeStage1());

  // Staged reduction step 2: break blockCombinerOpH apart.
  // Note, if necessary, we could have additional intermediate steps.
  Value warpParallelForeachThreadOp, warpParallelFillH, warpCombinerOpH;
  if (infos.threadDistribution3Stages->hasStage2()) {
    std::tie(warpParallelForeachThreadOp, warpParallelFillH, warpCombinerOpH) =
        createReductionStrategy3StageThreadDistributionStep(
            b, blockCombinerOpH, infos.reductionRank,
            infos.threadDistribution3Stages->getWarpShuffleSize(),
            infos.threadDistribution3Stages->getVectorSizeStage1());
  } else {
    warpCombinerOpH = blockCombinerOpH;
  }

  // Staged reduction step 3: break blockCombinerOpH apart.
  // Map the combiner reduction to one thread along y so it can be mapped
  // further via predication.
  auto threadY = mlir::gpu::GPUThreadMappingAttr::get(b.getContext(),
                                                      mlir::gpu::Threads::DimY);
  iree_compiler::buildTileFuseDistToForeachThreadWithTileSizes(
      b, warpCombinerOpH, {}, getAsOpFoldResult(b.getI64ArrayAttr({1})),
      b.getArrayAttr(threadY));

  // Map the potential maybeTiledTrailingH.
  if (infos.maybeTrailingRank > 0) {
    createElementwiseStrategyThreadStep(
        b, maybeTiledTrailingH, infos.maybeTrailingRank,
        infos.threadDistribution3Stages->getNumThreadsXInBlock());
  }
}

/// Builds the transform IR tiling reductions for CUDA targets. Supports
/// reductions in the last dimension, with optional leading and trailing
/// elementwise operations.
static void createReductionCudaStrategy(
    ImplicitLocOpBuilder &b, Value variantH,
    const GPUReductionStrategyInfos &infos) {
  // Step 1. Call the matcher. Note that this is the same matcher as used to
  // trigger this compilation path, so it must always apply.
  b.create<RegisterMatchCallbacksOp>();
  auto [maybeLeadingH, fillH, reductionH, maybeTrailingH] =
      unpackRegisteredMatchCallback<4>(
          b, "reduction", transform::FailurePropagationMode::Propagate,
          variantH);

  // Step 2. Use tiling to introduce a single-iteration loop mapped to a
  // single block/workgroup. Keep everything fused.
  auto [maybeLeadingHFused, gridReductionH, maybeTiledTrailingH] =
      createReductionStrategyBlockDistribution(
          b, maybeLeadingH, fillH, reductionH, maybeTrailingH, infos);
  maybeLeadingH = maybeLeadingHFused;

  // Step 3. Split the reduction and tile the pieces to ensure vector
  // load/stores and mapping to a single warp with shuffles.
  createReductionStrategy3StageThreadDistribution(
      b, gridReductionH, maybeLeadingH, maybeTiledTrailingH, infos);

  // Step 4. Bufferize and drop HAL decriptor from memref ops.
  Value funcH = b.create<MatchOp>(variantH, func::FuncOp::getOperationName());
  funcH = iree_compiler::buildVectorize(b, funcH);
  variantH = iree_compiler::buildBufferize(b, variantH,
                                           /*targetGpu=*/true);

  // Step 5. Post-bufferization mapping to blocks and threads.
  // Need to match again since bufferize invalidated all handles.
  // TODO: assumes a single func::FuncOp to transform, may need hardening.
  funcH = b.create<MatchOp>(variantH, func::FuncOp::getOperationName());
  funcH =
      iree_compiler::buildMapToBlockAndThreads(b, funcH, infos.workgroupSize);

  // Step 6. Post-bufferization vector distribution with rank-reduction.
  iree_compiler::buildDistributeVectors(b, variantH, funcH);
}

struct GPUReductionConfig {
  int64_t maxNumThreads;
  int64_t warpShuffleSize;
};

// TODO: Lift some of the strategy sizing logic as hints and/or heuristics to
// also work properly in the dynamic case.
// TODO: Support more HW configs and make it more pluggable.
static GPUReductionConfig getReductionConfigRTX2080Ti(
    const transform_ext::MatchedReductionCaptures &captures) {
  int64_t maxNumThreads = 8 * iree_compiler::kCudaWarpSize;
  int64_t warpShuffleSize = 2 * iree_compiler::kCudaWarpSize;
  if (captures.reductionDimensionSize > 0) {
    if (captures.reductionDimensionSize <= iree_compiler::kCudaWarpSize) {
      maxNumThreads = warpShuffleSize = iree_compiler::kCudaWarpSize;
    } else if (captures.reductionDimensionSize <=
               2 * iree_compiler::kCudaWarpSize) {
      maxNumThreads = 2 * iree_compiler::kCudaWarpSize;
      warpShuffleSize = iree_compiler::kCudaWarpSize;
    } else if (captures.reductionDimensionSize <=
               4 * iree_compiler::kCudaWarpSize) {
      maxNumThreads = 4 * iree_compiler::kCudaWarpSize;
      warpShuffleSize = 2 * iree_compiler::kCudaWarpSize;
    }
  }
  return GPUReductionConfig{maxNumThreads, warpShuffleSize};
}

static FailureOr<GPUReductionStrategyInfos> matchGPUReduction(
    linalg::LinalgOp op) {
  StructuredOpMatcher reduction, fill, leading, trailing;
  transform_ext::MatchedReductionCaptures captures;
  makeReductionMatcher(reduction, fill, leading, trailing, captures);
  if (!matchPattern(op, reduction)) return failure();

  GPUReductionStrategyInfos infos(op->getContext(), captures);
  GPUReductionConfig gpuReductionConfig = getReductionConfigRTX2080Ti(captures);
  infos.computeThreadDistribution(gpuReductionConfig.maxNumThreads,
                                  gpuReductionConfig.warpShuffleSize);

  // Tile all the parallel dimensions to 1 and create many blocks.
  // TODO: Better strategy for very small reductions requires tiling across
  // other dimensions than threadIdx.x.
  int64_t numParallelLoops = captures.reductionRank - 1;
  infos.workgroupTileSizes.append(numParallelLoops, 1);
  // Tile and distribute the reduction across `reductionTileSizeStage1`
  // threads.
  infos.workgroupSize = infos.threadDistribution3Stages->getNumThreadsInBlock();
  return infos;
}

LogicalResult iree_compiler::matchAndSetGPUReductionTransformStrategy(
    func::FuncOp entryPoint, linalg::LinalgOp op) {
  // 1. Match.
  FailureOr<GPUReductionStrategyInfos> maybeInfos = matchGPUReduction(op);
  if (failed(maybeInfos)) return failure();
  auto strategyBuilder = [&](ImplicitLocOpBuilder &b, Value variant) {
    return createReductionCudaStrategy(b, variant, *maybeInfos);
  };
  // 2. Add the strategy.
  createTransformRegion(entryPoint, strategyBuilder);
  return success();
}

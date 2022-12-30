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
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
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
/// Structure to hold the parameters related to GPU reduction strategy.
struct GPUReductionStrategy {
  GPUReductionStrategy(MLIRContext *context,
                       const transform_ext::MatchedReductionCaptures &captures)
      : context(context), captures(captures) {
    auto blockX =
        mlir::gpu::GPUBlockMappingAttr::get(context, mlir::gpu::Blocks::DimX);
    auto blockY =
        mlir::gpu::GPUBlockMappingAttr::get(context, mlir::gpu::Blocks::DimY);
    auto blockZ =
        mlir::gpu::GPUBlockMappingAttr::get(context, mlir::gpu::Blocks::DimZ);
    allBlockAttrs = SmallVector<Attribute>{blockX, blockY, blockZ};
    auto threadX =
        mlir::gpu::GPUThreadMappingAttr::get(context, mlir::gpu::Threads::DimX);
    auto threadY =
        mlir::gpu::GPUThreadMappingAttr::get(context, mlir::gpu::Threads::DimY);
    auto threadZ =
        mlir::gpu::GPUThreadMappingAttr::get(context, mlir::gpu::Threads::DimZ);
    allThreadAttrs = SmallVector<Attribute>{threadX, threadY, threadZ};
  }

  virtual ~GPUReductionStrategy() {}

  virtual bool isProfitable() = 0;
  virtual std::array<int64_t, 3> getNumThreadsInBlock() const = 0;

  /// Constructor quantities.
  MLIRContext *context;
  transform_ext::MatchedReductionCaptures captures;

  /// Derived quantities.
  SmallVector<Attribute> allBlockAttrs;
  SmallVector<Attribute> allThreadAttrs;
  // Tile sizes for the workgroup / determines grid size.
  SmallVector<int64_t> workgroupTileSizes;
};

/// Encodes a 3-staged strategy for a 1-d reduction mapped to a block.
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
/// Stage 2 (optional): the second stage of the reduction is normalized to
/// reduce from a "k-warps" abstraction (k determined in Step 1) to a single
/// `warpShuffleSize` that can be reduced unambiguously well on a single warp
/// during Stage 3. This second stage is optional and only occurs when k > 1.
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
class ReductionStrategyStagedThreadDistribution : public GPUReductionStrategy {
 public:
  ReductionStrategyStagedThreadDistribution(
      MLIRContext *context,
      const transform_ext::MatchedReductionCaptures &captures,
      int64_t maxNumThreadsToUse, int64_t warpShuffleSize)
      : GPUReductionStrategy(context, captures) {
    compute(maxNumThreadsToUse, warpShuffleSize);
  }

  ReductionStrategyStagedThreadDistribution(
      const ReductionStrategyStagedThreadDistribution &) = default;

  ReductionStrategyStagedThreadDistribution &operator=(
      const ReductionStrategyStagedThreadDistribution &) = default;

  int64_t getNumThreadsXInBlock() const { return reductionTileSizeStage1; }
  int64_t getNumThreadsYInBlock() const { return 1; }
  int64_t getNumThreadsZInBlock() const { return 1; }
  std::array<int64_t, 3> getNumThreadsInBlock() const override {
    return {getNumThreadsXInBlock(), getNumThreadsYInBlock(),
            getNumThreadsZInBlock()};
  }

  // Always profitable.
  bool isProfitable() override { return true; }

  int64_t getVectorSizeStage1() const { return vectorSizeStage1; }

  bool hasStage2() const { return reductionTileSizeStage2.has_value(); }
  int64_t getWarpShuffleSize() const { return reductionTileSizeStage2.value(); }
  int64_t getVectorSizeStage2() const { return vectorSizeStage2.value(); }

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

  /// Compute the staged strategy based on the reductionDimensionSize, the
  /// `maxNumThreadsToUse` and the `warpShuffleSize`.
  /// The latter 2 numbers control the tradeoff between parallelism and shared
  /// memory consumption.
  // TODO: Characterize shared memory consumption and limit for good occupancy.
  // TODO: Support various elemental types.
  void compute(int64_t maxNumThreadsToUse, int64_t warpShuffleSize);
};

void ReductionStrategyStagedThreadDistribution::compute(
    int64_t maxNumThreadsToUse, int64_t warpShuffleSize) {
  assert(maxNumThreadsToUse > 0 && "maxNumThreadsToUse must be > 0");
  assert(maxNumThreadsToUse >= iree_compiler::kCudaWarpSize &&
         "not even a warp?");
  assert(warpShuffleSize > 0 && "warpShuffleSize must be > 0");
  assert(warpShuffleSize % iree_compiler::kCudaWarpSize == 0 &&
         "warpShuffleSize must be a multiple of warpSize");
  assert(warpShuffleSize <= 4 * iree_compiler::kCudaWarpSize &&
         "must be smaller or equal to 4 * warp_size");

  // Block-level
  // ===========
  // Tile all the parallel dimensions to 1 and create many blocks.
  int64_t numParallelLoops = captures.reductionRank - 1;
  workgroupTileSizes.append(numParallelLoops, 1);

  // Thread-level
  // ============
  // Stage 1
  // -------
  // Maximal vector size that divides the problem size.
  // TODO: Split to ensure 4 on most of the problem and use a 1-epilogue.
  int64_t reductionDimensionSize = captures.reductionDimensionSize;
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

  // Stage 2
  // -------
  // Only needed if `reductionTileSizeStage1` consists of multiple
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

  // Stage 3
  // -------
  // Stage 3 is non-ambiguous and does not ned to be configured (always uses
  // warp shuffles).
}

/// Encodes a strategy targeted at (very) small reductions, for which other
/// strategies perform poorly.
///
/// In the case of small reductions, we cannot make an efficient use of warp
/// shuffles. Instead, try to take advantage of caches.
/// This strategy aims at running the reduction sequentially within each thread
/// and taking parallelism from outer dimensions that we would otherwise use
/// for block-level parallelism.
///
/// There are 2 cases:
///   1. we can find good divisors of outer parallel dimensions and avoid
///      creating dynamic tile sizes. We can then vectorize to the reduction
///      size.
///   2. we cannot find good divisors, we pay the price of dynamic loops.
///
// TODO: Refine 1. with linalg splitting on the reduction dimension.
// TODO: Refine 2. with linalg splitting on the parallel dimension.
//
// Note: All this is to be able to handle very small and small-ish reductions
// without catastrophic regressions.
// TODO: Add a strategy based on segmented scans, which can allow us to force
// sizes that don't divide properly into warp shuffles.
class SmallReductionStrategy : public GPUReductionStrategy {
 public:
  /// `hasTrailingElementwise` is currently used to guard against pathological
  /// cases where IREE can't bound a buffer and crashes.
  // TODO: Fix codegen/Common/PadDynamicAlloc.cpp which calls into upstream
  // code that tries to compose affine maps too aggressively when it could
  // instead resolve bounding by being more eager.
  SmallReductionStrategy(
      MLIRContext *context,
      const transform_ext::MatchedReductionCaptures &captures,
      int64_t maxNumThreadsToUse, bool hasTrailingElementwise)
      : GPUReductionStrategy(context, captures) {
    compute(maxNumThreadsToUse, hasTrailingElementwise);
  }

  SmallReductionStrategy(const SmallReductionStrategy &) = default;
  SmallReductionStrategy &operator=(const SmallReductionStrategy &) = default;

  int64_t getNumThreadsXInBlock() const { return getNumThreadsInBlock()[0]; }
  int64_t getNumThreadsYInBlock() const { return getNumThreadsInBlock()[1]; }
  int64_t getNumThreadsZInBlock() const { return getNumThreadsInBlock()[2]; }
  std::array<int64_t, 3> getNumThreadsInBlock() const override {
    std::array<int64_t, 3> res{1, 1, 1};
    for (int64_t i = 0, e = workgroupTileSizes.size(); i < e; ++i)
      res[i] = workgroupTileSizes[i];
    return res;
  }

  bool isProfitable() override { return profitable; }

 private:
  bool profitable = false;

  /// Compute the small strategy based on the problem size and the
  /// `maxNumThreadsToUse`.
  /// `hasTrailingElementwise` is currently used to guard against pathological
  /// cases where IREE can't bound a buffer and crashes.
  // TODO: Fix IREE's odegen/Common/PadDynamicAlloc.cpp.
  void compute(int64_t maxNumThreadsToUse, bool hasTrailingElementwise);
};

void SmallReductionStrategy::compute(int64_t maxNumThreadsToUse,
                                     bool hasTrailingElementwise) {
  assert(maxNumThreadsToUse > 0 && "maxNumThreadsToUse must be > 0");
  assert(maxNumThreadsToUse >= iree_compiler::kCudaWarpSize &&
         "not even a warp?");

  // Block-level
  // ===========
  // TODO: capture more dims than just the most minor parallel and have a more
  // powerful `maybeDivisor` evaluation.
  FailureOr<int64_t> maybeDivisor =
      mlir::iree_compiler::maxDivisorOfValueBelowLimit(
          captures.mostMinorParallelDimensionSize, maxNumThreadsToUse);

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
  if (ShapedType::isDynamic(captures.reductionDimensionSize)) return;

  // Otherwise, still only support the small cases for now and fall back to
  // other strategies otherwise.
  // TODO: evolve towards expressing this constraint with a perf-directed
  // matcher that composes with the existing structural matchers.
  if (captures.reductionDimensionSize >= 2 * iree_compiler::kCudaWarpSize)
    return;

  // If the captured dimension has no satisfactory divisor, just tile the last
  // parallel dimension by 2 * iree_compiler::kCudaWarpSize.
  int64_t numParallelLoops = captures.reductionRank - 1;
  workgroupTileSizes.append(numParallelLoops, 1);
  workgroupTileSizes.back() =
      hasTrailingElementwise
          ? *maybeDivisor
          : std::min((int64_t)maxNumThreadsToUse,
                     (int64_t)(2 * iree_compiler::kCudaWarpSize));

  // Thread-level
  // ============
  // Just running sequentially on each thread and relying on cache for
  // locality.
  // TODO: evolve towards expressing constraints with perf-directed matchers.
  profitable = true;
}

}  // namespace

static std::tuple<Value, Value, Value, Value>
createReductionStrategyBlockDistribution(ImplicitLocOpBuilder &b,
                                         Value maybeLeadingH, Value fillH,
                                         Value reductionH, Value maybeTrailingH,
                                         const GPUReductionStrategy &strategy) {
  auto [fusionTargetH, fusionGroupH] =
      iree_compiler::buildSelectFirstNonEmpty(b, maybeTrailingH, reductionH);
  ArrayRef<Attribute> allBlocksRef(strategy.allBlockAttrs);
  iree_compiler::TileToForeachThreadAndFuseAndDistributeResult tileResult =
      iree_compiler::
          buildTileFuseDistToForeachThreadAndWorgroupCountWithTileSizes(
              /*builder=*/b,
              /*rootH=*/fusionTargetH,
              /*opsToFuseH=*/fusionGroupH,
              /*tileSizes=*/
              getAsOpFoldResult(b.getI64ArrayAttr(strategy.workgroupTileSizes)),
              /*threadDimMapping=*/
              b.getArrayAttr(allBlocksRef.take_front(
                  strategy.captures.reductionRank - 1)));
  fillH = b.create<FuseIntoContainingOp>(fillH, tileResult.foreachThreadH);
  maybeLeadingH =
      b.create<FuseIntoContainingOp>(maybeLeadingH, tileResult.foreachThreadH);

  auto [gridReductionH, maybeGridTrailingH] =
      iree_compiler::buildSelectFirstNonEmpty(
          b, tileResult.resultingFusedOpsHandles.front(), tileResult.tiledOpH);
  return std::make_tuple(maybeLeadingH, fillH, gridReductionH,
                         maybeGridTrailingH);
}

static std::tuple<Value, Value, Value>
createReductionStrategyStagedThreadDistributionStep(
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

static void createReductionStrategyStagedThreadDistribution(
    ImplicitLocOpBuilder &b, Value gridReductionH, Value maybeTiledLeadingH,
    Value maybeTiledTrailingH,
    const ReductionStrategyStagedThreadDistribution &strategy) {
  // Map the potential maybeTiledLeadingH.
  // TODO: Consider fusing leading elementwise into threads.
  if (strategy.captures.maybeLeadingRank > 0) {
    createElementwiseStrategyThreadStep(b, maybeTiledLeadingH,
                                        strategy.captures.maybeLeadingRank,
                                        strategy.getNumThreadsXInBlock());
  }

  // Staged reduction step 1: break gridReductionH apart.
  auto [blockParallelForeachThreadOp, blockParallelFillH, blockCombinerOpH] =
      createReductionStrategyStagedThreadDistributionStep(
          b, gridReductionH, strategy.captures.reductionRank,
          strategy.getNumThreadsXInBlock(), strategy.getVectorSizeStage1());

  // Staged reduction step 2: break blockCombinerOpH apart.
  // Note, if necessary, we could have additional intermediate steps.
  Value warpParallelForeachThreadOp, warpParallelFillH, warpCombinerOpH;
  if (strategy.hasStage2()) {
    std::tie(warpParallelForeachThreadOp, warpParallelFillH, warpCombinerOpH) =
        createReductionStrategyStagedThreadDistributionStep(
            b, blockCombinerOpH, strategy.captures.reductionRank,
            strategy.getWarpShuffleSize(), strategy.getVectorSizeStage1());
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
  if (strategy.captures.maybeTrailingRank > 0) {
    createElementwiseStrategyThreadStep(b, maybeTiledTrailingH,
                                        strategy.captures.maybeTrailingRank,
                                        strategy.getNumThreadsXInBlock());
  }
}

/// Take care of the last common steps in a GPU strategy (i.e. vectorize,
/// bufferize, maps to blocks and threads and distribute vectors).
static void createCommonTrailingStrategy(ImplicitLocOpBuilder &b,
                                         Value variantH,
                                         const GPUReductionStrategy &strategy) {
  // Step N-2. Bufferize and drop HAL decriptor from memref ops.
  Value funcH = b.create<MatchOp>(variantH, func::FuncOp::getOperationName());
  funcH = iree_compiler::buildVectorize(b, funcH);
  variantH = iree_compiler::buildBufferize(b, variantH,
                                           /*targetGpu=*/true);

  // Step N-1. Post-bufferization mapping to blocks and threads.
  // Need to match again since bufferize invalidated all handles.
  // TODO: assumes a single func::FuncOp to transform, may need hardening.
  funcH = b.create<MatchOp>(variantH, func::FuncOp::getOperationName());
  funcH = iree_compiler::buildMapToBlockAndThreads(
      b, funcH, strategy.getNumThreadsInBlock());

  // Step N. Post-bufferization vector distribution with rank-reduction.
  iree_compiler::buildDistributeVectors(b, variantH, funcH);
}

/// Builds the transform IR tiling reductions for CUDA targets. Supports
/// reductions in the last dimension, with optional leading and trailing
/// elementwise operations.
static void createCudaReductionStrategyStagedThreadDistribution(
    ImplicitLocOpBuilder &b, Value variantH,
    const ReductionStrategyStagedThreadDistribution &strategy) {
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
      createReductionStrategyBlockDistribution(
          b, maybeLeadingH, fillH, reductionH, maybeTrailingH, strategy);

  // Step 3. Split the reduction and tile the pieces to ensure vector
  // load/stores and mapping to a single warp with shuffles.
  // TODO: consider fusing gridFillH.
  createReductionStrategyStagedThreadDistribution(
      b, gridReductionH, maybeLeadingHBlock, maybeTiledTrailingHBlock,
      strategy);

  // Step 4-6. Common trailing steps.
  createCommonTrailingStrategy(b, variantH, strategy);
}

static std::tuple<Value, Value, Value>
createSmallReductionStrategyThreadDistribution(
    ImplicitLocOpBuilder &b, Value maybeLeadingH, Value fillH, Value reductionH,
    Value maybeTrailingH, const GPUReductionStrategy &strategy) {
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

  // Scalarize all ops to ensure vectorization.
  auto pdlOperation = pdl::OperationType::get(b.getContext());
  fillH = b.create<ScalarizeOp>(pdlOperation, fillH);
  maybeLeadingH = b.create<ScalarizeOp>(pdlOperation, maybeLeadingH);
  Value tiledH = b.create<ScalarizeOp>(pdlOperation, tileResult.tiledOpH);
  Value fusedH = b.create<ScalarizeOp>(
      pdlOperation, tileResult.resultingFusedOpsHandles.front());

  auto [blockReductionH, maybeBlockTrailingH] =
      iree_compiler::buildSelectFirstNonEmpty(b, fusedH, tiledH);
  return std::make_tuple(maybeLeadingH, blockReductionH, maybeBlockTrailingH);
}

/// Builds the transform IR tiling reductions for CUDA targets. Supports
/// reductions in the last dimension, with optional leading and trailing
/// elementwise operations.
static void createCudaSmallReductionStrategy(
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
      createReductionStrategyBlockDistribution(
          b, maybeLeadingH, fillH, reductionH, maybeTrailingH, strategy);

  // Step 3. Apply thread-level part of the strategy, keeps everything fused.
  createSmallReductionStrategyThreadDistribution(
      b, maybeLeadingHBlock, gridFillH, gridReductionH,
      maybeTiledTrailingHBlock, strategy);

  // Step 4-6. Common trailing steps.
  createCommonTrailingStrategy(b, variantH, strategy);
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

static ReductionStrategyStagedThreadDistribution
configureGPUReductionStrategyStagedThreadDistribution(
    MLIRContext *context,
    const transform_ext::MatchedReductionCaptures &captures) {
  // TODO: Generalize along the HW axis.
  GPUReductionConfig gpuReductionConfig = getReductionConfigRTX2080Ti(captures);
  ReductionStrategyStagedThreadDistribution strategy(
      context, captures, gpuReductionConfig.maxNumThreads,
      gpuReductionConfig.warpShuffleSize);
  LLVM_DEBUG(DBGS() << "use staged reduction strategy\n");
  return strategy;
}

static FailureOr<SmallReductionStrategy> configureGPUSmallReductionStrategy(
    MLIRContext *context,
    const transform_ext::MatchedReductionCaptures &captures,
    bool hasTrailingElementwise) {
  // TODO: Generalize along the HW axis.
  GPUReductionConfig gpuReductionConfig = getReductionConfigRTX2080Ti(captures);
  SmallReductionStrategy strategy(context, captures,
                                  gpuReductionConfig.maxNumThreads,
                                  hasTrailingElementwise);
  if (!strategy.isProfitable()) return failure();
  LLVM_DEBUG(DBGS() << "use small reduction strategy\n");
  return strategy;
}

LogicalResult iree_compiler::matchAndSetGPUReductionTransformStrategy(
    func::FuncOp entryPoint, linalg::LinalgOp op) {
  // 1. Match a reduction and surrounding ops.
  StructuredOpMatcher reduction, fill, leading, trailing;
  transform_ext::MatchedReductionCaptures captures;
  makeReductionMatcher(reduction, fill, leading, trailing, captures);
  if (!matchPattern(op, reduction)) return failure();

  // 2. Construct the configuration and the strategy builder.
  auto strategyBuilder = [&](ImplicitLocOpBuilder &b, Value variant) {
    // TODO: Better strategy for very small reductions.
    FailureOr<SmallReductionStrategy> maybeSmallStrategy =
        configureGPUSmallReductionStrategy(op->getContext(), captures,
                                           trailing.getCaptured());
    if (succeeded(maybeSmallStrategy))
      return createCudaSmallReductionStrategy(b, variant, *maybeSmallStrategy);

    ReductionStrategyStagedThreadDistribution strategy =
        configureGPUReductionStrategyStagedThreadDistribution(op->getContext(),
                                                              captures);
    return createCudaReductionStrategyStagedThreadDistribution(b, variant,
                                                               strategy);
  };

  // 3. Build strategy embedded into the IR.
  createTransformRegion(entryPoint, strategyBuilder);
  return success();
}

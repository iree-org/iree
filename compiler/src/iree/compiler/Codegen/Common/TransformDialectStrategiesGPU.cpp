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
#include "mlir/Dialect/SCF/IR/SCF.h"
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

/// Compute the (splitPoint, vectorSize) pair to break [0 .. upperBound] into
/// [0 .. splitPoint] and [splitPoint + 1 .. upperBound] such that `splitPoint`
/// is a multiple of `minMultiple * vectorSize`.
/// `vectorSize` is the maximal power of `2`, smaller than `maxVectorSize`, for
/// which `splitPoint` can be computed.
///
/// If such a positive multiple exists:
///   1. if it is `upperBound`, then `upperBound` is an even multiple of
///      `minMultiple` * `vectorSize` and we can tile evenly without splitting.
///      In this case we return (0, vectorSize).
///   2. otherwise, it is a split point at which we can split with vectorSize
///      to obtain the largest divisible tiling.
///      In this case we return (splitPoint, vectorSize).
/// Otherwise we return (0, 1) to signify no splitting and a vector size of 1.
// TODO: support the dynamic case, taking future stride and alignment into
// account and returning Values. The op then needs to become part of the
// transform dialect.
static std::pair<int64_t, int64_t> computeSplitPoint(int64_t upperBound,
                                                     int64_t minMultiple,
                                                     int64_t maxVectorSize) {
  assert((maxVectorSize & (maxVectorSize - 1)) == 0 && "must be a power of 2");
  if (ShapedType::isDynamic(upperBound)) return std::make_pair(0l, 1l);
  for (int64_t vectorSize = maxVectorSize; vectorSize >= 1; vectorSize >>= 1) {
    int64_t splitPoint =
        iree_compiler::previousMultipleOf(upperBound, minMultiple * vectorSize);
    if (splitPoint > 0) {
      return (upperBound == splitPoint)
                 ? std::make_pair(0l, vectorSize)
                 : std::make_pair(splitPoint, vectorSize);
    }
  }
  return std::make_pair(0l, 1l);
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
/// Stage 3: second stage of the the warp shuffle step reduces a vector<k x T>
/// element to a single element. Only threadIdx == 0 commits to memory.
///
/// Stage 2: the second stage of the reduction is the first stage of the warp
/// shuffle step. It is normalized to reduce from a "k-warps" abstraction,
/// across all warps in parallel, to a k-element result. Only the first thread
/// within each warp (e.g. threadIdx % kCudaWarpSize == 0) commits to memory.
///
/// Stage 1: the first stage of the reduction is normalized to run on "k-warps"
/// of maximal vector size for both the hardware and the problem sizes.
/// The overprovisioning to "k-warps" allows multiple warps to run in parallel.
/// The `reductionTileSize` is this "k-warps" quantity and is also the
/// number of threads (i.e. blockDim.x) used to parallelize the problem.
/// This also results in `reductionTileSize` live values that are
/// allocated in shared memory and creates a tradeoff between parallelism and
/// occupancy.
/// The normalization guarantees that whatever the problem size P, we reduce
/// from `tensor<P x T>` to `tensor<reductionTileSize x T>` by using the
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
      int64_t maxNumThreadsToUse, int64_t vectorSize)
      : GPUReductionStrategy(context, captures) {
    compute(maxNumThreadsToUse, vectorSize);
  }

  ReductionStrategyStagedThreadDistribution(
      const ReductionStrategyStagedThreadDistribution &) = default;

  ReductionStrategyStagedThreadDistribution &operator=(
      const ReductionStrategyStagedThreadDistribution &) = default;

  int64_t getNumThreadsXInBlock() const { return reductionTileSize; }
  int64_t getNumThreadsYInBlock() const { return 1; }
  int64_t getNumThreadsZInBlock() const { return 1; }
  std::array<int64_t, 3> getNumThreadsInBlock() const override {
    return {getNumThreadsXInBlock(), getNumThreadsYInBlock(),
            getNumThreadsZInBlock()};
  }

  // Always profitable.
  bool isProfitable() override { return true; }

  int64_t getVectorSize() const { return vectorSize; }

 private:
  /// Maximal vector size (among {1, 2, 4}) that divides the
  /// `reductionDimensionSize` and is used for vector transfers in Stage 1.
  int64_t vectorSize;
  /// Maximal "k-warp" size within the limits of the `maxNumThreadsToUse` and
  /// `reductionDimensionSize` parameters.
  /// This is also the blockDim.x of the kernel.
  int64_t reductionTileSize;

  /// Compute the staged strategy based on the reductionDimensionSize, the
  /// `maxNumThreadsToUse` and the `vectorSize`.
  /// The latter 2 numbers control the tradeoff between parallelism and shared
  /// memory consumption.
  // TODO: Characterize shared memory consumption and limit for good occupancy.
  // TODO: Support various elemental types.
  void compute(int64_t maxNumThreadsToUse, int64_t maxVectorSize);
};

void ReductionStrategyStagedThreadDistribution::compute(
    int64_t maxNumThreadsToUse, int64_t maxVectorSize) {
  assert(maxNumThreadsToUse > 0 && "maxNumThreadsToUse must be > 0");
  assert(maxNumThreadsToUse >= iree_compiler::kCudaWarpSize &&
         "not even a warp?");
  assert(maxVectorSize > 0 && "maxVectorSize must be > 0");
  assert(maxVectorSize <= 4 && "must be smaller or equal to 4");

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
  int64_t reductionDimensionSize = captures.reductionOpSizes.back();
  if (reductionDimensionSize > 0 && reductionDimensionSize % 4 == 0)
    vectorSize = std::min(maxVectorSize, int64_t(4));
  else if (reductionDimensionSize > 0 && reductionDimensionSize % 2 == 0)
    vectorSize = std::min(maxVectorSize, int64_t(2));
  else
    vectorSize = 1;
  // Tile reduction to the maximal multiple `vectorSize` allowed.
  // This locally reduces the large unknown reduction into a guaranteed
  // multiple of `vectorSize`.
  if (reductionDimensionSize > 0) {
    reductionTileSize = std::min(
        iree_compiler::nextMultipleOf(reductionDimensionSize / vectorSize,
                                      iree_compiler::kCudaWarpSize),
        iree_compiler::previousMultipleOf(maxNumThreadsToUse,
                                          iree_compiler::kCudaWarpSize));
  } else {
    reductionTileSize = iree_compiler::previousMultipleOf(
        maxNumThreadsToUse, iree_compiler::kCudaWarpSize);
  }
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
  int64_t mostMinorParallelDimensionSize =
      ArrayRef<int64_t>(captures.reductionOpSizes).drop_back().back();
  FailureOr<int64_t> maybeDivisor =
      mlir::iree_compiler::maxDivisorOfValueBelowLimit(
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
  if (reductionDimensionSize >= 2 * iree_compiler::kCudaWarpSize) return;

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

/// Given a handle `elementwiseH` to an elementwise op of rank `rank`, sizes
/// `elementwiseSizes` mapped to `numThreadsXInBlock` threads along dimension x.
/// Build a schedule that maps the most minor dimension to a scf.foreach op
/// itself mapped to the `gpu.thread x` dimension.
/// The schedule first performs a split of the largest possible multiple of
/// `numThreadsXInBlock * maxVectorSize` to form a maximally divisible region
/// Assumes the most minor dimension of the op is the last one.
// TODO: More robustness wrt selecting the most minor dimension otherwise
// performance may suffer.
// TODO: Split point should be dynamic and aware of future stride / alignment
// to also guarantee proper vector alignments.
static void create1DSplittingStrategyWithOptionalThreadMapping(
    ImplicitLocOpBuilder &b, Value elementwiseH, int64_t rank,
    SmallVector<int64_t> elementwiseSizes, int64_t numThreads,
    int64_t maxVectorSize = 4) {
  if (rank == 0) return;

  int64_t mostMinorDim = rank - 1;
  int64_t mostMinorSize = elementwiseSizes[mostMinorDim];
  auto [splitPoint, vectorSize] =
      computeSplitPoint(mostMinorSize, numThreads, maxVectorSize);

  SmallVector<int64_t> scfForTileSizes(rank, 0), foreachTileSizes(rank, 0);
  scfForTileSizes[mostMinorDim] = numThreads * vectorSize;
  foreachTileSizes[mostMinorDim] = numThreads;

  auto threadX = mlir::gpu::GPUThreadMappingAttr::get(b.getContext(),
                                                      mlir::gpu::Threads::DimX);
  // Split, tile and map the most minor dimension to `gpu.thread x`.
  if (splitPoint > 0) {
    auto pdlOperation = pdl::OperationType::get(b.getContext());
    auto split =
        b.create<transform::SplitOp>(pdlOperation, pdlOperation, elementwiseH,
                                     b.getI64IntegerAttr(mostMinorDim), Value(),
                                     b.getI64IntegerAttr(splitPoint));
    elementwiseH = split.getFirst();
    if (vectorSize > 1) {
      auto res = iree_compiler::buildTileFuseToScfFor(
          b, elementwiseH, {},
          getAsOpFoldResult(b.getI64ArrayAttr({scfForTileSizes})));
      elementwiseH = res.tiledOpH;
    }
    if (numThreads > 1) {
      iree_compiler::buildTileFuseDistToForeachThreadWithNumThreads(
          b, elementwiseH, {},
          getAsOpFoldResult(b.getI64ArrayAttr(foreachTileSizes)),
          b.getArrayAttr({threadX}));
    }
    elementwiseH = split.getSecond();
  }
  // Tile and map the most minor dimension of the remainder to `gpu.thread x`.
  if (vectorSize > 1) {
    auto res = iree_compiler::buildTileFuseToScfFor(
        b, elementwiseH, {},
        getAsOpFoldResult(b.getI64ArrayAttr({scfForTileSizes})));
    elementwiseH = res.tiledOpH;
  }
  if (numThreads > 1) {
    iree_compiler::buildTileFuseDistToForeachThreadWithNumThreads(
        b, elementwiseH, {},
        getAsOpFoldResult(b.getI64ArrayAttr(foreachTileSizes)),
        b.getArrayAttr({threadX}));
  }
}

static void createReductionStrategyStagedThreadDistribution(
    ImplicitLocOpBuilder &b, Value gridReductionH, Value maybeTiledLeadingH,
    Value maybeTiledTrailingH,
    const ReductionStrategyStagedThreadDistribution &strategy) {
  // Map the potential maybeTiledLeadingH.
  // TODO: Consider fusing leading elementwise into threads.
  if (strategy.captures.maybeLeadingRank > 0) {
    create1DSplittingStrategyWithOptionalThreadMapping(
        b, maybeTiledLeadingH, strategy.captures.maybeLeadingRank,
        strategy.captures.leadingOpSizes, strategy.getNumThreadsXInBlock());
  }

  // Staged reduction step 1: break gridReductionH apart.
  auto [blockParallelForeachThreadOp, blockParallelFillH, blockCombinerOpH] =
      createReductionStrategyStagedThreadDistributionStep(
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
    create1DSplittingStrategyWithOptionalThreadMapping(
        b, maybeTiledTrailingH, strategy.captures.maybeTrailingRank,
        strategy.captures.trailingOpSizes, strategy.getNumThreadsXInBlock());
  }
}

/// Take care of the last common steps in a GPU strategy (i.e. vectorize,
/// bufferize, maps to blocks and threads and distribute vectors).
/// Return the handles to the updated variant and the func::FuncOp ops under
/// the variant op.
static std::pair<Value, Value> createCommonTrailingStrategy(
    ImplicitLocOpBuilder &b, Value variantH,
    const GPUReductionStrategy &strategy) {
  // Step N-1. Bufferize and drop HAL decriptor from memref ops.
  Value funcH = b.create<MatchOp>(variantH, func::FuncOp::getOperationName());
  funcH = iree_compiler::buildVectorize(b, funcH);
  variantH = iree_compiler::buildBufferize(b, variantH,
                                           /*targetGpu=*/true);

  // Step N. Post-bufferization mapping to blocks and threads.
  // Need to match again since bufferize invalidated all handles.
  // TODO: assumes a single func::FuncOp to transform, may need hardening.
  funcH = b.create<MatchOp>(variantH, func::FuncOp::getOperationName());
  funcH = iree_compiler::buildMapToBlockAndThreads(
      b, funcH, strategy.getNumThreadsInBlock());

  return std::make_pair(variantH, funcH);
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

  // Step 4-5. Common trailing steps.
  auto [variantH2, funcH] = createCommonTrailingStrategy(b, variantH, strategy);

  // Step 6. The staged strategy has a post-bufferization vector distribution
  // with rank-reduction. The vector distribution occurs on multiple warps and
  // is itself staged in 2 steps.
  iree_compiler::buildDistributeVectors(b, variantH2, funcH,
                                        strategy.getNumThreadsXInBlock());
}

static void createSmallReductionStrategyThreadDistribution(
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
  create1DSplittingStrategyWithOptionalThreadMapping(
      b, blockReductionH, strategy.captures.reductionRank,
      strategy.captures.reductionOpSizes,
      /*numThreads=*/1);

  // 3. apply the 1d splitting strategy to the trailing elementwise.
  create1DSplittingStrategyWithOptionalThreadMapping(
      b, maybeBlockTrailingH, strategy.captures.maybeTrailingRank,
      strategy.captures.trailingOpSizes,
      strategy.getNumThreadsInBlock().back());
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

  // Step 4-5. Common trailing steps.
  createCommonTrailingStrategy(b, variantH, strategy);
}

struct GPUReductionConfig {
  int64_t maxNumThreads;
  int64_t vectorSize;
};

// TODO: Lift some of the strategy sizing logic as hints and/or heuristics to
// also work properly in the dynamic case.
// TODO: Support more HW configs and make it more pluggable.
static GPUReductionConfig getStagedReductionConfigRTX2080Ti(
    const transform_ext::MatchedReductionCaptures &captures) {
  int64_t vectorSize = 4;
  int64_t maxNumThreads = 8 * iree_compiler::kCudaWarpSize;
  // No adjustments in the dynamic case, we need extra information to make a
  // good decision.
  int64_t reductionDimensionSize = captures.reductionOpSizes.back();
  if (ShapedType::isDynamic(reductionDimensionSize))
    return GPUReductionConfig{maxNumThreads, vectorSize};
  // Scale down to smaller sizes (4, 8, 16)-warps.
  if (reductionDimensionSize <= 4 * iree_compiler::kCudaWarpSize) {
    vectorSize = 1;
    maxNumThreads = 4 * iree_compiler::kCudaWarpSize;
  } else if (reductionDimensionSize <= 8 * iree_compiler::kCudaWarpSize) {
    vectorSize = 2;
    maxNumThreads = 4 * iree_compiler::kCudaWarpSize;
  } else if (reductionDimensionSize <= 8 * 2 * iree_compiler::kCudaWarpSize) {
    vectorSize = 4;
    maxNumThreads = 4 * iree_compiler::kCudaWarpSize;
  }
  // Scale up to larger sizes (32, 64, 128+)-warps, using vector-4.
  if (!captures.trailingOpSizes.empty()) {
    if (reductionDimensionSize >= 32 * 4 * iree_compiler::kCudaWarpSize) {
      vectorSize = 4;
      maxNumThreads = 32 * iree_compiler::kCudaWarpSize;
    } else if (reductionDimensionSize >=
               16 * 4 * iree_compiler::kCudaWarpSize) {
      vectorSize = 4;
      maxNumThreads = 16 * iree_compiler::kCudaWarpSize;
    } else if (reductionDimensionSize >= 8 * 4 * iree_compiler::kCudaWarpSize) {
      vectorSize = 4;
      maxNumThreads = 8 * iree_compiler::kCudaWarpSize;
    }
  }
  return GPUReductionConfig{maxNumThreads, vectorSize};
}

static ReductionStrategyStagedThreadDistribution
configureGPUReductionStrategyStagedThreadDistribution(
    MLIRContext *context,
    const transform_ext::MatchedReductionCaptures &captures) {
  // TODO: Generalize along the HW axis.
  GPUReductionConfig gpuReductionConfig =
      getStagedReductionConfigRTX2080Ti(captures);
  ReductionStrategyStagedThreadDistribution strategy(
      context, captures, gpuReductionConfig.maxNumThreads,
      gpuReductionConfig.vectorSize);
  LLVM_DEBUG(DBGS() << "use staged reduction strategy\n");
  return strategy;
}

static GPUReductionConfig getSmallReductionConfigRTX2080Ti(
    const transform_ext::MatchedReductionCaptures &captures) {
  int64_t maxNumThreads = 4 * iree_compiler::kCudaWarpSize;
  return GPUReductionConfig{maxNumThreads, 0};
}

static FailureOr<SmallReductionStrategy> configureGPUSmallReductionStrategy(
    MLIRContext *context,
    const transform_ext::MatchedReductionCaptures &captures,
    bool hasTrailingElementwise) {
  // TODO: Generalize along the HW axis.
  GPUReductionConfig gpuReductionConfig =
      getSmallReductionConfigRTX2080Ti(captures);
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

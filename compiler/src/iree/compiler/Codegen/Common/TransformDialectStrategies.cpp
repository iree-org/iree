// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/TransformDialectStrategies.h"

#include <numeric>
#include <type_traits>

#include "iree-dialects/Dialect/LinalgTransform/StructuredTransformOpsExt.h"
#include "iree/compiler/Codegen/Common/TransformExtensions/CommonExtensions.h"
#include "iree/compiler/Codegen/Common/TransformExtensions/TransformMatchers.h"
#include "iree/compiler/Codegen/LLVMGPU/TransformExtensions/LLVMGPUExtensions.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;

namespace mlir {
namespace iree_compiler {

#define DEBUG_TYPE "iree-transform-builder"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

// TODO: significantly better namespacing.
using iree_compiler::IREE::transform_dialect::AllDims;
using iree_compiler::IREE::transform_dialect::ApplyPatternsOp;
using iree_compiler::IREE::transform_dialect::ConfigExtractPart;
using iree_compiler::IREE::transform_dialect::ForeachThreadToWorkgroupOp;
using iree_compiler::IREE::transform_dialect::IREEBufferizeOp;
using iree_compiler::IREE::transform_dialect::
    IREEEraseHALDescriptorTypeFromMemRefOp;
using iree_compiler::IREE::transform_dialect::IsPermutation;
using iree_compiler::IREE::transform_dialect::m_StructuredOp;
using iree_compiler::IREE::transform_dialect::
    MapNestedForeachThreadToGpuThreadsOp;
using iree_compiler::IREE::transform_dialect::NumEqualsTo;
using iree_compiler::IREE::transform_dialect::ShapeKind;
using iree_compiler::IREE::transform_dialect::StructuredOpMatcher;
using iree_compiler::IREE::transform_dialect::
    TileToForeachThreadAndWorkgroupCountRegionOp;
using iree_compiler::IREE::transform_dialect::VectorToWarpExecuteOnLane0Op;
using iree_compiler::IREE::transform_dialect::VectorWarpDistributionOp;
using transform::FuseIntoContainingOp;
using transform::MatchOp;
using transform::MergeHandlesOp;
using transform::PrintOp;
using transform::SequenceOp;
using transform::SplitHandlesOp;
using transform::SplitReductionOp;
using transform::TileToForeachThreadOp;
using transform::VectorizeOp;

/// Matches `args` within `targetH` and unpacks a number of handles `N`.
/// Assumes there are exactly `N` matched ops (but could be relaxed).
/// Returns the tuple of handles.
template <int N, typename... MatchingArgs>
auto matchAndUnpack(ImplicitLocOpBuilder &b, Value targetH,
                    MatchingArgs... args) {
  Value matchedH = b.create<MatchOp>(targetH, args...);
  auto matchOp = b.create<SplitHandlesOp>(matchedH,
                                          /*numHandles=*/N);
  assert(matchOp->getNumResults() == N && "Unexpected number of results");
  std::array<Value, N> a;
  for (int64_t i = 0; i < N; ++i) a[i] = matchOp->getResult(i);
  return std::tuple_cat(a);
}

//===----------------------------------------------------------------------===//
// Low-level reusable builder APIs, these should follow MLIR-style builders.
//===----------------------------------------------------------------------===//

/// Prints `handles` in order. Prints the whole IR if `handles` is empty.
static void buildPrint(ImplicitLocOpBuilder &b, ValueRange handles = {}) {
  if (handles.empty()) b.create<PrintOp>();
  for (auto h : handles) b.create<PrintOp>(h);
}

/// Performs the following transformations:
///   1. Tiles `rootH` to scf.foreach_thread to with `tileSizesOrNumThreads`
///      according to whether spec is a TileSizesSpec or a NumThreadsSpec.
///   2. Maps the resulting scf.foreach_thread to threads according to
///      `threadDimMapping`.
///   3. Iterates over `opsHToFuse` in order and fuses into the containing op.
/// Returns a handle to the resulting scf.foreach_thread.
///
/// Fusion operates in batch mode: a single fusion command is issued and a
/// topological sort is automatically computed by the fusion.
/// Since this applies a single fusion, no interleaved canonicalization / cse /
/// enabling transformation occurs and the resulting fusion may not be as good.
///
/// In the future, an iterative mode in which the user is responsible for
/// providing the fusion order and has interleaved canonicalization / cse /
/// enabling transform will be introduced and may result in better fusions.
///
/// If `resultingFusedOpsHandles` is a non-null pointer, the fused operation are
/// appended in order.
// TODO: apply forwarding pattern.
template <typename TilingTransformOp, typename TileOrNumThreadSpec>
static Value buildTileAndFuseAndDistributeImpl(
    ImplicitLocOpBuilder &b, Value rootH, ValueRange opsHToFuse,
    ArrayRef<OpFoldResult> tileSizesOrNumThreads, ArrayAttr threadDimMapping,
    SmallVectorImpl<Value> *resultingFusedOpsHandles) {
  auto tileToForeachOp = b.create<TilingTransformOp>(
      rootH, tileSizesOrNumThreads, TileOrNumThreadSpec(), threadDimMapping);
  Value foreachThreadH = tileToForeachOp.getForeachThreadOp();
  // Batch fusion.
  Value mergedOpsH = b.create<MergeHandlesOp>(opsHToFuse, /*deduplicate=*/true);
  b.create<FuseIntoContainingOp>(mergedOpsH, foreachThreadH);
  assert(!resultingFusedOpsHandles && "Handle needs unpacking");
  return foreachThreadH;
}

/// Call buildTileAndFuseAndDistributeImpl with ArrayRef<int64_t> tilesSizes.
template <typename TilingTransformOp = TileToForeachThreadOp>
static Value buildTileFuseDistWithTileSizes(
    ImplicitLocOpBuilder &b, Value rootH, ValueRange opsHToFuse,
    ArrayRef<OpFoldResult> tileSizes, ArrayAttr threadDimMapping,
    SmallVectorImpl<Value> *resultingFusedOpsHandles = nullptr) {
  return buildTileAndFuseAndDistributeImpl<TilingTransformOp,
                                           transform::TileSizesSpec>(
      b, rootH, opsHToFuse, tileSizes, threadDimMapping,
      resultingFusedOpsHandles);
}

/// Call buildTileAndFuseAndDistributeImpl with ArrayRef<int64_t> numThreads.
template <typename TilingTransformOp = TileToForeachThreadOp>
static Value buildTileFuseDistWithNumThreads(
    ImplicitLocOpBuilder &b, Value rootH, ValueRange opsHToFuse,
    ArrayRef<int64_t> numThreads, ArrayAttr threadDimMapping,
    SmallVectorImpl<Value> *resultingFusedOpsHandles = nullptr) {
  return buildTileAndFuseAndDistributeImpl<TilingTransformOp,
                                           transform::NumThreadsSpec>(
      b, rootH, opsHToFuse, getAsOpFoldResult(b.getI64ArrayAttr(numThreads)),
      threadDimMapping, resultingFusedOpsHandles);
}

/// Call buildTileAndFuseAndDistributeImpl with a handle to multiple numThreads.
template <typename TilingTransformOp = TileToForeachThreadOp>
static Value buildTileFuseDistWithNumThreads(
    ImplicitLocOpBuilder &b, Value rootH, ValueRange opsHToFuse,
    Value numThreads, ArrayAttr threadDimMapping,
    SmallVectorImpl<Value> *resultingFusedOpsHandles = nullptr) {
  return buildTileAndFuseAndDistributeImpl<TilingTransformOp,
                                           transform::NumThreadsSpec>(
      b, rootH, opsHToFuse, ArrayRef<OpFoldResult>{numThreads},
      threadDimMapping, resultingFusedOpsHandles);
}

/// Apply patterns and vectorize (for now always applies rank-reduction).
/// Takes a handle to a func.func and returns an updated handle to a
/// func.func.
// TODO: configure patterns.
static Value buildVectorizeStrategy(ImplicitLocOpBuilder &b, Value funcH) {
  funcH = b.create<ApplyPatternsOp>(funcH, /*rankReducing=*/true);
  return b.create<VectorizeOp>(funcH);
}

/// Post-bufferization mapping to blocks and threads.
/// Takes a handle to a func.func and returns an updated handle to a
/// func.func.
static Value buildMapToBlockAndThreads(ImplicitLocOpBuilder &b, Value funcH,
                                       ArrayRef<int64_t> blockSize) {
  funcH = b.create<ForeachThreadToWorkgroupOp>(funcH);
  return b.create<MapNestedForeachThreadToGpuThreadsOp>(funcH, blockSize);
}

static constexpr unsigned kCudaWarpSize = 32;

/// Post-bufferization vector distribution with rank-reduction.
/// Takes a handle to a func.func and returns an updated handle to a
/// func.func.
static Value buildDistributeVectors(ImplicitLocOpBuilder &b, Value variantH,
                                    Value funcH,
                                    int64_t warpSize = kCudaWarpSize) {
  funcH = b.create<ApplyPatternsOp>(funcH, /*rankReducing=*/true);
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

/// Various handles produced by reduction splitting.
struct ReductionSplitResult {
  /// Handle to the leading elementwise operation, may be null if no such
  /// operation is present.
  Value leadingEltwiseH;
  /// Handle to the fill operation feeding the init of a higher-rank
  /// more-parallel reduction.
  Value splitFillH;
  /// Handle to the higher-rank more-parallel reduction.
  Value splitLinalgH;
  /// Handle to the final reduction.
  Value combinerH;
  /// Handle to the original fill operation, may be null if the operation was
  /// not re-matched.
  Value originalFillH;
  /// Handle to the trailing fill operation, may be null if the operation was
  /// not re-matched.
  Value trailingEltwiseH;
};

/// Builds transform IR requesting to bubble up the "expand_shape" operation
/// produced as parent of reduction splitting if necessary for fusion of the
/// leading elementwise operation.
// TODO: consider passing a problem-specific struct to control information.
static ReductionSplitResult createExpansionBubbleUp(
    ImplicitLocOpBuilder &b, Value variantH,
    SplitReductionOp splitReductionTransformOp, bool hasLeadingEltwise,
    bool hasTrailingEltwise) {
  ReductionSplitResult result;
  if (!hasLeadingEltwise) {
    result.splitFillH = splitReductionTransformOp.getFillOp();
    result.splitLinalgH = splitReductionTransformOp.getSplitLinalgOp();
    result.combinerH = splitReductionTransformOp.getCombiningLinalgOp();
    return result;
  }

  auto funcH = b.create<MatchOp>(variantH, func::FuncOp::getOperationName());
  auto applyPatterns = b.create<ApplyPatternsOp>(funcH, /*rankReducing=*/false);
  applyPatterns->setAttr(applyPatterns.getBubbleCollapseExpandAttrName(),
                         b.getUnitAttr());
  std::tie(result.originalFillH, result.splitFillH) =
      matchAndUnpack<2>(b, variantH, linalg::FillOp::getOperationName());
  if (hasTrailingEltwise) {
    std::tie(result.leadingEltwiseH, result.splitLinalgH, result.combinerH,
             result.trailingEltwiseH) =
        matchAndUnpack<4>(b, variantH, linalg::GenericOp::getOperationName());
  } else {
    std::tie(result.leadingEltwiseH, result.splitLinalgH, result.combinerH) =
        matchAndUnpack<3>(b, variantH, linalg::GenericOp::getOperationName());
  }
  return result;
}

/// Distribute to blocks using the current IREE lowering config.
// TODO: consider passing a problem-specific struct to control information.
static Value createReductionStrategyBlockDistributionPart(
    ImplicitLocOpBuilder &b, Value variantH, Value originalFillH,
    Value reductionH, Value optionalFusionRootH,
    ArrayRef<OpFoldResult> tileSizes0Generic, bool hasLeadingEltwise = false,
    bool hasTrailingEltwise = false) {
  // Step 1. Split the reduction to get meatier parallelism.
  // TODO: use a scf.foreach_thread for this.
  auto splitReductionTransformOp =
      b.create<SplitReductionOp>(reductionH,
                                 /*splitFactor=*/2,
                                 /*insertSplitDimension=*/1);
  ReductionSplitResult rs =
      createExpansionBubbleUp(b, variantH, splitReductionTransformOp,
                              hasLeadingEltwise, hasTrailingEltwise);

  // TODO: IREE needs own workgroup mapping attribute.
  // TODO: num of GPU block mapping attr is statically known here which is
  // brittle. In the future, the builder of scf.foreach_thread can trim the
  // number of mapping dims to the number of sizes.
  auto x = mlir::gpu::GPUBlockMappingAttr::get(b.getContext(),
                                               ::mlir::gpu::Blocks::DimX);
  // Step 2. First level of tiling + fusion parallelizes to blocks using
  // `tileSizes`. If the fusion root was the reduction op, update it to be the
  // combiner op. Otherwise, fuse the combiner op into root.
  SmallVector<Value> opsHToFuse(
      {rs.originalFillH ? rs.originalFillH : originalFillH, rs.splitFillH,
       rs.splitLinalgH});
  if (!optionalFusionRootH) {
    optionalFusionRootH = rs.combinerH;
  } else {
    optionalFusionRootH =
        rs.trailingEltwiseH ? rs.trailingEltwiseH : optionalFusionRootH;
    opsHToFuse.push_back(rs.combinerH);
  }
  if (rs.leadingEltwiseH) {
    opsHToFuse.push_back(rs.leadingEltwiseH);
  }

  // The presence of leading elementwise operation implies that dispatch region
  // formation happened using another transform dialect script and doesn't need
  // the workgroup count part.
  if (hasLeadingEltwise) {
    buildTileFuseDistWithTileSizes<TileToForeachThreadOp>(
        b, optionalFusionRootH, opsHToFuse, tileSizes0Generic,
        b.getArrayAttr({x}));
  } else {
    buildTileFuseDistWithTileSizes<
        TileToForeachThreadAndWorkgroupCountRegionOp>(
        b, optionalFusionRootH, opsHToFuse, tileSizes0Generic,
        b.getArrayAttr({x}));
  }

  return variantH;
}

// TODO: consider passing a problem-specific struct to control information.
static Value createReductionStrategyThreadDistributionPart(
    ImplicitLocOpBuilder &b, Value variantH, ArrayRef<int64_t> tileSizes1Fill,
    ArrayRef<int64_t> tileSizes1Generic, bool hasLeadingEltwise,
    bool hasTrailingEltwise) {
  // TODO: Relying on ordering is brittle, harden this.
  Value matchedH = b.create<MatchOp>(
      variantH, ArrayRef<StringRef>{linalg::GenericOp::getOperationName(),
                                    linalg::FillOp::getOperationName()});
  auto split = b.create<SplitHandlesOp>(
      matchedH,
      /*numResultHandles=*/4 + hasLeadingEltwise + hasTrailingEltwise);
  Value firstFusionRootH = split.getResults()[1 + hasLeadingEltwise];
  SmallVector<Value> firstFusionGroupHs =
      split.getResults().take_front(1 + hasLeadingEltwise);
  Value secondFusionRootH = split.getResults().back();
  SmallVector<Value> secondFusionGroupHs =
      split.getResults().drop_front(2 + hasLeadingEltwise).drop_back();

  auto z = mlir::gpu::GPUThreadMappingAttr::get(b.getContext(),
                                                ::mlir::gpu::Threads::DimZ);
  auto y = mlir::gpu::GPUThreadMappingAttr::get(b.getContext(),
                                                ::mlir::gpu::Threads::DimY);

  // clang-format off
  buildTileFuseDistWithTileSizes<TileToForeachThreadOp>(b,
                   /*rootH=*/secondFusionRootH,
                   /*opsHToFuse=*/secondFusionGroupHs,
                   /*tileSizes=*/getAsOpFoldResult(b.getI64ArrayAttr(tileSizes1Fill)),
                   /*threadDimMapping=*/b.getArrayAttr({z}));
  buildTileFuseDistWithTileSizes<TileToForeachThreadOp>(b,
                   /*rootH=*/firstFusionRootH,
                   /*opsHToFuse=*/firstFusionGroupHs,
                   /*tileSizes=*/getAsOpFoldResult(b.getI64ArrayAttr(tileSizes1Generic)),
                   /*threadDimMapping=*/b.getArrayAttr({z,y}));
  // clang-format on
  return variantH;
}

/// Structure to hold the parameters related to GPU reduction strategy.
struct GPUReductionStrategyInfos {
  std::array<int64_t, 3> workgroupSize;
  SmallVector<int64_t> workgroupTileSizes;
  SmallVector<int64_t> fillSecondTileSizes;
  SmallVector<int64_t> genericSecondTileSizes;
  bool hasLeadingEltwise;
  bool hasTrailingEltwise;
};

/// Returns a triple of handles: the leading elementwise operation, the
/// reduction operation and the fusion root. The leading elementwise and the
/// fusion root may be null. If the fusion root is null, the reduction operation
/// should be used as fusion root instead.
// TODO: consider passing a problem-specific struct to control information.
static std::tuple<Value, Value, Value>
createMatchReductionBlockDistributionHandles(ImplicitLocOpBuilder &b,
                                             Value variantH,
                                             bool hasLeadingEltwise,
                                             bool hasTrailingEltwise) {
  Value originalGenericH =
      b.create<MatchOp>(variantH, linalg::GenericOp::getOperationName());
  auto op = b.create<SplitHandlesOp>(
      originalGenericH,
      /*numResultHandles=*/1 + hasLeadingEltwise + hasTrailingEltwise);
  return std::make_tuple(hasLeadingEltwise ? op.getResults().front() : Value(),
                         op.getResults().drop_front(hasLeadingEltwise).front(),
                         hasTrailingEltwise ? op.getResults().back() : Value());
}

// TODO: generalize and automate over and over.
// TODO: significantly shrink this down.
// TODO: consider passing a problem-specific struct to control information.
static void createReductionCudaStrategy(
    ImplicitLocOpBuilder &b, Value variantH,
    const GPUReductionStrategyInfos &infos) {
  // Step 0. Match the ops.
  Value originalFillH =
      b.create<MatchOp>(variantH, linalg::FillOp::getOperationName());
  auto [leadingH, reductionH, fusionRootH] =
      createMatchReductionBlockDistributionHandles(
          b, variantH, infos.hasLeadingEltwise, infos.hasTrailingEltwise);

  // Step 1: Distribute to blocks using the current IREE lowering config.
  variantH = createReductionStrategyBlockDistributionPart(
      b, variantH, originalFillH, reductionH, fusionRootH,
      getAsOpFoldResult(b.getI64ArrayAttr(infos.workgroupTileSizes)),
      infos.hasLeadingEltwise, infos.hasTrailingEltwise);

  // Step 2. Second level of tiling + fusion parallelizes to threads.
  variantH = createReductionStrategyThreadDistributionPart(
      b, variantH, infos.fillSecondTileSizes, infos.genericSecondTileSizes,
      infos.hasLeadingEltwise, infos.hasTrailingEltwise);

  // Step 3. Rank-reduce and vectorize.
  // TODO: assumes a single func::FuncOp to transform, may need hardening.
  Value funcH = b.create<MatchOp>(variantH, func::FuncOp::getOperationName());
  funcH = buildVectorizeStrategy(b, funcH);

  // Step 4. Bufferize and drop HAL decriptor from memref ops.
  variantH = b.create<IREEBufferizeOp>(variantH, /*targetGpu=*/true);
  Value memrefFunc =
      b.create<MatchOp>(variantH, func::FuncOp::getOperationName());
  b.create<IREEEraseHALDescriptorTypeFromMemRefOp>(memrefFunc);

  // Step 5. Post-bufferization mapping to blocks and threads.
  // Need to match again since bufferize invalidated all handles.
  // TODO: assumes a single func::FuncOp to transform, may need hardening.
  funcH = b.create<MatchOp>(variantH, func::FuncOp::getOperationName());
  funcH = buildMapToBlockAndThreads(b, funcH, infos.workgroupSize);

  // Step 6. Post-bufferization vector distribution with rank-reduction.
  buildDistributeVectors(b, variantH, funcH);
}

// TODO: consider passing a problem-specific struct to control information.
static bool matchGPUReduction(linalg::LinalgOp op,
                              GPUReductionStrategyInfos &info) {
  // TODO: match the sequence the strategy supports.
  StructuredOpMatcher pattern, fill, leadingEltwise, trailingEltwise;
  makeGPUReductionMatcher(pattern, fill, leadingEltwise, trailingEltwise);
  if (!matchPattern(op, pattern)) return false;

  info.hasLeadingEltwise = leadingEltwise.getCaptured() != nullptr;
  info.hasTrailingEltwise = trailingEltwise.getCaptured() != nullptr;

  // Hardcoded workagroup size, this could be deduced from the reduction dim.
  info.workgroupSize = {32, 2, 1};
  SmallVector<unsigned> partitionedLoops =
      cast<PartitionableLoopsInterface>(op.getOperation())
          .getPartitionableLoops(kNumMaxParallelDims);
  size_t numLoops = partitionedLoops.empty() ? 0 : partitionedLoops.back() + 1;
  // Tile all the parallel dimension to 1.
  info.workgroupTileSizes.append(numLoops, 1);
  info.fillSecondTileSizes = {1, 0, 0};
  info.genericSecondTileSizes = {1, 1, 0};
  return true;
}

/// Structure to hold the parameters related to GPU reduction strategy.
struct CPUReductionStrategyInfos {
  int64_t workgroupSize;
  SmallVector<int64_t> tileSizes;
};

static bool matchCPUReduction(linalg::LinalgOp op,
                              CPUReductionStrategyInfos &infos) {
  // TODO: match the sequence the strategy supports.
  auto fill = m_StructuredOp<linalg::FillOp>();
  auto pattern = m_StructuredOp()
                     .dim(AllDims(), ShapeKind::Static)
                     .dim(-1, utils::IteratorType::reduction)
                     .output(NumEqualsTo(1))
                     .output(0, fill);

  // TODO: set the right config as expected by the strategy.
  infos.workgroupSize = 1;
  SmallVector<unsigned> partitionedLoops =
      cast<PartitionableLoopsInterface>(op.getOperation())
          .getPartitionableLoops(kNumMaxParallelDims);
  size_t numLoops = partitionedLoops.empty() ? 0 : partitionedLoops.back() + 1;
  // Tile all the parallel dimension to 1.
  infos.tileSizes.append(numLoops, 1);
  return true;
}

using StrategyBuilderFn = std::function<void(ImplicitLocOpBuilder &, Value)>;

static void createTransformRegion(func::FuncOp entryPoint,
                                  StrategyBuilderFn buildStrategy) {
  MLIRContext *ctx = entryPoint.getContext();
  Location loc = entryPoint.getLoc();
  OpBuilder b(ctx);
  b.setInsertionPointAfter(entryPoint);
  auto topLevelTransformModule = b.create<ModuleOp>(loc);
  Region &topLevelTransformRegion = topLevelTransformModule.getBodyRegion();
  b.setInsertionPointToStart(&topLevelTransformRegion.front());
  auto sequence = b.create<::transform_ext::CanonicalizedSequenceOp>(
      loc, transform::FailurePropagationMode::Propagate,
      [&](OpBuilder &b, Location loc, Value variantH) {
        ImplicitLocOpBuilder ib(loc, b);
        buildStrategy(ib, variantH);
        b.create<transform::YieldOp>(loc);
      });
  (void)sequence;
  LLVM_DEBUG(DBGS() << "transformation script:\n");
  LLVM_DEBUG(DBGS() << "verification: " << sequence.verify().succeeded()
                    << "\n");
  LLVM_DEBUG(sequence.print(DBGS()));
}

// TODO: generalize and automate over and over.
// TODO: significantly shrink this down.
static LogicalResult createReductionCpuStrategy(
    ImplicitLocOpBuilder &b, Value variantH,
    const CPUReductionStrategyInfos &info) {
  // Step 0. Fetch transform information from the config and materialize it in
  // the payload IR.
  // TODO: this still requires specific knowledge of ops present in the IR
  // and is very brittle.
  Value originalFillH =
      b.create<MatchOp>(variantH, linalg::FillOp::getOperationName());
  Value originalGenericH =
      b.create<MatchOp>(variantH, linalg::GenericOp::getOperationName());

  // Step 1: Distribute to blocks using the current IREE lowering config.
  variantH = createReductionStrategyBlockDistributionPart(
      b, variantH, originalFillH, originalGenericH, Value(),
      getAsOpFoldResult(b.getI64ArrayAttr(info.tileSizes)));

  // Step 2. Rank-reduce and buildVectorizeStrategy.
  // TODO: assumes a single func::FuncOp to transform, may need hardening.
  Value funcH = b.create<MatchOp>(variantH, func::FuncOp::getOperationName());
  funcH = buildVectorizeStrategy(b, funcH);

  // Step 3. Bufferize and drop HAL decriptor from memref ops.
  variantH = b.create<IREEBufferizeOp>(variantH, /*targetGpu=*/true);
  Value memrefFunc =
      b.create<MatchOp>(variantH, func::FuncOp::getOperationName());
  b.create<IREEEraseHALDescriptorTypeFromMemRefOp>(memrefFunc);

  // Step 4. Post-bufferization mapping to blocks only.
  // Need to match again since bufferize invalidated all handles.
  // TODO: assumes a single func::FuncOp to transform, may need hardening.
  funcH = b.create<MatchOp>(variantH, func::FuncOp::getOperationName());
  funcH = b.create<ForeachThreadToWorkgroupOp>(funcH);

  return success();
}

LogicalResult matchAndSetGPUReductionTransformStrategy(func::FuncOp entryPoint,
                                                       linalg::LinalgOp op) {
  // 1. Match
  GPUReductionStrategyInfos infos;
  if (!matchGPUReduction(op, infos)) return failure();
  auto strategyBuilder = [&](ImplicitLocOpBuilder &b, Value variant) {
    return createReductionCudaStrategy(b, variant, infos);
  };
  // 2. Add the strategy.
  createTransformRegion(entryPoint, strategyBuilder);
  return success();
}

LogicalResult matchAndSetCPUReductionTransformStrategy(func::FuncOp entryPoint,
                                                       linalg::LinalgOp op) {
  // 1. Match
  CPUReductionStrategyInfos infos;
  if (!matchCPUReduction(op, infos)) return failure();
  auto startegyBuilder = [&](ImplicitLocOpBuilder &b, Value variant) {
    return createReductionCpuStrategy(b, variant, infos);
  };
  // 2. Add the strategy.
  createTransformRegion(entryPoint, startegyBuilder);
  return success();
}

}  // namespace iree_compiler
}  // namespace mlir

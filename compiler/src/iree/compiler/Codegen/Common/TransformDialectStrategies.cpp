// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/TransformDialectStrategies.h"

#include <numeric>
#include <type_traits>

#include "iree-dialects/Dialect/LinalgTransform/StructuredTransformOpsExt.h"
#include "iree-dialects/Transforms/TransformMatchers.h"
#include "iree/compiler/Codegen/Common/TransformExtensions/CommonExtensions.h"
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
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
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

#define DEBUG_TYPE "iree-transform-builder"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

// TODO: significantly better namespacing.
using iree_compiler::IREE::transform_dialect::ApplyPatternsOp;
using iree_compiler::IREE::transform_dialect::ApplyPatternsOpPatterns;
using iree_compiler::IREE::transform_dialect::ConfigExtractPart;
using iree_compiler::IREE::transform_dialect::ForeachThreadToWorkgroupOp;
using iree_compiler::IREE::transform_dialect::IREEBufferizeOp;
using iree_compiler::IREE::transform_dialect::IREEEliminateEmptyTensorsOp;
using iree_compiler::IREE::transform_dialect::
    IREEEraseHALDescriptorTypeFromMemRefOp;
using iree_compiler::IREE::transform_dialect::
    MapNestedForeachThreadToGpuThreadsOp;
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
using transform_ext::AllDims;
using transform_ext::IsPermutation;
using transform_ext::m_StructuredOp;
using transform_ext::NumEqualsTo;
using transform_ext::ShapeKind;
using transform_ext::StructuredOpMatcher;

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
void mlir::iree_compiler::buildPrint(ImplicitLocOpBuilder &b,
                                     ValueRange handles) {
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
static iree_compiler::TileAndFuseAndDistributeResult
buildTileAndFuseAndDistributeImpl(ImplicitLocOpBuilder &b, Value rootH,
                                  ValueRange opsHToFuse,
                                  ArrayRef<OpFoldResult> tileSizesOrNumThreads,
                                  ArrayAttr threadDimMapping) {
  iree_compiler::TileAndFuseAndDistributeResult result;
  auto tileToForeachOp = b.create<TilingTransformOp>(
      rootH, tileSizesOrNumThreads, TileOrNumThreadSpec(), threadDimMapping);
  result.foreachThreadH = tileToForeachOp.getForeachThreadOp();
  result.tiledOpH = tileToForeachOp.getTiledOp();

  // Batch fusion if requested.
  if (opsHToFuse.size() > 1) {
    Value mergedOpsH =
        b.create<MergeHandlesOp>(opsHToFuse, /*deduplicate=*/true);
    b.create<FuseIntoContainingOp>(mergedOpsH, result.foreachThreadH);
  } else if (opsHToFuse.size() == 1) {
    Value fusedH = b.create<FuseIntoContainingOp>(opsHToFuse.front(),
                                                  result.foreachThreadH);
    result.resultingFusedOpsHandles.push_back(fusedH);
  }
  return result;
}

// TODO: if someone knows how to properly export templates go for it ..
// sigh.
template <typename TilingTransformOp>
static iree_compiler::TileAndFuseAndDistributeResult
buildTileFuseDistWithTileSizes(ImplicitLocOpBuilder &b, Value rootH,
                               ValueRange opsHToFuse,
                               ArrayRef<OpFoldResult> tileSizes,
                               ArrayAttr threadDimMapping) {
  return buildTileAndFuseAndDistributeImpl<TilingTransformOp,
                                           transform::TileSizesSpec>(
      b, rootH, opsHToFuse, tileSizes, threadDimMapping);
}
iree_compiler::TileAndFuseAndDistributeResult
mlir::iree_compiler::buildTileFuseDistToForeachThreadWithTileSizes(
    ImplicitLocOpBuilder &b, Value rootH, ValueRange opsHToFuse,
    ArrayRef<OpFoldResult> tileSizes, ArrayAttr threadDimMapping) {
  return buildTileFuseDistWithTileSizes<TileToForeachThreadOp>(
      b, rootH, opsHToFuse, tileSizes, threadDimMapping);
}
iree_compiler::TileAndFuseAndDistributeResult mlir::iree_compiler::
    buildTileFuseDistToForeachThreadAndWorgroupCountWithTileSizes(
        ImplicitLocOpBuilder &b, Value rootH, ValueRange opsHToFuse,
        ArrayRef<OpFoldResult> tileSizes, ArrayAttr threadDimMapping) {
  return buildTileFuseDistWithTileSizes<
      TileToForeachThreadAndWorkgroupCountRegionOp>(
      b, rootH, opsHToFuse, tileSizes, threadDimMapping);
}

/// Call buildTileAndFuseAndDistributeImpl with ArrayRef<int64_t> numThreads.
// TODO: if someone knows how to properly export templates go for it ..
// sigh.
template <typename TilingTransformOp>
static iree_compiler::TileAndFuseAndDistributeResult
buildTileFuseDistWithNumThreads(ImplicitLocOpBuilder &b, Value rootH,
                                ValueRange opsHToFuse,
                                ArrayRef<OpFoldResult> numThreads,
                                ArrayAttr threadDimMapping) {
  return buildTileAndFuseAndDistributeImpl<TilingTransformOp,
                                           transform::NumThreadsSpec>(
      b, rootH, opsHToFuse, numThreads, threadDimMapping);
}
iree_compiler::TileAndFuseAndDistributeResult
mlir::iree_compiler::buildTileFuseDistToForeachThreadWithNumThreads(
    ImplicitLocOpBuilder &b, Value rootH, ValueRange opsHToFuse,
    ArrayRef<OpFoldResult> tileSizes, ArrayAttr threadDimMapping) {
  return buildTileFuseDistWithNumThreads<TileToForeachThreadOp>(
      b, rootH, opsHToFuse, tileSizes, threadDimMapping);
}
iree_compiler::TileAndFuseAndDistributeResult mlir::iree_compiler::
    buildTileFuseDistToForeachThreadAndWorgroupCountWithNumThreads(
        ImplicitLocOpBuilder &b, Value rootH, ValueRange opsHToFuse,
        ArrayRef<OpFoldResult> tileSizes, ArrayAttr threadDimMapping) {
  return buildTileFuseDistWithNumThreads<
      TileToForeachThreadAndWorkgroupCountRegionOp>(
      b, rootH, opsHToFuse, tileSizes, threadDimMapping);
}

/// Apply patterns and vectorize (for now always applies rank-reduction).
/// Takes a handle to a func.func and returns an updated handle to a
/// func.func.
// TODO: configure patterns.
Value mlir::iree_compiler::buildVectorize(ImplicitLocOpBuilder &b,
                                          Value funcH) {
  ApplyPatternsOpPatterns patterns;
  patterns.rankReducing = true;
  funcH = b.create<ApplyPatternsOp>(funcH, patterns);
  return b.create<VectorizeOp>(funcH);
}

/// Bufferize and drop HAL descriptor from memref ops.
Value mlir::iree_compiler::buildBufferize(ImplicitLocOpBuilder &b,
                                          Value variantH, bool targetGpu) {
  Value funcH = b.create<MatchOp>(variantH, func::FuncOp::getOperationName());
  ApplyPatternsOpPatterns patterns;
  patterns.foldReassociativeReshapes = true;
  funcH = b.create<ApplyPatternsOp>(funcH, patterns);
  variantH = b.create<IREEEliminateEmptyTensorsOp>(variantH);
  variantH = b.create<IREEBufferizeOp>(variantH, /*targetGpu=*/true);
  Value memrefFunc =
      b.create<MatchOp>(variantH, func::FuncOp::getOperationName());
  b.create<IREEEraseHALDescriptorTypeFromMemRefOp>(memrefFunc);
  return variantH;
}

/// Post-bufferization mapping to blocks and threads.
/// Takes a handle to a func.func and returns an updated handle to a
/// func.func.
Value mlir::iree_compiler::buildMapToBlockAndThreads(
    ImplicitLocOpBuilder &b, Value funcH, ArrayRef<int64_t> blockSize) {
  funcH = b.create<ForeachThreadToWorkgroupOp>(funcH);
  return b.create<MapNestedForeachThreadToGpuThreadsOp>(funcH, blockSize);
}

static constexpr unsigned kCudaWarpSize = 32;

/// Post-bufferization vector distribution with rank-reduction.
/// Takes a handle to a func.func and returns an updated handle to a
/// func.func.
Value mlir::iree_compiler::buildDistributeVectors(ImplicitLocOpBuilder &b,
                                                  Value variantH, Value funcH,
                                                  int64_t warpSize) {
  ApplyPatternsOpPatterns patterns;
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

namespace {
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
  /// Handle to the original fill operation, may be null if the operation
  /// was not re-matched.
  Value originalFillH;
  /// Handle to the trailing fill operation, may be null if the operation
  /// was not re-matched.
  Value trailingEltwiseH;
};
}  // namespace

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
  ApplyPatternsOpPatterns patterns;
  patterns.bubbleCollapseExpand = true;
  b.create<ApplyPatternsOp>(funcH, patterns);
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
Value mlir::iree_compiler::createReductionStrategyBlockDistributionPart(
    ImplicitLocOpBuilder &b, Value variantH, Value originalFillH,
    Value reductionH, Value optionalFusionRootH,
    ArrayRef<OpFoldResult> tileSizes0Generic, bool hasLeadingEltwise,
    bool hasTrailingEltwise) {
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
  // `tileSizes`. If the fusion root was the reduction op, update it to be
  // the combiner op. Otherwise, fuse the combiner op into root.
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

  // The presence of leading elementwise operation implies that dispatch
  // region formation happened using another transform dialect script and
  // doesn't need the workgroup count part.
  if (hasLeadingEltwise) {
    iree_compiler::buildTileFuseDistToForeachThreadWithTileSizes(
        b, optionalFusionRootH, opsHToFuse, tileSizes0Generic,
        b.getArrayAttr({x}));
  } else {
    iree_compiler::
        buildTileFuseDistToForeachThreadAndWorgroupCountWithTileSizes(
            b, optionalFusionRootH, opsHToFuse, tileSizes0Generic,
            b.getArrayAttr({x}));
  }

  return variantH;
}

void mlir::iree_compiler::createTransformRegion(
    func::FuncOp entryPoint, StrategyBuilderFn buildStrategy) {
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

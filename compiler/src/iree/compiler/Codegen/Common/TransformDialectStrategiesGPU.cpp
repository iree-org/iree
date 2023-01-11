// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/TransformDialectStrategiesGPU.h"

#include <numeric>
#include <type_traits>

#include "iree-dialects/Dialect/LinalgTransform/StructuredTransformOpsExt.h"
#include "iree-dialects/Transforms/TransformMatchers.h"
#include "iree/compiler/Codegen/Common/TransformDialectStrategies.h"
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
using iree_compiler::IREE::transform_dialect::ConfigExtractPart;
using iree_compiler::IREE::transform_dialect::ForeachThreadToWorkgroupOp;
using iree_compiler::IREE::transform_dialect::IREEBufferizeOp;
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
using transform_ext::MatchCallbackOp;
using transform_ext::NumEqualsTo;
using transform_ext::RegisterMatchCallbacksOp;
using transform_ext::ShapeKind;
using transform_ext::StructuredOpMatcher;
using transform_ext::TakeFirstOp;

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
// Higher-level problem-specific strategy creation APIs, these should favor
// user-friendliness.
//===----------------------------------------------------------------------===//

/// Structure to hold the parameters related to GPU reduction strategy.
struct GPUReductionStrategyInfos {
  std::array<int64_t, 3> workgroupSize;
  SmallVector<int64_t> workgroupTileSizes;
  SmallVector<int64_t> fillSecondTileSizes;
  SmallVector<int64_t> genericSecondTileSizes;
  int64_t reductionDimensionSize;
};

static std::pair<Value, Value> createReductionStrategyBlockDistribution(
    ImplicitLocOpBuilder &b, Value maybeLeadingH, Value fillH, Value reductionH,
    Value maybeTrailingH) {
  auto pdlOperation = pdl::OperationType::get(b.getContext());
  auto fusionTargetSelector = b.create<TakeFirstOp>(
      pdlOperation, pdlOperation, ArrayRef<Value>{maybeTrailingH, reductionH});
  Value fusionTargetH = fusionTargetSelector.getFirst();
  Value fusionGroupH = fusionTargetSelector.getRest();
  auto blockX = mlir::gpu::GPUBlockMappingAttr::get(b.getContext(),
                                                    mlir::gpu::Blocks::DimX);
  iree_compiler::TileAndFuseAndDistributeResult tileResult = iree_compiler::
      buildTileFuseDistToForeachThreadAndWorgroupCountWithTileSizes(
          b, fusionTargetH, fusionGroupH,
          getAsOpFoldResult(b.getI64ArrayAttr({1})), b.getArrayAttr(blockX));
  Value foreachThreadH =
      b.create<FuseIntoContainingOp>(fillH, tileResult.foreachThreadH);
  foreachThreadH =
      b.create<FuseIntoContainingOp>(maybeLeadingH, foreachThreadH);
  auto gridReductionSelector = b.create<TakeFirstOp>(
      pdlOperation, pdlOperation,
      ArrayRef<Value>(
          {tileResult.resultingFusedOpsHandles.front(), tileResult.tiledOpH}));

  return std::make_pair(gridReductionSelector.getFirst(),
                        gridReductionSelector.getRest());
}

static void createReductionStrategyThreadDistribution(
    ImplicitLocOpBuilder &b, Value gridReductionH, Value maybeTiledTrailingH,
    int64_t reductionDimensionSize) {
  // Select tile sizes. Perfectly tile by:
  //   - 128 to obtain 32 threads working on vector<4xf32> when possible;
  //   - 64 to obtain 32 threads working on vector<2xf32> when possible;
  //   - 32 otherwise.
  // TODO: refine sizes based on the bitwidth of the elemental type.
  int64_t firstReductionSize = iree_compiler::kCudaWarpSize;
  int64_t vectorTileSize = 1;
  if (reductionDimensionSize % (4 * iree_compiler::kCudaWarpSize) == 0) {
    firstReductionSize = 4 * iree_compiler::kCudaWarpSize;
    vectorTileSize = 4;
  } else if (reductionDimensionSize % (2 * iree_compiler::kCudaWarpSize) == 0) {
    firstReductionSize = 2 * iree_compiler::kCudaWarpSize;
    vectorTileSize = 2;
  }

  auto pdlOperation = pdl::OperationType::get(b.getContext());
  auto threadX = mlir::gpu::GPUThreadMappingAttr::get(b.getContext(),
                                                      mlir::gpu::Threads::DimX);
  auto threadY = mlir::gpu::GPUThreadMappingAttr::get(b.getContext(),
                                                      mlir::gpu::Threads::DimY);

  // Split the reduction into a parallel and combiner part, then tile the
  // parallel part and map it to a full warp so it works on vectors.
  auto tileReduction = b.create<transform::TileReductionUsingScfOp>(
      pdlOperation, pdlOperation, pdlOperation, pdlOperation, gridReductionH,
      b.getI64ArrayAttr({0, firstReductionSize}));
  Value blockParallelFillH = tileReduction.getFillOp();
  Value blockParallelOpH = tileReduction.getSplitLinalgOp();
  Value blockCombinerOpH = tileReduction.getCombiningLinalgOp();
  iree_compiler::buildTileFuseDistToForeachThreadWithNumThreads(
      b, blockParallelOpH, {},
      getAsOpFoldResult(b.getI64ArrayAttr({0, iree_compiler::kCudaWarpSize})),
      b.getArrayAttr(threadX));

  // Tile the fill so it maps to vectors.
  // TODO: fuse once the support is available
  // (https://reviews.llvm.org/D139844).
  iree_compiler::buildTileFuseDistToForeachThreadWithTileSizes(
      b, blockParallelFillH, {},
      getAsOpFoldResult(b.getI64ArrayAttr({0, vectorTileSize})),
      b.getArrayAttr(threadX));

  // Map the combiner reduction to one thread along y so it can be mapped
  // further via predication. Fuse it into the trailing elementwise if present.
  auto selector = b.create<TakeFirstOp>(
      pdlOperation, pdlOperation,
      ArrayRef<Value>({maybeTiledTrailingH, blockCombinerOpH}));
  Value fusionRootH = selector.getFirst();
  Value fusionGroupH = selector.getRest();
  iree_compiler::buildTileFuseDistToForeachThreadWithTileSizes(
      b, fusionRootH, fusionGroupH, getAsOpFoldResult(b.getI64ArrayAttr({1})),
      b.getArrayAttr(threadY));
}

/// Builds the transform IR tiling reductions for CUDA targets. Supports
/// reductions in the last dimension with static shape divisible by 32 (CUDA
/// warp size), with optional leading and trailing elementwise operations.
static void createReductionCudaStrategy(
    ImplicitLocOpBuilder &b, Value variantH,
    const GPUReductionStrategyInfos &infos) {
  // Step 1. Call the matcher. Note that this is the same matcher as used to
  // trigger this compilation path, so it must always apply.
  b.create<RegisterMatchCallbacksOp>();
  SmallVector<Type> matchedTypes(4, pdl::OperationType::get(b.getContext()));
  auto match = b.create<MatchCallbackOp>(
      matchedTypes, "reduction", transform::FailurePropagationMode::Propagate,
      variantH);
  Value maybeLeadingH = match.getResult(0);
  Value fillH = match.getResult(1);
  Value reductionH = match.getResult(2);
  Value maybeTrailingH = match.getResult(3);

  // Step 2. Use tiling to introduce a single-iteration loop mapped to a single
  // block/workgroup. Keep everything fused.
  auto [gridReductionH, maybeTiledTrailingH] =
      createReductionStrategyBlockDistribution(b, maybeLeadingH, fillH,
                                               reductionH, maybeTrailingH);

  // Step 3. Split the reduction and tile the pieces to ensure vector
  // load/stores and mapping to a single warp with shuffles.
  createReductionStrategyThreadDistribution(
      b, gridReductionH, maybeTiledTrailingH, infos.reductionDimensionSize);

  // Step 4. Bufferize and drop HAL decriptor from memref ops.
  Value funcH = b.create<MatchOp>(variantH, func::FuncOp::getOperationName());
  funcH = iree_compiler::buildVectorize(b, funcH);
  variantH = iree_compiler::buildBufferize(b, variantH, /*targetGpu=*/true);

  // Step 5. Post-bufferization mapping to blocks and threads.
  // Need to match again since bufferize invalidated all handles.
  // TODO: assumes a single func::FuncOp to transform, may need hardening.
  funcH = b.create<MatchOp>(variantH, func::FuncOp::getOperationName());
  funcH =
      iree_compiler::buildMapToBlockAndThreads(b, funcH, infos.workgroupSize);

  // Step 6. Post-bufferization vector distribution with rank-reduction.
  iree_compiler::buildDistributeVectors(b, variantH, funcH);
}

// TODO: consider passing a problem-specific struct to control information.
static bool matchGPUReduction(linalg::LinalgOp op,
                              GPUReductionStrategyInfos &info) {
  // TODO: match the sequence the strategy supports.
  StructuredOpMatcher pattern, fill, leadingEltwise, trailingEltwise;
  makeReductionMatcher(pattern, fill, leadingEltwise, trailingEltwise,
                       info.reductionDimensionSize);
  if (!matchPattern(op, pattern)) return false;

  // Hardcoded workgroup size, this could be deduced from the reduction dim.
  info.workgroupSize = {32, 1, 1};
  SmallVector<unsigned> partitionedLoops =
      cast<iree_compiler::PartitionableLoopsInterface>(op.getOperation())
          .getPartitionableLoops(iree_compiler::kNumMaxParallelDims);
  size_t numLoops = partitionedLoops.empty() ? 0 : partitionedLoops.back() + 1;
  // Tile all the parallel dimension to 1.
  info.workgroupTileSizes.append(numLoops, 1);
  info.fillSecondTileSizes = {1, 0, 0};
  info.genericSecondTileSizes = {1, 1, 0};
  return true;
}

LogicalResult iree_compiler::matchAndSetGPUReductionTransformStrategy(
    func::FuncOp entryPoint, linalg::LinalgOp op) {
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

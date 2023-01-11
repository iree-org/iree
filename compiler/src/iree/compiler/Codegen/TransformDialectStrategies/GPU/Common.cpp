// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/TransformDialectStrategies/GPU/Common.h"

#include "iree-dialects/Dialect/LinalgTransform/StructuredTransformOpsExt.h"
#include "iree/compiler/Codegen/Common/TransformExtensions/CommonExtensions.h"
#include "iree/compiler/Codegen/LLVMGPU/TransformExtensions/LLVMGPUExtensions.h"
#include "iree/compiler/Codegen/TransformDialectStrategies/Common/Common.h"
#include "iree/compiler/Codegen/TransformDialectStrategies/GPU/AbstractReductionStrategy.h"
#include "iree/compiler/Codegen/TransformDialectStrategies/GPU/SmallReductionStrategy.h"
#include "iree/compiler/Codegen/TransformDialectStrategies/GPU/StagedReductionStrategy.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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

using iree_compiler::buildReductionStrategyBlockDistribution;
using iree_compiler::maxDivisorOfValueBelowLimit;
using iree_compiler::gpu::AbstractReductionStrategy;
using iree_compiler::gpu::build1DSplittingStrategyWithOptionalThreadMapping;
using iree_compiler::gpu::buildCommonTrailingStrategy;
using iree_compiler::gpu::buildGpuSmallReductionStrategy;
using iree_compiler::gpu::buildMapToBlockAndThreads;
using iree_compiler::gpu::buildStagedReductionStrategy;
using iree_compiler::gpu::getSmallReductionConfig;
using iree_compiler::gpu::kCudaMaxNumThreads;
using iree_compiler::gpu::kCudaMaxVectorLoadBitWidth;
using iree_compiler::gpu::kCudaWarpSize;
using iree_compiler::gpu::ReductionConfig;
using iree_compiler::gpu::SmallReductionStrategy;
using iree_compiler::gpu::StagedReductionStrategy;

//===----------------------------------------------------------------------===//
// General helpers.
//===----------------------------------------------------------------------===//

/// Return max(1, (value * 32) / bitwidth).
int64_t mlir::iree_compiler::gpu::scaleUpByBitWidth(int64_t value,
                                                    int64_t bitWidth) {
  assert((bitWidth & bitWidth - 1) == 0 && "bitWidth must be a power of 2");
  return std::max((value * 32) / bitWidth, int64_t(1));
}

/// Adjust the number of warps to use to benefit from packing multiple smaller
/// elemental types within a single 128 bit shuffled element.
int64_t mlir::iree_compiler::gpu::adjustNumberOfWarpsForBlockShuffle(
    int64_t numWarpsToUse, int64_t bitWidth) {
  // Try to scale down the number of warps to use 32b elements in warp shuffles.
  assert((bitWidth & bitWidth - 1) == 0 && "bitWidth must be a power of 2");
  int64_t factor;
  for (factor = scaleUpByBitWidth(1, bitWidth); factor > 1; factor >>= 1)
    if (numWarpsToUse % factor == 0) break;
  numWarpsToUse /= factor;
  // Try to scale to using 128b elements in warp shuffles.
  return std::max(numWarpsToUse / 4, int64_t(1));
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
  if (ShapedType::isDynamic(upperBound)) {
    return std::make_pair(int64_t(0), int64_t(1));
  }
  for (int64_t vectorSize = maxVectorSize; vectorSize >= 1; vectorSize >>= 1) {
    int64_t splitPoint =
        iree_compiler::previousMultipleOf(upperBound, minMultiple * vectorSize);
    if (splitPoint > 0) {
      return (upperBound == splitPoint)
                 ? std::make_pair(int64_t(0), vectorSize)
                 : std::make_pair(splitPoint, vectorSize);
    }
  }
  return std::make_pair(int64_t(0), int64_t(1));
}

//===----------------------------------------------------------------------===//
// Low-level reusable retargetable builder APIs, follow MLIR-style builders.
//===----------------------------------------------------------------------===//
/// Post-bufferization mapping to blocks and threads.
/// Takes a handle to a func.func and returns an updated handle to a
/// func.func.
Value mlir::iree_compiler::gpu::buildMapToBlockAndThreads(
    ImplicitLocOpBuilder &b, Value funcH, ArrayRef<int64_t> blockSize) {
  funcH = b.create<ForeachThreadToWorkgroupOp>(funcH);
  return b.create<MapNestedForeachThreadToGpuThreadsOp>(funcH, blockSize);
}

/// Post-bufferization vector distribution with rank-reduction.
/// Takes a handle to a func.func and returns an updated handle to a
/// func.func.
Value mlir::iree_compiler::gpu::buildDistributeVectors(ImplicitLocOpBuilder &b,
                                                       Value variantH,
                                                       Value funcH,
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
// Mid-level problem-specific strategy builder APIs, follow MLIR-style builders.
//===----------------------------------------------------------------------===//
void mlir::iree_compiler::gpu::
    build1DSplittingStrategyWithOptionalThreadMapping(
        ImplicitLocOpBuilder &b, Value elementwiseH, int64_t rank,
        SmallVector<int64_t> elementwiseSizes, int64_t numThreads,
        int64_t maxVectorSize) {
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

/// Take care of the last common steps in a GPU strategy (i.e. vectorize,
/// bufferize, maps to blocks and threads and distribute vectors).
/// Return the handles to the updated variant and the func::FuncOp ops under
/// the variant op.
std::pair<Value, Value> mlir::iree_compiler::gpu::buildCommonTrailingStrategy(
    ImplicitLocOpBuilder &b, Value variantH,
    const AbstractReductionStrategy &strategy) {
  // Step N-1. Bufferize and drop HAL descriptor from memref ops.
  Value funcH = b.create<MatchOp>(variantH, func::FuncOp::getOperationName());
  funcH = iree_compiler::buildVectorize(b, funcH);
  variantH = iree_compiler::buildBufferize(b, variantH,
                                           /*targetGpu=*/true);

  // Step N. Post-bufferization mapping to blocks and threads.
  // Need to match again since bufferize invalidated all handles.
  // TODO: assumes a single func::FuncOp to transform, may need hardening.
  funcH = b.create<MatchOp>(variantH, func::FuncOp::getOperationName());
  funcH = buildMapToBlockAndThreads(b, funcH, strategy.getNumThreadsInBlock());

  return std::make_pair(variantH, funcH);
}

//===----------------------------------------------------------------------===//
// Higher-level problem-specific strategy creation APIs, these should favor
// user-friendliness.
//===----------------------------------------------------------------------===//
/// Return success if the IR matches what the GPU reduction strategy can
/// handle. If it is success it will append the transform dialect after the
/// entry point module.
LogicalResult mlir::iree_compiler::gpu::matchAndSetReductionStrategy(
    func::FuncOp entryPoint, linalg::LinalgOp op) {
  // 1. Match a reduction and surrounding ops.
  StructuredOpMatcher reduction, fill, leading, trailing;
  transform_ext::MatchedReductionCaptures captures;
  makeReductionMatcher(reduction, fill, leading, trailing, captures);
  if (!matchPattern(op, reduction)) return failure();

  // 2. Construct the configuration and the strategy builder.
  // TODO: Generalize along the HW axis.
  auto strategyBuilder = [&](ImplicitLocOpBuilder &b, Value variant) {
    // Take the small strategy if it is deemed profitable.
    auto maybeSmallStrategy =
        SmallReductionStrategy::create(op->getContext(), captures);
    if (succeeded(maybeSmallStrategy))
      return buildGpuSmallReductionStrategy(b, variant, *maybeSmallStrategy);
    // Otherwise, always fallback to the staged strategy.
    auto strategy = StagedReductionStrategy::create(op->getContext(), captures);
    return buildStagedReductionStrategy(b, variant, strategy);
  };

  // 3. Build strategy embedded into the IR.
  createTransformRegion(entryPoint, strategyBuilder);

  return success();
}

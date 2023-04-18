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
#include "llvm/Support/ErrorHandling.h"
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
using iree_compiler::IREE::transform_dialect::ForallToWorkgroupOp;
using iree_compiler::IREE::transform_dialect::MapNestedForallToGpuThreadsOp;
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
using iree_compiler::gpu::buildMapToBlockAndThreads;
using iree_compiler::gpu::buildSmallReductionStrategy;
using iree_compiler::gpu::buildStagedReductionStrategy;
using iree_compiler::gpu::GPUModel;
using iree_compiler::gpu::kCudaMaxNumThreads;
using iree_compiler::gpu::kCudaMaxVectorLoadBitWidth;
using iree_compiler::gpu::kCudaWarpSize;
using iree_compiler::gpu::ReductionConfig;
using iree_compiler::gpu::ReductionStrategy;
using iree_compiler::gpu::scaleUpByBitWidth;
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
/// is a multiple of `fixedSize * vectorSize`.
/// The returned `vectorSize` is the maximal power of `2`, smaller than
/// `maxVectorSize`, for which `splitPoint` can be computed.
///
/// Note: `vectorSize` may be smaller than `maxVectorSize` when the upperBound
/// is small enough. In such cases we give preference to keeping the `fixedSize`
/// parameter unchanged and reducing the `vectorSize`. `fixedSize` generally
/// captures the number of threads and we do not alter decisions on parallelism
/// at this level.
///
/// If such a positive multiple exists:
///   1. if it is `upperBound`, then `upperBound` is an even multiple of
///      `fixedSize` * `vectorSize` and we can tile evenly without splitting.
///      In this case we return (0, vectorSize).
///   2. otherwise, it is a split point at which we can split with vectorSize
///      to obtain the largest divisible tiling.
///      In this case we return (splitPoint, vectorSize).
/// Otherwise we return (0, 1) to signify no splitting and a vector size of 1.
// TODO: support the dynamic case, taking future stride and alignment into
// account and returning Values. The op then needs to become part of the
// transform dialect.
static std::pair<int64_t, int64_t> computeSplitPoint(int64_t upperBound,
                                                     int64_t fixedSize,
                                                     int64_t maxVectorSize) {
  assert((maxVectorSize & (maxVectorSize - 1)) == 0 && "must be a power of 2");
  if (ShapedType::isDynamic(upperBound)) {
    return std::make_pair(int64_t(0), int64_t(1));
  }
  for (int64_t vectorSize = maxVectorSize; vectorSize >= 1; vectorSize >>= 1) {
    int64_t splitPoint =
        iree_compiler::previousMultipleOf(upperBound, fixedSize * vectorSize);
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
    ImplicitLocOpBuilder &b, Value funcH, ArrayRef<int64_t> blockSize,
    ArrayRef<int64_t> warpDims) {
  b.create<ForallToWorkgroupOp>(funcH);
  b.create<MapNestedForallToGpuThreadsOp>(funcH, blockSize, warpDims);
  return funcH;
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
  patterns.rankReducingVector = true;
  b.create<ApplyPatternsOp>(funcH, patterns);
  Value ifH = b.create<MatchOp>(funcH, scf::IfOp::getOperationName());
  // Locally suppress failures for this op only because it doesn't cover the
  // `threadIdx.x == 0 && threadIdx.y == 0` case at the moment.
  auto sequence = b.create<SequenceOp>(
      TypeRange(), transform::FailurePropagationMode::Suppress, variantH,
      /*extraBindings=*/ValueRange());
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
        ImplicitLocOpBuilder &b, Value isolatedParentOpH, Value opH,
        int64_t rank, int64_t mostMinorDim, SmallVector<int64_t> opSizes,
        int64_t numThreads, Attribute mappingAttr, int64_t maxVectorSize) {
  // Poor man's handling of optionality in C++. Will need to be converted to
  // proper transform dialect filters or handling of emptiness.
  if (rank == 0) return;

  // Compute split point to guarantee we form a maximal chunk divisible by
  // numThreads * vectorSize.
  // This chunk is currently not aligned for proper vector accesses.
  // In the future, this can be solved either by:
  //   1. doing an extra prologue split that is cognizant of the future stride.
  //   2. or, aligning allocations to a multiple of 128b on the most minor
  //      dimensions but without changing problem sizes (i.e. poor man's
  //      packing).
  int64_t mostMinorSize = opSizes[mostMinorDim];
  auto [splitPoint, vectorSize] = computeSplitPoint(
      /*upperBound=*/mostMinorSize, /*fixedSize=*/numThreads,
      /*maxVectorSize=*/maxVectorSize);

  // Create 1-D tile sizes for the first, divisible, part.
  SmallVector<int64_t> scfForTileSizes(rank, 0), foreachTileSizes(rank, 0);
  scfForTileSizes[mostMinorDim] = numThreads * vectorSize;
  foreachTileSizes[mostMinorDim] = numThreads;

  // Split, tile and map the most minor dimension to `mappingAttr`.
  if (splitPoint > 0) {
    auto pdlOperation = pdl::OperationType::get(b.getContext());
    auto split = b.create<transform::SplitOp>(
        pdlOperation, pdlOperation, opH, b.getI64IntegerAttr(mostMinorDim),
        Value(), b.getI64IntegerAttr(splitPoint));
    opH = split.getFirst();
    if (vectorSize > 1) {
      auto res = iree_compiler::buildTileFuseToScfFor(
          /*b=*/b,
          /*isolatedParentOpH=*/isolatedParentOpH,
          /*rootH=*/opH,
          /*opsHToFuse=*/{},
          /*tileSizes=*/
          getAsOpFoldResult(b.getI64ArrayAttr({scfForTileSizes})));
      opH = res.tiledOpH;
      // Reset the vector size to 1 for the tail, which is known to not be
      // divisible by `numThreads * vectorSize`.
      vectorSize = 1;
    }
    if (numThreads > 1) {
      assert(mappingAttr && "must specify a mapping attribute");
      iree_compiler::buildTileFuseDistToForallWithNumThreads(
          /*b=*/b,
          /*isolatedParentOpH=*/isolatedParentOpH,
          /*rootH=*/opH,
          /*opsHToFuse=*/{},
          /*numThreads=*/getAsOpFoldResult(b.getI64ArrayAttr(foreachTileSizes)),
          /*threadDimMapping=*/b.getArrayAttr({mappingAttr}));
    }
    opH = split.getSecond();
  }

  // Tile and map the most minor dimension of the remainder to mappingAttr.
  if (vectorSize > 1) {
    auto res = iree_compiler::buildTileFuseToScfFor(
        /*b=*/b,
        /*isolatedParentOpH=*/isolatedParentOpH,
        /*rootH=*/opH,
        /*opsHToFuse=*/{},
        /*tileSizes=*/getAsOpFoldResult(b.getI64ArrayAttr({scfForTileSizes})));
    opH = res.tiledOpH;
  }
  if (numThreads > 1) {
    assert(mappingAttr && "must specify a mapping attribute");
    iree_compiler::buildTileFuseDistToForallWithNumThreads(
        /*b=*/b,
        /*isolatedParentOpH=*/isolatedParentOpH,
        /*rootH=*/opH,
        /*opsHToFuse=*/{},
        /*numThreads=*/getAsOpFoldResult(b.getI64ArrayAttr(foreachTileSizes)),
        /*threadDimMapping=*/b.getArrayAttr({mappingAttr}));
  }
}

/// Take care of the last common steps in a GPU strategy (i.e. vectorize,
/// bufferize, maps to blocks and threads and distribute vectors).
/// Return the handles to the updated variant and the func::FuncOp ops under
/// the variant op.
std::pair<Value, Value> mlir::iree_compiler::gpu::buildCommonTrailingStrategy(
    ImplicitLocOpBuilder &b, Value variantH,
    const AbstractReductionStrategy &strategy) {
  Value funcH = b.create<MatchOp>(variantH, func::FuncOp::getOperationName());

  // Step N-5. Fold tensor.empty to avoid large allocations.
  ApplyPatternsOpPatterns configuration;
  configuration.foldTensorEmptyExtract = true;

  // Step N-4. Perform a pass of canonicalization + enabling after tiling.
  funcH = mlir::iree_compiler::buildCanonicalizationAndEnablingTransforms(
      b, configuration, funcH);
  funcH = iree_compiler::buildVectorize(b, funcH);

  // Step N-3. Perform a pass of canonicalization + enabling after vectorization
  // as well as hoisting subset operations such as vector.transfer_read/write.
  funcH = mlir::iree_compiler::buildCanonicalizationAndEnablingTransforms(
      b, configuration, funcH);
  funcH = iree_compiler::buildHoisting(b, funcH);

  // Step N-2. Bufferize and drop HAL descriptor from memref ops.
  variantH = iree_compiler::buildBufferize(b, variantH, /*targetGpu=*/true);

  // Step N-1. Post-bufferization mapping to blocks and threads.
  // Need to match again since bufferize invalidated all handles.
  // TODO: assumes a single func::FuncOp to transform, may need hardening.
  funcH = b.create<MatchOp>(variantH, func::FuncOp::getOperationName());
  funcH = buildMapToBlockAndThreads(b, funcH, strategy.getNumThreadsInBlock());

  // Step N. Perform a final pass of canonicalization + enabling before
  // returning.
  variantH = mlir::iree_compiler::buildCanonicalizationAndEnablingTransforms(
      b, configuration, variantH);
  return std::make_pair(variantH, funcH);
}

//===----------------------------------------------------------------------===//
// Higher-level problem-specific strategy creation APIs, these should favor
// user-friendliness.
//===----------------------------------------------------------------------===//

/// Placeholder to encode fixed reductions that should take finer-grained
/// precedence over other heuristics. In the future, this could be lifted to
/// e.g. `gpuModel` or higher up in some transform dialect database summary of
/// "known good things".
static FailureOr<ReductionConfig> applyKnownGoodReductionConfigurations(
    const transform_ext::MatchedReductionCaptures &captures,
    const GPUModel &gpuModel) {
  auto staged = ReductionStrategy::Staged;
  int64_t reductionSize = captures.reductionOpSizes.back();
  if (gpuModel.model == GPUModel::kDefaultGPU) {
    if (captures.reductionOutputElementalTypeBitWidth == 32) {
      if (reductionSize == 64)
        return ReductionConfig{/*maxNumThreads=*/64, /*vectorSize=*/1, staged};
      if (reductionSize == 128)
        return ReductionConfig{/*maxNumThreads=*/32, /*vectorSize=*/4, staged};
      if (reductionSize == 512)
        return ReductionConfig{/*maxNumThreads=*/256, /*vectorSize=*/2, staged};
    }
  }
  return failure();
}

/// The configurations below have been determined empirically by performing a
/// manual tradeoff between problem size, amount of parallelism and vector
/// size on a particular NVIDIA RTX2080Ti 12GB card. This is a coarse tradeoff
/// that should generally give reasonably good results but that begs to be
/// complemented by hardcoded known good configurations and ultimately a
/// database and/or a random forest compression of configurations with
/// guaranteed performance.
// TODO: Lift some of the strategy sizing logic as hints and/or heuristics to
// also work properly in the dynamic case.
// TODO: Support more HW configs and make it more pluggable.
static ReductionConfig getReductionConfig(
    const transform_ext::MatchedReductionCaptures &captures,
    const GPUModel &gpuModel) {
  auto maybeHardcodedConfiguration =
      applyKnownGoodReductionConfigurations(captures, gpuModel);
  if (succeeded(maybeHardcodedConfiguration))
    return *maybeHardcodedConfiguration;

  //===--------------------------------------------------------------------===//
  // Small reduction strategy.
  //===--------------------------------------------------------------------===//
  // Dynamic reductions are never supported by default because we can
  // never know offhand whether we are in a small-reduction regime mode.
  // Since this mode does not coalesce reads, perf will suffer
  // catastrophically on larger runtime reduction.
  // TODO: explicit hint from above that we really want to do that.
  int64_t redSize = captures.reductionOpSizes.back();
  bool isDynamicReduction = ShapedType::isDynamic(redSize);
  // Otherwise, still only support the small cases for now and fall back to
  // other strategies otherwise.
  bool isSmallReduction = (redSize < 2 * kCudaWarpSize);
  if (!isDynamicReduction && isSmallReduction) {
    int64_t maxNumThreads = 4 * kCudaWarpSize;
    return ReductionConfig{maxNumThreads, 0, ReductionStrategy::Small};
  }

  //===--------------------------------------------------------------------===//
  // Staged reduction strategy.
  //===--------------------------------------------------------------------===//
  int64_t bitWidth = captures.reductionOutputElementalTypeBitWidth;
  int64_t vectorSize = scaleUpByBitWidth(4, bitWidth);
  int64_t maxNumThreads = 8 * kCudaWarpSize;
  // No adjustments in the dynamic case, we need extra information to make a
  // good decision.
  if (ShapedType::isDynamic(redSize))
    return ReductionConfig{maxNumThreads, vectorSize,
                           ReductionStrategy::Staged};
  // Scale down to smaller sizes (4, 8, 16)-warps.
  if (scaleUpByBitWidth(redSize, bitWidth) <= 4 * kCudaWarpSize) {
    vectorSize = scaleUpByBitWidth(1, bitWidth);
    maxNumThreads = 4 * kCudaWarpSize;
  } else if (scaleUpByBitWidth(redSize, bitWidth) <= 8 * kCudaWarpSize) {
    vectorSize = scaleUpByBitWidth(2, bitWidth);
    maxNumThreads = 4 * kCudaWarpSize;
  } else if (scaleUpByBitWidth(redSize, bitWidth) <= 8 * 2 * kCudaWarpSize) {
    vectorSize = scaleUpByBitWidth(4, bitWidth);
    maxNumThreads = 4 * kCudaWarpSize;
  }
  // Scale up to larger sizes (32, 64, 128+)-warps, using vector-4.
  if (!captures.trailingOpSizes.empty()) {
    if (scaleUpByBitWidth(redSize, bitWidth) >= 128 * 4 * kCudaWarpSize) {
      vectorSize = scaleUpByBitWidth(4, bitWidth);
      maxNumThreads = 32 * kCudaWarpSize;
    } else if (scaleUpByBitWidth(redSize, bitWidth) >= 64 * 4 * kCudaWarpSize) {
      vectorSize = scaleUpByBitWidth(4, bitWidth);
      maxNumThreads = 16 * kCudaWarpSize;
    } else if (scaleUpByBitWidth(redSize, bitWidth) >= 32 * 4 * kCudaWarpSize) {
      vectorSize = scaleUpByBitWidth(4, bitWidth);
      maxNumThreads = 8 * kCudaWarpSize;
    } else if (scaleUpByBitWidth(redSize, bitWidth) >= 16 * 4 * kCudaWarpSize) {
      vectorSize = scaleUpByBitWidth(4, bitWidth);
      maxNumThreads = 4 * kCudaWarpSize;
    }
  }
  return ReductionConfig{maxNumThreads, vectorSize, ReductionStrategy::Staged};
}

/// Map an N-D parallel, 1-D reduction operation with optional leading and
/// optional trailing elementwise operations.
/// The 1-D reduction dimension must be in the most minor dimension.
/// The innermost dimensions of the leading and trailing operations must be
/// most minor along all accesses. Return failure if matching fails. On a
/// successful match, configure a reduction strategy based on a proxy model of
/// the hardware and construct transform dialect IR that implements the
/// reduction strategy. The transform dialect IR is added in a top-level
/// ModuleOp after the `entryPoint` func::FuncOp.
static LogicalResult matchAndSetReductionStrategy(func::FuncOp entryPoint,
                                                  linalg::LinalgOp op,
                                                  const GPUModel &gpuModel) {
  if (!gpuModel.hasWarpShuffle) return failure();
  // 1. Match a reduction and surrounding ops.
  StructuredOpMatcher *reduction;
  transform_ext::MatchedReductionCaptures captures;
  transform_ext::MatcherContext matcherContext;
  makeReductionMatcher(matcherContext, reduction, captures);
  if (!matchPattern(op, *reduction)) return failure();

  // 2. Construct the configuration and the strategy builder.
  // TODO: Generalize along the HW axis.
  auto strategyBuilder = [&](ImplicitLocOpBuilder &b, Value variant) {
    ReductionConfig reductionConfig = getReductionConfig(captures, gpuModel);
    if (reductionConfig.strategy == ReductionStrategy::Small) {
      auto strategy = SmallReductionStrategy::create(op->getContext(), captures,
                                                     reductionConfig);
      return buildSmallReductionStrategy(b, variant, strategy);
    } else if (reductionConfig.strategy == ReductionStrategy::Staged) {
      // Otherwise, always fallback to the staged strategy.
      auto strategy = StagedReductionStrategy::create(
          op->getContext(), captures, reductionConfig);
      return buildStagedReductionStrategy(b, variant, strategy);
    } else {
      return llvm_unreachable("Unknown strategy");
    }
  };

  // 3. Build strategy embedded into the IR.
  mlir::iree_compiler::createTransformRegion(entryPoint, strategyBuilder);

  return success();
}

LogicalResult mlir::iree_compiler::gpu::matchAndSetTransformStrategy(
    func::FuncOp entryPoint, Operation *op, const GPUModel &gpuModel) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp) return failure();
  if (succeeded(matchAndSetReductionStrategy(entryPoint, linalgOp, gpuModel)))
    return success();
  // TODO: Add more transform dialect strategy for other kind of dispatch
  // regions.
  return failure();
}

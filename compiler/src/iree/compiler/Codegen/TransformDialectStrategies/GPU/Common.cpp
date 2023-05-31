// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/TransformDialectStrategies/GPU/Common.h"

#include <tuple>

#include "iree-dialects/Dialect/LinalgTransform/StructuredTransformOpsExt.h"
#include "iree/compiler/Codegen/Common/TransformExtensions/CommonExtensions.h"
#include "iree/compiler/Codegen/LLVMGPU/TransformExtensions/LLVMGPUExtensions.h"
#include "iree/compiler/Codegen/TransformDialectStrategies/Common/Common.h"
#include "iree/compiler/Codegen/TransformDialectStrategies/GPU/MatmulTensorCoreStrategy.h"
#include "iree/compiler/Codegen/TransformDialectStrategies/GPU/SmallReductionStrategy.h"
#include "iree/compiler/Codegen/TransformDialectStrategies/GPU/StagedReductionStrategy.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/TransformOps/MemRefTransformOps.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/TransformOps/SCFTransformOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/TransformOps/VectorTransformOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;

#define DEBUG_TYPE "iree-transform-builder"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << '[' << DEBUG_TYPE << "] " << X)

llvm::cl::opt<bool> clGPUEnableTransformDialectMatmulTensorCoreStrategy(
    "iree-codegen-llvmgpu-enable-transform-dialect-matmul-tensorcore-strategy",
    llvm::cl::desc("activate the matmul tensorcore strategy"),
    llvm::cl::init(true));

// TODO: significantly better namespacing.
using iree_compiler::IREE::transform_dialect::ApplyPatternsOp;
using iree_compiler::IREE::transform_dialect::ApplyPatternsOpPatterns;
using iree_compiler::IREE::transform_dialect::ForallToWorkgroupOp;
using iree_compiler::IREE::transform_dialect::MapNestedForallToGpuThreadsOp;
using iree_compiler::IREE::transform_dialect::VectorToWarpExecuteOnLane0Op;
using iree_compiler::IREE::transform_dialect::VectorWarpDistributionOp;

using iree_compiler::buildReductionStrategyBlockDistribution;
using iree_compiler::buildTileFuseDistToForallWithNumThreads;
using iree_compiler::buildTileFuseDistToForallWithTileSizes;
using iree_compiler::maxDivisorOfValueBelowLimit;
using iree_compiler::TileToForallAndFuseAndDistributeResult;
using iree_compiler::gpu::AbstractGemmLikeStrategy;
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
using iree_compiler::IREE::transform_dialect::ApplyBufferOptimizationsOp;
using iree_compiler::IREE::transform_dialect::IREEBufferizeOp;
using iree_compiler::IREE::transform_dialect::IREEEliminateEmptyTensorsOp;
using iree_compiler::IREE::transform_dialect::
    IREEEraseHALDescriptorTypeFromMemRefOp;
using iree_compiler::IREE::transform_dialect::
    IREEPopulateWorkgroupCountRegionUsingNumThreadsSliceOp;
using iree_compiler::IREE::transform_dialect::ShareForallOperandsOp;
using transform::FuseIntoContainingOp;
using transform::MatchOp;
using transform::ScalarizeOp;
using transform::SequenceOp;
using transform_ext::MatchCallbackOp;
using transform_ext::RegisterMatchCallbacksOp;
using transform_ext::StructuredOpMatcher;

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
    ArrayRef<int64_t> numThreadsInBlock) {
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
  iree_compiler::buildHoisting(b, funcH);

  // Step N-2. Bufferize and drop HAL descriptor from memref ops.
  variantH = iree_compiler::buildBufferize(b, variantH, /*targetGpu=*/true);

  // Step N-1. Post-bufferization mapping to blocks and threads.
  // Need to match again since bufferize invalidated all handles.
  // TODO: assumes a single func::FuncOp to transform, may need hardening.
  funcH = b.create<MatchOp>(variantH, func::FuncOp::getOperationName());
  funcH = buildMapToBlockAndThreads(b, funcH, numThreadsInBlock);

  // Step N. Perform a final pass of canonicalization + enabling before
  // returning.
  variantH = mlir::iree_compiler::buildCanonicalizationAndEnablingTransforms(
      b, configuration, variantH);
  return std::make_pair(variantH, funcH);
}

//===----------------------------------------------------------------------===//
// Subset of mid-level builders currently used for GEMM-like problems.
//===----------------------------------------------------------------------===//

/// Key function for vtable.
AbstractGemmLikeStrategy::~AbstractGemmLikeStrategy() {}

/// Build transform IR to hoist the padded output operand of a padded matmul.
/// Additionally, this attempts to fold the padding into the producing fill, if
/// available.
Value mlir::iree_compiler::gpu::buildHoistOutputPaddingOp(
    ImplicitLocOpBuilder &b, Value variantH, Value paddedMatmulOpH,
    int64_t numLoopsToHoist) {
  // Find the output pad and hoist it.
  // TODO: don't hardcode output operand number.
  // TODO: Better builders.
  Value outputH = b.create<transform::GetProducerOfOperand>(
      paddedMatmulOpH.getType(), paddedMatmulOpH, b.getI64IntegerAttr(2));

  // Hoist the padding above the 1 innermost reduction loop.
  auto padOpType = transform::OperationType::get(
      b.getContext(), tensor::PadOp::getOperationName());
  outputH = b.create<transform::CastOp>(padOpType, outputH);
  b.create<transform::HoistPadOp>(paddedMatmulOpH.getType(), outputH,
                                  numLoopsToHoist);

  // Perform a pass of canonicalization cleanups + folding fill + pad into pad
  // by applying `foldTensorSubsets` and `tilingCanonicalization`.
  {
    ApplyPatternsOpPatterns config;
    config.foldTensorSubsets = true;
    config.tilingCanonicalization = true;
    iree_compiler::buildCanonicalizationAndEnablingTransforms(b, config,
                                                              variantH);
  }

  // The canonicalization above should have rewritten hoistPad into a FillOp.
  // Unfortunately, the listener drops handles if the op types don't match. We
  // need better behavior here, for now we rematch.
  // TODO: use value handles.
  Value fillOpH = b.create<transform::MatchOp>(
      variantH, linalg::FillOp::getOperationName());

  return fillOpH;
}

/// Helper function to distribute one pad or copy operation.
/// Note: When `foldIfBranch` is true, one must later perform masked
/// vectorization of the result.
/// This amounts to injecting knowledge about future transformations without
/// adding leaky semantics.
std::tuple<Value, Value>
mlir::iree_compiler::gpu::buildDistributeOnePadOrCopyWithTileSizes(
    ImplicitLocOpBuilder &b, Value variantH, Value copyOpH,
    ArrayRef<int64_t> tileSizes, ArrayRef<Attribute> threadDimMapping,
    bool foldIfBranch) {
  TileToForallAndFuseAndDistributeResult res =
      buildTileFuseDistToForallWithTileSizes(
          /*builder=*/b,
          /*isolatedParentOpH=*/variantH,
          /*rootH=*/copyOpH,
          /*opsToFuseH=*/{},
          /*tileSizes=*/
          getAsOpFoldResult(b.getI64ArrayAttr(tileSizes)),
          /*threadDimMapping=*/
          b.getArrayAttr(threadDimMapping));
  if (foldIfBranch) {
    Value ifOpH = b.create<transform::MatchOp>(res.forallH,
                                               scf::IfOp::getOperationName());
    b.create<transform::TakeAssumedBranchOp>(
        ifOpH, /*takeElseBranch=*/b.getUnitAttr());
  }
  return std::make_tuple(res.tiledOpH, res.forallH);
}

/// Helper function to distribute one pad or copy operation.
/// Note: When `foldIfBranch` is true, one must later perform masked
/// vectorization of the result.
/// This amounts to injecting knowledge about future transformations without
/// adding leaky semantics.
Value mlir::iree_compiler::gpu::buildDistributeOnePadOrCopyWithNumThreads(
    ImplicitLocOpBuilder &b, Value variantH, Value copyOpH,
    ArrayRef<int64_t> numThreads, ArrayRef<Attribute> threadDimMapping,
    bool foldIfBranch) {
  TileToForallAndFuseAndDistributeResult res =
      buildTileFuseDistToForallWithNumThreads(
          /*builder=*/b,
          /*isolatedParentOpH=*/variantH,
          /*rootH=*/copyOpH,
          /*opsToFuseH=*/{},
          /*numThreads=*/
          getAsOpFoldResult(b.getI64ArrayAttr(numThreads)),
          /*threadDimMapping=*/
          b.getArrayAttr(threadDimMapping));
  if (foldIfBranch) {
    Value ifOpH = b.create<transform::MatchOp>(res.forallH,
                                               scf::IfOp::getOperationName());
    b.create<transform::TakeAssumedBranchOp>(
        ifOpH, /*takeElseBranch=*/b.getUnitAttr());
  }
  return res.tiledOpH;
}

/// Distribute the explicit copies involved in a matmul operation
/// `paddedMatmulOpH`.
std::tuple<Value, Value, Value>
mlir::iree_compiler::gpu::buildDistributeMatmulCopies(
    ImplicitLocOpBuilder &b, Value variantH, Value paddedMatmulOpH,
    const AbstractGemmLikeStrategy &strategy) {
  // Explicitly materialize the parent parallel_insert into a copy to avoid late
  // bufferization interferences.
  // TODO: Avoid brittle rematching.
  Value insertSliceH = b.create<transform::MatchOp>(
      variantH, tensor::ParallelInsertSliceOp::getOperationName());
  Value copyBackOpH = b.create<transform::InsertSliceToCopyOp>(
      insertSliceH.getType(), insertSliceH);

  Value lhsH = b.create<transform::GetProducerOfOperand>(
      paddedMatmulOpH.getType(), paddedMatmulOpH, b.getI64IntegerAttr(0));
  Value rhsH = b.create<transform::GetProducerOfOperand>(
      paddedMatmulOpH.getType(), paddedMatmulOpH, b.getI64IntegerAttr(1));

  AbstractGemmLikeStrategy::MappingInfo lhsCopyMapping =
      strategy.lhsCopyMapping();
  Value lhsCopyOpH = buildDistributeOnePadOrCopyWithNumThreads(
      b, variantH, lhsH, /*numThreads=*/lhsCopyMapping.numThreads,
      /*threadDimMapping=*/lhsCopyMapping.threadMapping, /*foldIfBranch=*/true);

  AbstractGemmLikeStrategy::MappingInfo rhsCopyMapping =
      strategy.rhsCopyMapping();
  Value rhsCopyOpH = buildDistributeOnePadOrCopyWithNumThreads(
      b, variantH, rhsH, /*numThreads=*/rhsCopyMapping.numThreads,
      /*threadDimMapping=*/rhsCopyMapping.threadMapping, /*foldIfBranch=*/true);

  AbstractGemmLikeStrategy::MappingInfo resCopyMapping =
      strategy.resCopyMapping();
  copyBackOpH = buildDistributeOnePadOrCopyWithNumThreads(
      b, variantH, copyBackOpH,
      /*numThreads=*/resCopyMapping.numThreads,
      /*threadDimMapping=*/rhsCopyMapping.threadMapping);

  return std::make_tuple(lhsCopyOpH, rhsCopyOpH, copyBackOpH);
}

/// Specific pattern to perform masked vectorization of copies give as
/// parameters, cleanup and vectorize the rest.
// TODO: generalize and don't hardcode.
void mlir::iree_compiler::gpu::buildMatmulVectorization(
    ImplicitLocOpBuilder &b, Value variantH, Value lhsCopyOpH, Value rhsCopyOpH,
    Value copyBackOpH, const AbstractGemmLikeStrategy &strategy) {
  // Canonicalize to make padOp outputs static shaped: this is currently a
  // prerequisite for vector masking.
  // Also, no canonicalization is allowed after vector masking and before we
  // lower the masks: masks are currently quite brittle and do not like
  // canonicalization or anything else that may insert an op in their region.
  ApplyPatternsOpPatterns configuration;
  variantH = iree_compiler::buildCanonicalizationAndEnablingTransforms(
      b, configuration, variantH);

  // Apply vector masking.
  AbstractGemmLikeStrategy::MappingInfo lhsCopyMapping =
      strategy.lhsCopyMapping();
  b.create<transform::MaskedVectorizeOp>(lhsCopyOpH, ValueRange(), false,
                                         lhsCopyMapping.tileSizes);
  AbstractGemmLikeStrategy::MappingInfo rhsCopyMapping =
      strategy.rhsCopyMapping();
  b.create<transform::MaskedVectorizeOp>(rhsCopyOpH, ValueRange(), false,
                                         rhsCopyMapping.tileSizes);
  AbstractGemmLikeStrategy::MappingInfo resCopyMapping =
      strategy.resCopyMapping();
  b.create<transform::MaskedVectorizeOp>(copyBackOpH, ValueRange(), false,
                                         resCopyMapping.tileSizes);

  // Lower all masked vector transfers at this point, as they make
  // canonicalization generate incorrect IR.
  // TODO: don't rematch, apply on the variant op directly.
  Value funcH =
      b.create<transform::MatchOp>(variantH, func::FuncOp::getOperationName());
  funcH = buildLowerMaskedTransfersAndCleanup(b, funcH);

  // Apply vectorization + cleanups to what remains.
  funcH = iree_compiler::buildVectorize(b, funcH, /*applyCleanups=*/true);
}

/// Build the transform IR to perform conversion to tensor core operations.
/// This is currently subject to phase orderings as follows:
///   - Vector transfer_read and transfer_write patterns have different subview
///     folding behavior, force a fold_memref_aliases on them to enable
///     redundant vector transfer hoisting.
///   - Unfortunately, fold_memref_aliases breaks vector_to_mma conversion
///     across scf.for after unrolling due to insert_strided_slice /
///     extract_strided_slice across iter_args boundaries.
///   - Hoist redundant vector transfers to allow conversion to tensor core to
///     proceed. We really don't want to do this after bufferization but we need
///     to atm.
Value mlir::iree_compiler::gpu::buildConvertToTensorCoreOp(
    ImplicitLocOpBuilder &b, Value funcH,
    const AbstractGemmLikeStrategy &strategy) {
  // TODO: Fewer canonicalization.
  iree_compiler::buildCanonicalizationAndEnablingTransforms(
      b, ApplyPatternsOpPatterns(), funcH);
  b.create<iree_compiler::IREE::transform_dialect::HoistStaticAllocOp>(funcH);
  {
    ApplyPatternsOpPatterns config;
    config.foldMemrefAliases = true;
    b.create<ApplyPatternsOp>(funcH, config);
  }
  {
    ApplyPatternsOpPatterns config;
    config.extractAddressComputations = true;
    b.create<ApplyPatternsOp>(funcH, config);
  }
  iree_compiler::buildCanonicalizationAndEnablingTransforms(
      b, ApplyPatternsOpPatterns(), funcH);
  {
    ApplyPatternsOpPatterns config;
    if (strategy.useMmaSync)
      config.unrollVectorsGpuMmaSync = true;
    else
      config.unrollVectorsGpuWmma = true;
    b.create<ApplyPatternsOp>(funcH, config);
  }
  // TODO: not a functional style transform and avoid returning funcH.
  funcH = b.create<transform::HoistRedundantVectorTransfersOp>(
      pdl::OperationType::get(b.getContext()), funcH);
  iree_compiler::buildCanonicalizationAndEnablingTransforms(
      b, ApplyPatternsOpPatterns(), funcH);
  b.create<ApplyBufferOptimizationsOp>(funcH);
  auto vectorToMMaConversionOp =
      b.create<iree_compiler::IREE::transform_dialect::VectorToMMAConversionOp>(
          funcH);
  // TODO: proper builder instead of a setting post-hoc.
  if (strategy.useMmaSync)
    vectorToMMaConversionOp.setUseMmaSync(true);
  else
    vectorToMMaConversionOp.setUseWmma(true);
  return funcH;
}

void mlir::iree_compiler::gpu::buildMultiBuffering(
    ImplicitLocOpBuilder &b, Value funcH,
    const AbstractGemmLikeStrategy &strategy) {
  iree_compiler::buildCanonicalizationAndEnablingTransforms(
      b, ApplyPatternsOpPatterns(), funcH);
  ApplyPatternsOpPatterns config;
  config.foldMemrefAliases = true;
  b.create<ApplyPatternsOp>(funcH, config);
  // TODO: Avoid brittle matching here.
  // TODO: Better builder after integrate.
  Value allocH = b.create<transform::MatchOp>(
      transform::OperationType::get(b.getContext(), "memref.alloc"), funcH,
      b.getStrArrayAttr({memref::AllocOp::getOperationName()}),
      /*matchInterfaceEnum=*/transform::MatchInterfaceEnumAttr(),
      /*opAttrs=*/DictionaryAttr(),
      /*filterResultType=*/TypeAttr());
  // TODO: Better builder instead of setting post-hoc.
  auto multiBufferOp = b.create<transform::MemRefMultiBufferOp>(
      pdl::OperationType::get(b.getContext()), allocH);
  multiBufferOp.setFactor(strategy.pipelineDepth);
  multiBufferOp.setSkipAnalysis(true);
}

Value mlir::iree_compiler::gpu::buildConvertToAsyncCopies(
    ImplicitLocOpBuilder &b, Value funcH,
    const AbstractGemmLikeStrategy &strategy) {
  // Atm, vectors need to be lowered to 1-D for cp.async mapping to connect.
  // TODO: not a functional style op to avoid invalidating artificially.
  auto transferToScfOp = b.create<transform::TransferToScfOp>(
      pdl::OperationType::get(b.getContext()), funcH);
  // TODO: proper builder instead of a setting post-hoc.
  transferToScfOp.setMaxTransferRank(1);
  transferToScfOp.setFullUnroll(true);
  funcH = transferToScfOp->getResult(0);
  iree_compiler::buildCanonicalizationAndEnablingTransforms(
      b, ApplyPatternsOpPatterns(), funcH);
  auto createAsyncGroupOp =
      b.create<iree_compiler::IREE::transform_dialect::CreateAsyncGroupsOp>(
          TypeRange{}, funcH);
  // TODO: proper builder instead of a setting post-hoc.
  createAsyncGroupOp.setUseMmaSync(strategy.useMmaSync);
  ApplyPatternsOpPatterns config;
  config.foldMemrefAliases = true;
  iree_compiler::buildCanonicalizationAndEnablingTransforms(b, config, funcH);
  return funcH;
}

void mlir::iree_compiler::gpu::buildPipelineSharedMemoryCopies(
    ImplicitLocOpBuilder &b, Value funcH,
    const AbstractGemmLikeStrategy &strategy) {
  Value computeOpH;
  if (strategy.useMmaSync) {
    computeOpH = b.create<transform::MatchOp>(
        funcH, mlir::nvgpu::MmaSyncOp::getOperationName());
  } else {
    computeOpH = b.create<transform::MatchOp>(
        funcH, mlir::gpu::SubgroupMmaComputeOp::getOperationName());
  }
  // TODO: Better builder.
  Value forOpH = b.create<transform::GetParentForOp>(
      pdl::OperationType::get(b.getContext()), computeOpH);
  // TODO: Better builder instead of setting post-hoc.
  auto pipelineOp = b.create<
      iree_compiler::IREE::transform_dialect::PipelineSharedMemoryCopiesOp>(
      pdl::OperationType::get(b.getContext()), forOpH);
  // TODO: depth from strategy, or directly from individual buffers.
  pipelineOp.setDepth(strategy.pipelineDepth);
}

Value mlir::iree_compiler::gpu::buildBufferize(ImplicitLocOpBuilder &b,
                                               Value variantH) {
  ApplyPatternsOpPatterns patterns;
  patterns.canonicalization = true;
  patterns.cse = true;
  patterns.licm = true;
  b.create<ApplyPatternsOp>(variantH, patterns);
  b.create<IREEEliminateEmptyTensorsOp>(variantH);
  auto bufferizeOp = b.create<IREEBufferizeOp>(variantH, /*targetGpu=*/true);
  bufferizeOp.setTargetGpu(true);
  variantH = bufferizeOp.getResult();
  Value memrefFunc =
      b.create<MatchOp>(variantH, func::FuncOp::getOperationName());
  b.create<IREEEraseHALDescriptorTypeFromMemRefOp>(memrefFunc);
  b.create<ApplyBufferOptimizationsOp>(memrefFunc);
  return variantH;
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
  if (!gpuModel.hasWarpShuffle) {
    LDBG("--Reduction strategy no warp shuffle\n");
    return failure();
  }

  // 1. Match a reduction and surrounding ops.
  StructuredOpMatcher *reduction;
  transform_ext::MatchedReductionCaptures captures;
  transform_ext::MatcherContext matcherContext;
  makeReductionMatcher(matcherContext, reduction, captures);
  if (!matchPattern(op, *reduction)) {
    LDBG("--Reduction strategy failed to match\n");
    return failure();
  }

  // 2. Construct the configuration and the strategy builder.
  // TODO: Generalize along the HW axis.
  auto strategyBuilder = [&](ImplicitLocOpBuilder &b, Value variant) {
    ReductionConfig reductionConfig = getReductionConfig(captures, gpuModel);
    if (reductionConfig.strategy == ReductionStrategy::Small) {
      SmallReductionStrategy strategy(captures, reductionConfig);
      return buildSmallReductionStrategy(b, variant, strategy);
    } else if (reductionConfig.strategy == ReductionStrategy::Staged) {
      // Otherwise, always fallback to the staged strategy.
      StagedReductionStrategy strategy(captures, reductionConfig);
      return buildStagedReductionStrategy(b, variant, strategy);
    } else {
      return llvm_unreachable("Unknown strategy");
    }
  };

  // 3. Build strategy embedded into the IR.
  mlir::iree_compiler::createTransformRegion(entryPoint, strategyBuilder);

  return success();
}

static LogicalResult matchAndSetMatmulStrategy(func::FuncOp entryPoint,
                                               linalg::LinalgOp op,
                                               const GPUModel &gpuModel) {
  if (!clGPUEnableTransformDialectMatmulTensorCoreStrategy) {
    LDBG("--Matmul strategy flag turned off\n");
    return failure();
  }
  if (!gpuModel.hasTF32TensorCore) {
    LDBG("--Matmul strategy no TF32 tensor core\n");
    return failure();
  }

  // 1. Match a reduction and surrounding ops.
  StructuredOpMatcher *fill;
  StructuredOpMatcher *matmul;
  StructuredOpMatcher *trailing;
  transform_ext::MatchedMatmulCaptures captures;
  transform_ext::MatcherContext matcherContext;
  makeMatmulMatcher(matcherContext, matmul, fill, trailing, captures);
  if (!matchPattern(op, *matmul)) {
    LDBG("--Matmul strategy fail to match\n");
    return failure();
  }

  // We are very peculiar about the dispatches we want to match for now:
  //   - f32 only atm.
  //   - Mandatory fill op.
  //   - No trailing op.
  //   - If the matmul is "too aligned", then use the default IREE strategy.
  //   - If the matmul is "too small", then use the default IREE strategy.
  //   - Otherwise, we take it.
  if (!fill->getCaptured() || trailing->getCaptured()) {
    LDBG("--Matmul strategy fill / trailing preconditions failed\n");
    return failure();
  }

  if (!captures.lhsElementType.isF32() || !captures.rhsElementType.isF32() ||
      !captures.outputElementType.isF32()) {
    LDBG("--Matmul strategy elemental type check failed\n");
    return failure();
  }

  // TODO: Generalize to a good mix of sizes, alignments and element types.
  const auto &matmulSize = captures.matmulOpSizes;
  if (matmulSize.size() != 3) {
    LDBG("--Matmul strategy size capture failed\n");
    return failure();
  }

  // Currently the unaligned transform strategy does not properly handle
  // optionality when padding operations get folded away. So we only use it when
  // all operands require a padding that does not fold. This is the case when
  // either:
  //   - m and k are not aligned to the tile sizes (conservatively, take 64, 16)
  //   - n and k are not aligned to the tile sizes (conservatively, take 64, 16)
  // Other cases currently result in folding and fall back to the default
  // unaligned IREE strategy.
  bool supportedUnalignedCases =
      (matmulSize[0] % 64 != 0 && matmulSize[2] % 16 != 0) ||
      (matmulSize[1] % 64 != 0 && matmulSize[2] % 16 != 0);

  if (!supportedUnalignedCases) {
    LDBG("--Matmul strategy alignment check failed\n");
    return failure();
  }

  // Currently the unaligned transform strategy does not properly handle
  // degenerate dimensions that should have been rank-reduced (e.g. `1`).
  // Also, it is unprofitable to force small matmuls through a high latency
  // tensorcore path, we are better off with a simple simt strategy.
  // TODO: profitability details can be ironed out in the future when we have a
  // heuristic to better select strategy parameters.
  bool unsupportedSmallCases = (matmulSize[0] > 0 && matmulSize[0] < 8) ||
                               (matmulSize[1] > 0 && matmulSize[1] < 8) ||
                               (matmulSize[2] > 0 && matmulSize[2] < 8);
  if (unsupportedSmallCases) {
    LDBG("--Matmul strategy small size check failed\n");
    return failure();
  }

  // 2. Construct the configuration and the strategy builder.
  // TODO: Generalize along the HW axis.
  auto strategyBuilder = [&](ImplicitLocOpBuilder &b, Value variant) {
    iree_compiler::gpu::MatmulStrategy strategy(op->getContext(), captures);
    return buildMatmulTensorCoreStrategy(b, variant, strategy);
  };

  // 3. Build strategy embedded into the IR.
  mlir::iree_compiler::createTransformRegion(entryPoint, strategyBuilder);

  return success();
}

LogicalResult mlir::iree_compiler::gpu::matchAndSetTransformStrategy(
    func::FuncOp entryPoint, Operation *op, const GPUModel &gpuModel) {
  LDBG("Look up a TD strategy for entryPoint:\n" << entryPoint << "\n");
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp) {
    LDBG("Not a Linalg op: " << *op << " -> Fail\n");
    return failure();
  }
  if (succeeded(matchAndSetReductionStrategy(entryPoint, linalgOp, gpuModel))) {
    LDBG("Activate reduction strategy\n");
    return success();
  }
  if (succeeded(matchAndSetMatmulStrategy(entryPoint, linalgOp, gpuModel))) {
    LDBG("Activate matmul\n");
    return success();
  }
  // TODO: Add more transform dialect strategy for other kind of dispatch
  // regions.
  LDBG("No suitable strategy found\n");
  return failure();
}

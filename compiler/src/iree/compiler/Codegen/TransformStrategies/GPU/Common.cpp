// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/TransformStrategies/GPU/Common.h"

#include <tuple>

#include "iree-dialects/Dialect/LinalgTransform/StructuredTransformOpsExt.h"
#include "iree-dialects/Transforms/TransformMatchers.h"
#include "iree/compiler/Codegen/Common/TransformExtensions/CommonExtensions.h"
#include "iree/compiler/Codegen/LLVMGPU/TransformExtensions/LLVMGPUExtensions.h"
#include "iree/compiler/Codegen/TransformStrategies/Common/Common.h"
#include "iree/compiler/Codegen/TransformStrategies/GPU/ConvolutionImplicitGemmStrategy.h"
#include "iree/compiler/Codegen/TransformStrategies/GPU/MatmulTensorCoreStrategy.h"
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
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/TransformOps/VectorTransformOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;

// TODO: significantly better namespacing.
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
using iree_compiler::gpu::GPUModel;
using iree_compiler::IREE::transform_dialect::EliminateGpuBarriersOp;
using iree_compiler::IREE::transform_dialect::IREEBufferizeOp;
using iree_compiler::IREE::transform_dialect::IREEEliminateEmptyTensorsOp;
using iree_compiler::IREE::transform_dialect::
    IREEPopulateWorkgroupCountRegionUsingNumThreadsSliceOp;
using iree_compiler::IREE::transform_dialect::ShareForallOperandsOp;
using iree_compiler::IREE::transform_dialect::SynchronizeLoopOp;
using transform::FuseIntoContainingOp;
using transform::MatchOp;
using transform::MemRefEraseDeadAllocAndStoresOp;
using transform::RewriteInDestinationPassingStyleOp;
using transform::ScalarizeOp;
using transform::SequenceOp;

//===----------------------------------------------------------------------===//
// General helpers.
//===----------------------------------------------------------------------===//

/// Return max(1, (value * 32) / bitwidth).
int64_t mlir::iree_compiler::gpu::scaleUpByBitWidth(int64_t value,
                                                    int64_t bitWidth) {
  assert((bitWidth & (bitWidth - 1)) == 0 && "bitWidth must be a power of 2");
  return std::max((value * 32) / bitWidth, int64_t(1));
}

/// Adjust the number of warps to use to benefit from packing multiple smaller
/// elemental types within a single 128 bit shuffled element.
int64_t mlir::iree_compiler::gpu::adjustNumberOfWarpsForBlockShuffle(
    int64_t numWarpsToUse, int64_t bitWidth) {
  // Try to scale down the number of warps to use 32b elements in warp shuffles.
  assert(((bitWidth & (bitWidth - 1)) == 0) && "bitWidth must be a power of 2");
  int64_t factor;
  for (factor = scaleUpByBitWidth(1, bitWidth); factor > 1; factor >>= 1)
    if (numWarpsToUse % factor == 0)
      break;
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
    std::optional<int64_t> subgroupSize) {
  b.create<ForallToWorkgroupOp>(funcH);
  auto mapToThreadsOp =
      b.create<MapNestedForallToGpuThreadsOp>(funcH, blockSize);
  if (subgroupSize)
    mapToThreadsOp.setSubgroupSize(*subgroupSize);
  return funcH;
}

/// Post-bufferization vector distribution with rank-reduction.
/// Takes a handle to a func.func and returns an updated handle to a
/// func.func.
Value mlir::iree_compiler::gpu::buildDistributeVectors(ImplicitLocOpBuilder &b,
                                                       Value variantH,
                                                       Value funcH,
                                                       int64_t warpSize) {
  b.create<transform::ApplyPatternsOp>(funcH, [](OpBuilder &b, Location loc) {
    b.create<transform::ApplyFoldMemrefAliasOpsPatternsOp>(loc);
    b.create<transform::ApplyCastAwayVectorLeadingOneDimPatternsOp>(loc);
  });
  Value ifH = b.create<MatchOp>(funcH, scf::IfOp::getOperationName());
  // Locally suppress failures for this op only because it doesn't cover the
  // `threadIdx.x == 0 && threadIdx.y == 0` case at the moment.
  auto sequence = b.create<SequenceOp>(
      TypeRange(), transform::FailurePropagationMode::Suppress, variantH,
      /*extraBindings=*/ValueRange());
  {
    OpBuilder::InsertionGuard guard(b);
    b.createBlock(&sequence.getBody(), sequence.getBody().begin(),
                  transform::AnyOpType::get(b.getContext()), b.getLoc());
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
        ImplicitLocOpBuilder &b, Value variantH, Value opH, int64_t rank,
        int64_t mostMinorDim, SmallVector<int64_t> opSizes, int64_t numThreads,
        Attribute mappingAttr, int64_t maxVectorSize) {
  // Poor man's handling of optionality in C++. Will need to be converted to
  // proper transform dialect filters or handling of emptiness.
  if (rank == 0)
    return;

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
    auto anyOpType = transform::AnyOpType::get(b.getContext());
    auto split = b.create<transform::SplitOp>(
        anyOpType, anyOpType, opH, b.getI64IntegerAttr(mostMinorDim), Value(),
        b.getI64IntegerAttr(splitPoint));
    opH = split.getFirst();
    if (vectorSize > 1) {
      auto res = iree_compiler::buildTileFuseToScfFor(
          /*b=*/b,
          /*variantH=*/variantH,
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
          /*variantH=*/variantH,
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
        /*variantH=*/variantH,
        /*rootH=*/opH,
        /*opsHToFuse=*/{},
        /*tileSizes=*/getAsOpFoldResult(b.getI64ArrayAttr({scfForTileSizes})));
    opH = res.tiledOpH;
  }
  if (numThreads > 1) {
    assert(mappingAttr && "must specify a mapping attribute");
    iree_compiler::buildTileFuseDistToForallWithNumThreads(
        /*b=*/b,
        /*variantH=*/variantH,
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
  // Step N-4. Perform a pass of canonicalization + enabling after tiling.
  mlir::iree_compiler::buildCanonicalizationAndEnablingTransforms(
      b, funcH, [](OpBuilder &b, Location loc) {
        b.create<transform::ApplyFoldTensorEmptyPatternsOp>(loc);
      });
  funcH = iree_compiler::buildVectorize(b, funcH);

  // Step N-3. Perform a pass of canonicalization + enabling after vectorization
  // as well as hoisting subset operations such as vector.transfer_read/write.
  mlir::iree_compiler::buildCanonicalizationAndEnablingTransforms(
      b, funcH, [](OpBuilder &b, Location loc) {
        b.create<transform::ApplyFoldTensorEmptyPatternsOp>(loc);
      });
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
  mlir::iree_compiler::buildCanonicalizationAndEnablingTransforms(
      b, funcH, [](OpBuilder &b, Location loc) {
        b.create<transform::ApplyFoldTensorEmptyPatternsOp>(loc);
      });
  return std::make_pair(variantH, funcH);
}

//===----------------------------------------------------------------------===//
// Subset of mid-level builders currently used for GEMM-like problems.
//===----------------------------------------------------------------------===//

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
    Value funcH = b.create<transform::MatchOp>(
        variantH, func::FuncOp::getOperationName());
    iree_compiler::buildCanonicalizationAndEnablingTransforms(
        b, funcH, [](OpBuilder &b, Location loc) {
          b.create<transform::ApplyFoldTensorSubsetOpsPatternsOp>(loc);
          b.create<
              transform::ApplyMergeConsecutiveInsertExtractSlicePatternsOp>(
              loc);
        });
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
          /*variantH=*/variantH,
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
          /*variantH=*/variantH,
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
  // Aligned vs unaligned handling deviates here by converting the pads to
  // copies for the aligned case.
  // TODO: Unify aligned and unaligned codegen.
  Value copyBackOpH;
  if (!strategy.alignedRes()) {
    // Explicitly materialize the parent parallel_insert into a copy to avoid
    // late bufferization interferences.
    // TODO: Avoid brittle rematching.
    Value insertSliceH = b.create<transform::MatchOp>(
        variantH, tensor::ParallelInsertSliceOp::getOperationName());
    copyBackOpH = b.create<transform::InsertSliceToCopyOp>(
        insertSliceH.getType(), insertSliceH);
  } else {
    Value resH = b.create<transform::GetProducerOfOperand>(
        paddedMatmulOpH.getType(), paddedMatmulOpH, b.getI64IntegerAttr(2));
    copyBackOpH =
        b.create<RewriteInDestinationPassingStyleOp>(resH.getType(), resH);
  }

  Value lhsH = b.create<transform::GetProducerOfOperand>(
      paddedMatmulOpH.getType(), paddedMatmulOpH, b.getI64IntegerAttr(0));
  Value rhsH = b.create<transform::GetProducerOfOperand>(
      paddedMatmulOpH.getType(), paddedMatmulOpH, b.getI64IntegerAttr(1));

  // Rewrite aligned pads as destination passing (linalg.copy)
  if (strategy.alignedLhs() && strategy.packingDimensions[0])
    lhsH = b.create<RewriteInDestinationPassingStyleOp>(lhsH.getType(), lhsH);
  if (strategy.alignedRhs() && strategy.packingDimensions[1])
    rhsH = b.create<RewriteInDestinationPassingStyleOp>(rhsH.getType(), rhsH);

  MappingInfo lhsCopyMapping = strategy.lhsCopyMapping();
  Value lhsCopyOpH = buildDistributeOnePadOrCopyWithNumThreads(
      b, variantH, lhsH, /*numThreads=*/lhsCopyMapping.numThreads,
      /*threadDimMapping=*/lhsCopyMapping.threadMapping,
      /*foldIfBranch=*/!strategy.alignedLhs());

  MappingInfo rhsCopyMapping = strategy.rhsCopyMapping();
  Value rhsCopyOpH = buildDistributeOnePadOrCopyWithNumThreads(
      b, variantH, rhsH, /*numThreads=*/rhsCopyMapping.numThreads,
      /*threadDimMapping=*/rhsCopyMapping.threadMapping,
      /*foldIfBranch=*/!strategy.alignedRhs());

  if (!strategy.alignedRes()) {
    MappingInfo resCopyMapping = strategy.resCopyMapping();
    copyBackOpH = buildDistributeOnePadOrCopyWithNumThreads(
        b, variantH, copyBackOpH,
        /*numThreads=*/resCopyMapping.numThreads,
        /*threadDimMapping=*/resCopyMapping.threadMapping);
  }

  return std::make_tuple(lhsCopyOpH, rhsCopyOpH, copyBackOpH);
}

/// Specific pattern to perform masked vectorization of copies give as
/// parameters, cleanup and vectorize the rest.
// TODO: generalize and don't hardcode.
void mlir::iree_compiler::gpu::buildMatmulVectorization(
    ImplicitLocOpBuilder &b, Value variantH, Value lhsCopyOpH, Value rhsCopyOpH,
    Value copyBackOpH, const AbstractGemmLikeStrategy &strategy,
    bool vectorizePadding, bool vectorizeNdExtract) {
  // Canonicalize to make padOp outputs static shaped: this is currently a
  // prerequisite for vector masking.
  // Also, no canonicalization is allowed after vector masking and before we
  // lower the masks: masks are currently quite brittle and do not like
  // canonicalization or anything else that may insert an op in their region.
  Value funcH =
      b.create<transform::MatchOp>(variantH, func::FuncOp::getOperationName());
  iree_compiler::buildCanonicalizationAndEnablingTransforms(b, funcH);

  // Apply vector masking.
  if (!strategy.alignedLhs()) {
    MappingInfo lhsCopyMapping = strategy.lhsCopyMapping();
    SmallVector<bool> scalableSizes(lhsCopyMapping.tileSizes.size(), false);
    b.create<transform::VectorizeOp>(lhsCopyOpH, ValueRange(), nullptr,
                                     scalableSizes, lhsCopyMapping.tileSizes);
  }
  if (!strategy.alignedRhs()) {
    MappingInfo rhsCopyMapping = strategy.rhsCopyMapping();
    SmallVector<bool> scalableSizes(rhsCopyMapping.tileSizes.size(), false);
    b.create<transform::VectorizeOp>(rhsCopyOpH, ValueRange(), nullptr,
                                     scalableSizes, rhsCopyMapping.tileSizes);
  }
  if (!strategy.alignedRes()) {
    MappingInfo resCopyMapping = strategy.resCopyMapping();
    SmallVector<bool> scalableSizes(resCopyMapping.tileSizes.size(), false);
    b.create<transform::VectorizeOp>(copyBackOpH, ValueRange(), nullptr,
                                     scalableSizes, resCopyMapping.tileSizes);
  }

  // Lower all masked vector transfers at this point, as they make
  // canonicalization generate incorrect IR.
  // TODO: don't rematch, apply on the variant op directly.
  funcH =
      b.create<transform::MatchOp>(variantH, func::FuncOp::getOperationName());
  buildLowerMaskedTransfersAndCleanup(b, funcH, /*cleanup=*/false);

  // Apply vectorization + cleanups to what remains.
  funcH = iree_compiler::buildVectorize(b, funcH, /*applyCleanups=*/true,
                                        vectorizePadding, vectorizeNdExtract);
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
  iree_compiler::buildCanonicalizationAndEnablingTransforms(b, funcH);
  b.create<iree_compiler::IREE::transform_dialect::HoistStaticAllocOp>(funcH);
  b.create<transform::ApplyPatternsOp>(funcH, [](OpBuilder &b, Location loc) {
    b.create<transform::ApplyFoldMemrefAliasOpsPatternsOp>(loc);
  });
  b.create<transform::ApplyPatternsOp>(funcH, [](OpBuilder &b, Location loc) {
    b.create<transform::ApplyExtractAddressComputationsPatternsOp>(loc);
  });
  iree_compiler::buildCanonicalizationAndEnablingTransforms(b, funcH);
  if (strategy.useWmma) {
    b.create<transform::ApplyPatternsOp>(
        funcH, [&](OpBuilder &b, Location loc) {
          b.create<iree_compiler::IREE::transform_dialect::
                       ApplyUnrollVectorsGpuWmmaSyncPatternsOp>(loc);
        });
  } else if (strategy.useMmaSync) {
    b.create<transform::ApplyPatternsOp>(
        funcH, [&](OpBuilder &b, Location loc) {
          b.create<iree_compiler::IREE::transform_dialect::
                       ApplyUnrollVectorsGpuMmaSyncPatternsOp>(loc);
        });
  } /* else nothing to do for fma here */

  Value forH = b.create<transform::MatchOp>(
      transform::OperationType::get(b.getContext(), "scf.for"), funcH,
      b.getStrArrayAttr({scf::ForOp::getOperationName()}),
      /*matchInterfaceEnum=*/transform::MatchInterfaceEnumAttr(),
      /*opAttrs=*/DictionaryAttr(),
      /*filterResultType=*/TypeAttr());
  // TODO: At this time, this synchronization is needed for applying the
  // HoistRedundantVectorTransfersOp transform correctly. This is because the
  // transform does not take parallelism into accound.
  // In the future, HoistRedundantVectorTransfersOp + SynchronizeLoopOp need to
  // be replaced by a single transform.
  b.create<SynchronizeLoopOp>(forH);

  b.create<transform::ApplyPatternsOp>(funcH, [](OpBuilder &b, Location loc) {
    b.create<transform::ApplyFoldMemrefAliasOpsPatternsOp>(loc);
  });
  b.create<IREE::transform_dialect::ApplyCommonSubexpressionEliminationOp>(
      funcH);
  // TODO: not a functional style transform and avoid returning funcH.
  funcH = b.create<transform::HoistRedundantVectorTransfersOp>(
      transform::AnyOpType::get(b.getContext()), funcH);
  iree_compiler::buildCanonicalizationAndEnablingTransforms(b, funcH);
  b.create<MemRefEraseDeadAllocAndStoresOp>(funcH);

  if (strategy.useWmma) {
    auto vectorToMMaConversionOp = b.create<
        iree_compiler::IREE::transform_dialect::VectorToMMAConversionOp>(funcH);
    // TODO: proper builder instead of a setting post-hoc.
    vectorToMMaConversionOp.setUseWmma(true);
  } else if (strategy.useMmaSync) {
    auto vectorToMMaConversionOp = b.create<
        iree_compiler::IREE::transform_dialect::VectorToMMAConversionOp>(funcH);
    // TODO: proper builder instead of a setting post-hoc.
    vectorToMMaConversionOp.setUseMmaSync(true);
  } /* else nothing to do for fma here */

  // Post-hoc elimiation of barriers.
  funcH = b.create<EliminateGpuBarriersOp>(funcH);
  return funcH;
}

void mlir::iree_compiler::gpu::buildMultiBuffering(
    ImplicitLocOpBuilder &b, Value funcH,
    const AbstractGemmLikeStrategy &strategy) {
  iree_compiler::buildCanonicalizationAndEnablingTransforms(b, funcH);
  b.create<transform::ApplyPatternsOp>(funcH, [](OpBuilder &b, Location loc) {
    b.create<transform::ApplyFoldMemrefAliasOpsPatternsOp>(loc);
  });
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
      transform::AnyOpType::get(b.getContext()), allocH);
  multiBufferOp.setFactor(strategy.pipelineDepth);
  multiBufferOp.setSkipAnalysis(true);
}

Value mlir::iree_compiler::gpu::buildConvertToAsyncCopies(
    ImplicitLocOpBuilder &b, Value funcH,
    const AbstractGemmLikeStrategy &strategy) {
  b.create<transform::ApplyPatternsOp>(funcH, [&](OpBuilder &b, Location loc) {
    // Atm, vectors need to be lowered to 1-D for cp.async mapping to connect.
    // TODO: not a functional style op to avoid invalidating artificially.
    auto transferToScfOp =
        b.create<transform::ApplyTransferToScfPatternsOp>(loc);
    // TODO: proper builder instead of a setting post-hoc.
    transferToScfOp.setMaxTransferRank(1);
    transferToScfOp.setFullUnroll(true);
  });
  iree_compiler::buildCanonicalizationAndEnablingTransforms(b, funcH);
  auto createAsyncGroupOp =
      b.create<iree_compiler::IREE::transform_dialect::CreateAsyncGroupsOp>(
          TypeRange{}, funcH);
  if (strategy.useMmaSync) {
    // TODO: proper builder instead of a setting post-hoc.
    createAsyncGroupOp.setUseMmaSync(strategy.useMmaSync);
  }
  iree_compiler::buildCanonicalizationAndEnablingTransforms(
      b, funcH, [](OpBuilder &b, Location loc) {
        b.create<transform::ApplyFoldMemrefAliasOpsPatternsOp>(loc);
      });
  return funcH;
}

void mlir::iree_compiler::gpu::buildPipelineSharedMemoryCopies(
    ImplicitLocOpBuilder &b, Value funcH,
    const AbstractGemmLikeStrategy &strategy) {
  Value computeOpH;
  if (strategy.useWmma) {
    computeOpH = b.create<transform::MatchOp>(
        funcH, mlir::gpu::SubgroupMmaComputeOp::getOperationName());
  } else if (strategy.useMmaSync) {
    computeOpH = b.create<transform::MatchOp>(
        funcH, mlir::nvgpu::MmaSyncOp::getOperationName());
  } else {
    assert(strategy.useFma);
    computeOpH = b.create<transform::MatchOp>(
        funcH, mlir::vector::ContractionOp::getOperationName());
  }
  // TODO: Better builder.
  Value forOpH = b.create<transform::GetParentOp>(
      transform::AnyOpType::get(b.getContext()), computeOpH,
      /*isolated_from_above=*/false, /*allow_empty_results=*/false,
      /*op_name=*/b.getStringAttr("scf.for"), /*deduplicate=*/true);
  // TODO: Better builder instead of setting post-hoc.
  auto pipelineOp = b.create<
      iree_compiler::IREE::transform_dialect::PipelineSharedMemoryCopiesOp>(
      transform::AnyOpType::get(b.getContext()), forOpH);
  // TODO: depth from strategy, or directly from individual buffers.
  pipelineOp.setDepth(strategy.pipelineDepth);
  pipelineOp.setUseMmaSync(strategy.useMmaSync);
  pipelineOp.setPeelEpilogue(strategy.peelPipelineEpilogue);
}

Value mlir::iree_compiler::gpu::buildBufferize(ImplicitLocOpBuilder &b,
                                               Value variantH) {
  Value funcH =
      b.create<transform::MatchOp>(variantH, func::FuncOp::getOperationName());
  b.create<transform::ApplyPatternsOp>(funcH, [](OpBuilder &b, Location loc) {
    b.create<transform::ApplyCanonicalizationPatternsOp>(loc);
  });
  b.create<IREE::transform_dialect::ApplyLoopIndependentCodeMotionOp>(funcH);
  b.create<IREE::transform_dialect::ApplyCommonSubexpressionEliminationOp>(
      funcH);
  b.create<IREEEliminateEmptyTensorsOp>(variantH);
  auto bufferizeOp = b.create<IREEBufferizeOp>(variantH, /*targetGpu=*/true);
  bufferizeOp.setTargetGpu(true);
  variantH = bufferizeOp.getResult();
  Value memrefFunc =
      b.create<MatchOp>(variantH, func::FuncOp::getOperationName());
  b.create<MemRefEraseDeadAllocAndStoresOp>(memrefFunc);
  return variantH;
}

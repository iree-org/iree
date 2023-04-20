// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/TransformDialectStrategies/GPU/MatmulTensorCoreStrategy.h"

#include "iree-dialects/Dialect/LinalgTransform/StructuredTransformOpsExt.h"
#include "iree/compiler/Codegen/Common/TransformExtensions/CommonExtensions.h"
#include "iree/compiler/Codegen/LLVMGPU/TransformExtensions/LLVMGPUExtensions.h"
#include "iree/compiler/Codegen/TransformDialectStrategies/Common/Common.h"
#include "iree/compiler/Codegen/TransformDialectStrategies/GPU/Common.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/TransformOps/MemRefTransformOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/TransformOps/SCFTransformOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Dialect/Vector/TransformOps/VectorTransformOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

using namespace mlir;

#define DEBUG_TYPE "iree-transform-builder"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

// TODO: significantly better namespacing.
using iree_compiler::IREE::transform_dialect::ApplyBufferOptimizationsOp;
using iree_compiler::IREE::transform_dialect::ApplyPatternsOp;
using iree_compiler::IREE::transform_dialect::ApplyPatternsOpPatterns;
using iree_compiler::IREE::transform_dialect::ForallToWorkgroupOp;
using iree_compiler::IREE::transform_dialect::IREEBufferizeOp;
using iree_compiler::IREE::transform_dialect::IREEEliminateEmptyTensorsOp;
using iree_compiler::IREE::transform_dialect::
    IREEEraseHALDescriptorTypeFromMemRefOp;
using iree_compiler::IREE::transform_dialect::ShareForallOperandsOp;
using iree_compiler::IREE::transform_dialect::VectorToWarpExecuteOnLane0Op;
using iree_compiler::IREE::transform_dialect::VectorWarpDistributionOp;
using transform::FuseIntoContainingOp;
using transform::MatchOp;
using transform::ScalarizeOp;
using transform::SequenceOp;
using transform_ext::RegisterMatchCallbacksOp;
using transform_ext::StructuredOpMatcher;

using iree_compiler::buildSelectFirstNonEmpty;
using iree_compiler::buildTileFuseDistToForallAndWorkgroupCountWithTileSizes;
using iree_compiler::buildTileFuseDistToForallWithNumThreads;
using iree_compiler::TileToForallAndFuseAndDistributeResult;
using iree_compiler::gpu::adjustNumberOfWarpsForBlockShuffle;
using iree_compiler::gpu::build1DSplittingStrategyWithOptionalThreadMapping;
using iree_compiler::gpu::buildCommonTrailingStrategy;
using iree_compiler::gpu::buildDistributeVectors;
using iree_compiler::gpu::kCudaMaxNumThreads;
using iree_compiler::gpu::kCudaMaxVectorLoadBitWidth;
using iree_compiler::gpu::kCudaWarpSize;
using iree_compiler::gpu::MatmulStrategy;
using iree_compiler::gpu::scaleUpByBitWidth;

/// Build the transform IR to pad a matmul op `matmulOpH`.
// TODO: Less hardcoded, more generalization, extract information from strategy.
static Value buildPadMatmul(ImplicitLocOpBuilder &b, Value matmulOpH,
                            const MatmulStrategy &strategy) {
  SmallVector<float> paddingValues = {0.f, 0.f, 0.f};
  SmallVector<int64_t> paddingDimensions = {0, 1, 2};
  SmallVector<int64_t> packingDimensions = {1, 1, 1};
  // TODO: Better upstream builder.
  return b.create<transform::PadOp>(
      matmulOpH.getType(), matmulOpH, b.getF32ArrayAttr(paddingValues),
      b.getI64ArrayAttr(paddingDimensions),
      b.getI64ArrayAttr(packingDimensions), ArrayAttr());
}

/// Build transform IR to hoist the padded output operand of a padded matmul.
/// Additionally, this attempts to fold the padding into the producing fill, if
/// available.
// TODO: Generalize, this is not specific to a matmul.
// TODO: Better API
static Value buildHoistOutputPaddingOp(ImplicitLocOpBuilder &b, Value variantH,
                                       Value paddedMatmulOpH,
                                       int64_t numLoopsToHoist = 1) {
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
static Value buildDistributeOnePadOrCopy(ImplicitLocOpBuilder &b,
                                         Value variantH, Value copyOpH,
                                         ArrayRef<int64_t> numThreads,
                                         ArrayRef<Attribute> threadDimMapping,
                                         bool foldIfBranch = false) {
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
static std::tuple<Value, Value, Value> buildDistributeCopies(
    ImplicitLocOpBuilder &b, Value variantH, Value paddedMatmulOpH,
    const MatmulStrategy &strategy) {
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

  MatmulStrategy::MappingInfo lhsCopyMapping = strategy.lhsCopyMapping();
  Value lhsCopyOpH = buildDistributeOnePadOrCopy(
      b, variantH, lhsH, /*numThreads=*/lhsCopyMapping.numThreads,
      /*threadDimMapping=*/lhsCopyMapping.threadMapping, /*foldIfBranch=*/true);

  MatmulStrategy::MappingInfo rhsCopyMapping = strategy.rhsCopyMapping();
  Value rhsCopyOpH = buildDistributeOnePadOrCopy(
      b, variantH, rhsH, /*numThreads=*/rhsCopyMapping.numThreads,
      /*threadDimMapping=*/rhsCopyMapping.threadMapping, /*foldIfBranch=*/true);

  MatmulStrategy::MappingInfo resCopyMapping = strategy.resCopyMapping();
  copyBackOpH = buildDistributeOnePadOrCopy(
      b, variantH, copyBackOpH,
      /*numThreads=*/resCopyMapping.numThreads,
      /*threadDimMapping=*/rhsCopyMapping.threadMapping);

  return std::make_tuple(lhsCopyOpH, rhsCopyOpH, copyBackOpH);
}

/// Specific pattern to perform masked vectorization of copies give as
/// parameters, cleanup and vectorize the rest.
// TODO: generalize and don't hardcode.
static void buildMatmulVectorization(ImplicitLocOpBuilder &b, Value variantH,
                                     Value lhsCopyOpH, Value rhsCopyOpH,
                                     Value copyBackOpH,
                                     const MatmulStrategy &strategy) {
  // Canonicalize to make padOp outputs static shaped: this is currently a
  // prerequisite for vector masking.
  // Also, no canonicalization is allowed after vector masking and before we
  // lower the masks: masks are currently quite brittle and do not like
  // canonicalization or anything else that may insert an op in their region.
  {
    ApplyPatternsOpPatterns configuration;
    variantH = iree_compiler::buildCanonicalizationAndEnablingTransforms(
        b, configuration, variantH);
  }
  // Apply vector masking.
  MatmulStrategy::MappingInfo lhsCopyMapping = strategy.lhsCopyMapping();
  b.create<transform::MaskedVectorizeOp>(lhsCopyOpH, ValueRange(), false,
                                         lhsCopyMapping.tileSizes);
  MatmulStrategy::MappingInfo rhsCopyMapping = strategy.rhsCopyMapping();
  b.create<transform::MaskedVectorizeOp>(rhsCopyOpH, ValueRange(), false,
                                         rhsCopyMapping.tileSizes);
  MatmulStrategy::MappingInfo resCopyMapping = strategy.resCopyMapping();
  b.create<transform::MaskedVectorizeOp>(copyBackOpH, ValueRange(), false,
                                         resCopyMapping.tileSizes);

  // TODO: don't rematch, apply on the variant op directly.
  Value funcH =
      b.create<transform::MatchOp>(variantH, func::FuncOp::getOperationName());
  // TODO: avoid functional style transform so we can apply to the variant.
  funcH = b.create<transform::LowerMaskedTransfersOp>(funcH.getType(), funcH);
  {
    ApplyPatternsOpPatterns configuration;
    configuration.rankReducingLinalg = true;
    configuration.rankReducingVector = true;
    b.create<ApplyPatternsOp>(funcH, configuration);
  }
  b.create<transform::VectorizeOp>(funcH);
  {
    ApplyPatternsOpPatterns configuration;
    variantH = iree_compiler::buildCanonicalizationAndEnablingTransforms(
        b, configuration, variantH);
  }
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
static Value buildConvertToTensorCoreOp(ImplicitLocOpBuilder &b, Value funcH,
                                        const MatmulStrategy &strategy) {
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

static void buildMultiBuffering(ImplicitLocOpBuilder &b, Value funcH,
                                const MatmulStrategy &strategy) {
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

static Value buildConvertToAsyncCopies(ImplicitLocOpBuilder &b, Value funcH,
                                       const MatmulStrategy &strategy) {
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

static void buildPipelineSharedMemoryCopies(ImplicitLocOpBuilder &b,
                                            Value funcH,
                                            const MatmulStrategy &strategy) {
  Value subgroupMmaOpH = b.create<transform::MatchOp>(
      funcH, mlir::gpu::SubgroupMmaComputeOp::getOperationName());
  // TODO: Better builder.
  Value forOpH = b.create<transform::GetParentForOp>(
      pdl::OperationType::get(b.getContext()), subgroupMmaOpH);
  // TODO: Better builder instead of setting post-hoc.
  auto pipelineOp = b.create<
      iree_compiler::IREE::transform_dialect::PipelineSharedMemoryCopiesOp>(
      pdl::OperationType::get(b.getContext()), forOpH);
  // TODO: depth from strategy, or directly from individual buffers.
  pipelineOp.setDepth(strategy.pipelineDepth);
}

static std::tuple<Value, Value, Value, Value>
buildMatmulStrategyBlockDistribution(ImplicitLocOpBuilder &b, Value variantH,
                                     const MatmulStrategy &strategy) {
  // Step 1. Call the matcher. Note that this is the same matcher as used to
  // trigger this compilation path, so it must always apply.
  b.create<RegisterMatchCallbacksOp>();
  auto [fillH, matmulH, maybeTrailingH] = unpackRegisteredMatchCallback<3>(
      b, "matmul", transform::FailurePropagationMode::Propagate, variantH);

  // Step 2. Create the block/mapping tiling level and fusee.
  // auto [fusionTargetH, fusionGroupH] =
  //     buildSelectFirstNonEmpty(b, maybeTrailingH, matmulH);
  MatmulStrategy::MappingInfo blockMapping = strategy.getBlockMapping();
  TileToForallAndFuseAndDistributeResult tileResult =
      buildTileFuseDistToForallAndWorkgroupCountWithTileSizes(
          /*builder=*/b,
          /*isolatedParentOpH=*/variantH,
          /*rootH=*/matmulH,
          /*opsToFuseH=*/fillH,
          /*tileSizes=*/
          getAsOpFoldResult(b.getI64ArrayAttr(blockMapping.tileSizes)),
          /*threadDimMapping=*/
          b.getArrayAttr(blockMapping.threadMapping));

  // TODO: handle trailing op.
  return std::make_tuple(tileResult.resultingFusedOpsHandles.front(),
                         tileResult.tiledOpH, Value(), tileResult.forallH);
}

static Value buildBufferize(ImplicitLocOpBuilder &b, Value variantH) {
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

void iree_compiler::gpu::buildMatmulTensorCoreStrategy(
    ImplicitLocOpBuilder &b, Value variantH, const MatmulStrategy &strategy) {
  // Step 1. Apply block-level part of the strategy, keeps everything fused.
  auto [fillH, matmulH, maybeTiledTrailingHBlock, forall] =
      buildMatmulStrategyBlockDistribution(b, variantH, strategy);
  // Tile reduction loop.
  SmallVector<int64_t> tileSizes{0, 0, strategy.reductionTileSize};
  auto tileReductionResult =
      buildTileFuseToScfFor(b, variantH, matmulH, {},
                            getAsOpFoldResult(b.getI64ArrayAttr(tileSizes)));

  // Step 2. Pad the matmul op.
  auto paddedMatmulOpH =
      buildPadMatmul(b, tileReductionResult.tiledOpH, strategy);

  // Step 3. Hoist the padding of the output operand above the reduction loop.
  // The resulting fillOp will be mapped with the contraction using an SIMD
  // programming model.
  Value fillOpH = buildHoistOutputPaddingOp(b, variantH, paddedMatmulOpH);

  // Step 4. Distribute pad and copies: SIMT programming model.
  auto [lhsCopyOpH, rhsCopyOpH, copyBackOpH] =
      buildDistributeCopies(b, variantH, paddedMatmulOpH, strategy);

  // Step 5. Distribute to warps: SIMD programming model.
  // TODO: get the number of warps from strategy.
  MatmulStrategy::MappingInfo computeMapping = strategy.computeMapping();
  buildTileFuseDistToForallWithNumThreads(
      b, variantH, paddedMatmulOpH, ValueRange(),
      getAsOpFoldResult(b.getI64ArrayAttr(computeMapping.numThreads)),
      b.getArrayAttr(computeMapping.threadMapping));
  buildTileFuseDistToForallWithNumThreads(
      b, variantH, fillOpH, ValueRange(),
      getAsOpFoldResult(b.getI64ArrayAttr(computeMapping.numThreads)),
      b.getArrayAttr(computeMapping.threadMapping));

  // Step 6. Rank-reduce and vectorize.
  buildMatmulVectorization(b, variantH, lhsCopyOpH, rhsCopyOpH, copyBackOpH,
                           strategy);

  // Step 7. Bufferize and drop HAL descriptor from memref ops.
  variantH = ::buildBufferize(b, variantH);

  // Step 8. Post-bufferization mapping to blocks and threads.
  // Need to match again since bufferize invalidated all handles.
  // TODO: assumes a single func::FuncOp to transform, needs hardening.
  // TODO: extract info from strategy.
  Value funcH = b.create<MatchOp>(variantH, func::FuncOp::getOperationName());
  funcH = buildMapToBlockAndThreads(b, funcH, strategy.numThreads,
                                    strategy.numWarps);

  // Step 9. Convert to tensor core ops.
  // TODO: avoid consuming handles and returning here.
  funcH = buildConvertToTensorCoreOp(b, funcH, strategy);

  if (strategy.useAsyncCopies) {
    // Step 10. Multi-buffering.
    buildMultiBuffering(b, funcH, strategy);

    // Step 11. Convert to async copies.
    // TODO: avoid consuming handles and returning here.
    funcH = buildConvertToAsyncCopies(b, funcH, strategy);

    // Step 12. Pipeline shared memory copies.
    buildPipelineSharedMemoryCopies(b, funcH, strategy);
  }

  // Step 13. Late lowerings and cleanups.
  // TODO: not a functional style op to avoid invalidating artificially.
  funcH = b.create<transform::LowerMasksOp>(
      pdl::OperationType::get(b.getContext()), funcH);
  // TODO: not a functional style op to avoid invalidating artificially.
  funcH = b.create<transform::MaterializeMasksOp>(
      pdl::OperationType::get(b.getContext()), funcH);
  {
    ApplyPatternsOpPatterns config;
    config.foldMemrefAliases = true;
    iree_compiler::buildCanonicalizationAndEnablingTransforms(b, config, funcH);
  }
}

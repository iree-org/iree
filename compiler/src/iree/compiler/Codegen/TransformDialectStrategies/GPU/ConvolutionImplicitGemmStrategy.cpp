// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/TransformDialectStrategies/GPU/ConvolutionImplicitGemmStrategy.h"

#include "iree-dialects/Dialect/LinalgTransform/StructuredTransformOpsExt.h"
#include "iree/compiler/Codegen/Common/TransformExtensions/CommonExtensions.h"
#include "iree/compiler/Codegen/LLVMGPU/TransformExtensions/LLVMGPUExtensions.h"
#include "iree/compiler/Codegen/TransformDialectStrategies/Common/Common.h"
#include "iree/compiler/Codegen/TransformDialectStrategies/GPU/Common.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

using namespace mlir;

#define DEBUG_TYPE "iree-transform-builder"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

// TODO: significantly better namespacing.
using iree_compiler::IREE::transform_dialect::ApplyPatternsOp;
using iree_compiler::IREE::transform_dialect::ApplyPatternsOpPatterns;
using iree_compiler::IREE::transform_dialect::ApplyPatternsToNestedOp;
using iree_compiler::IREE::transform_dialect::
    ConvertConv2DToImg2ColAndAdjustWorkgroupCountOp;
using iree_compiler::IREE::transform_dialect::ForallToWorkgroupOp;
using iree_compiler::IREE::transform_dialect::GpuDistributeSharedMemoryCopyOp;
using iree_compiler::IREE::transform_dialect::HoistStaticAllocOp;
using iree_compiler::IREE::transform_dialect::IREEBufferizeOp;
using iree_compiler::IREE::transform_dialect::IREEEliminateEmptyTensorsOp;
using iree_compiler::IREE::transform_dialect::
    IREEEraseHALDescriptorTypeFromMemRefOp;
using iree_compiler::IREE::transform_dialect::MapNestedForallToGpuThreadsOp;
using iree_compiler::IREE::transform_dialect::PromoteOperandsOp;
using iree_compiler::IREE::transform_dialect::ShareForallOperandsOp;
using iree_compiler::IREE::transform_dialect::VectorToMMAConversionOp;
using iree_compiler::IREE::transform_dialect::VectorToWarpExecuteOnLane0Op;
using iree_compiler::IREE::transform_dialect::VectorWarpDistributionOp;
using transform::FuseIntoContainingOp;
using transform::MatchOp;
using transform::PrintOp;
using transform::ScalarizeOp;
using transform::SequenceOp;
using transform::SplitHandlesOp;
using transform::VectorizeOp;
using transform_ext::StructuredOpMatcher;

using iree_compiler::buildTileReductionUsingScfForeach;
using iree_compiler::gpu::AbstractReductionStrategy;
using iree_compiler::gpu::adjustNumberOfWarpsForBlockShuffle;
using iree_compiler::gpu::build1DSplittingStrategyWithOptionalThreadMapping;
using iree_compiler::gpu::buildCommonTrailingStrategy;
using iree_compiler::gpu::buildDistributeVectors;
using iree_compiler::gpu::ConvolutionImplicitGemmStrategy;
using iree_compiler::gpu::kCudaMaxNumThreads;
using iree_compiler::gpu::kCudaMaxVectorLoadBitWidth;
using iree_compiler::gpu::kCudaWarpSize;
using iree_compiler::gpu::ReductionConfig;
using iree_compiler::gpu::scaleUpByBitWidth;

ConvolutionImplicitGemmStrategy
mlir::iree_compiler::gpu::ConvolutionImplicitGemmStrategy::create(
    MLIRContext *context,
    const transform_ext::MatchedConvolutionCaptures &captures,
    const ConvolutionConfig &convolutionConfig) {
  ConvolutionImplicitGemmStrategy strategy(context, captures);
  strategy.configure(convolutionConfig);
  LLVM_DEBUG(DBGS() << "use GPU convolution implicit gemm strategy\n");
  return strategy;
}

void mlir::iree_compiler::gpu::ConvolutionImplicitGemmStrategy::configure(
    const ConvolutionConfig &convolutionConfig) {
  isSpirv = convolutionConfig.isSpirv;
  int64_t maxNumThreadsToUse = convolutionConfig.maxNumThreads;
  assert(maxNumThreadsToUse > 0 && "maxNumThreadsToUse must be > 0");
  assert(maxNumThreadsToUse >= convolutionConfig.subgroupSize &&
         "need at least a warp?");

  // Block-level
  // ===========

  // Batch dimension
  for (int i = 0, e = captures.convolutionDims.batch.size(); i < e; i++)
    workgroupTileSizes.push_back(1);

  isNchw = captures.convolutionDims.outputChannel[0] <
           captures.convolutionDims.outputImage[0];

  int channelSize = 1;
  int imageSize = 1;
  for (auto dim : captures.convolutionDims.outputChannel)
    channelSize *= captures.convolutionOpSizes[dim];
  for (auto dim : captures.convolutionDims.outputImage)
    imageSize *= captures.convolutionOpSizes[dim];

  int mSize, nSize;
  if (isNchw) {
    mSize = channelSize;
    nSize = imageSize;
  } else {
    mSize = imageSize;
    nSize = channelSize;
  }

  int kSize = 1;
  for (auto dim : captures.convolutionDims.filterLoop)
    kSize *= captures.convolutionOpSizes[dim];
  for (auto dim : captures.convolutionDims.inputChannel)
    kSize *= captures.convolutionOpSizes[dim];

  LLVM_DEBUG(DBGS() << "M size:" << mSize << ", " << mSize % 32 << "\n");
  LLVM_DEBUG(DBGS() << "N size:" << nSize << ", " << nSize % 32 << "\n");
  LLVM_DEBUG(DBGS() << "K size:" << kSize << ", " << kSize % 32 << "\n");

  int64_t mTileSize = 128;
  while (mSize % mTileSize != 0) mTileSize /= 2;
  workgroupTileSizes.push_back(mTileSize);

  int64_t nTileSize = 32;
  while (nSize % nTileSize != 0) nTileSize /= 2;
  workgroupTileSizes.push_back(nTileSize);

  tileM = mTileSize > nTileSize;
  int64_t threadTile = tileM ? mTileSize : nTileSize;

  int64_t im2colTile = isNchw ? nTileSize : mTileSize;

  // Thread-level
  // ============
  numThreadsXInBlock = std::min(
      maxNumThreadsToUse, iree_compiler::nextMultipleOf(
                              threadTile / 2, convolutionConfig.subgroupSize));
  numThreadsXToDistribute = std::min(threadTile, numThreadsXInBlock);
  numThreadsXForIm2Col = std::min(im2colTile, numThreadsXInBlock);
  numWarpsXInBlock = numThreadsXInBlock / convolutionConfig.subgroupSize;

  // Reduction tile size
  innerLoopTileSize = kSize % 32 != 0 ? 16 : 32;
}

/// Builds the transform IR tiling reductions for CUDA targets. Supports
/// reductions in the last dimension, with optional leading and trailing
/// elementwise operations.
void mlir::iree_compiler::gpu::buildConvolutionImplicitGemmStrategy(
    ImplicitLocOpBuilder &b, Value variantH,
    const ConvolutionImplicitGemmStrategy &strategy) {
  LLVM_DEBUG(b.create<PrintOp>(variantH));

  ApplyPatternsOpPatterns emptyConfiguration;
  auto pdlOperationType = pdl::OperationType::get(b.getContext());

  // Step 1. Call the matcher. Note that this is the same matcher as used to
  // trigger this compilation path, so it must always apply.
  b.create<transform_ext::RegisterMatchCallbacksOp>();
  auto [maybeFillH, convolutionH, maybeTrailingH] =
      unpackRegisteredMatchCallback<3>(
          b, "convolution", transform::FailurePropagationMode::Propagate,
          variantH);

  // Step 2. Apply im2col patterns.
  // b.create<ConvertConv2DToImg2ColAndAdjustWorkgroupCountOp>(convolutionH);
  auto img2colWorkgroupOp =
      b.create<ConvertConv2DToImg2ColAndAdjustWorkgroupCountOp>(convolutionH);
  auto img2colH = img2colWorkgroupOp.getImg2colTensor();
  auto transformedH = img2colWorkgroupOp.getTransformed();
  auto matmulH = b.create<transform::GetProducerOfOperand>(pdlOperationType,
                                                           transformedH, 0);

  LLVM_DEBUG(b.create<PrintOp>(variantH));

  // Step 3. Bubble reshapes introduced by im2col to the boundaries of the
  // kernel.
  ApplyPatternsOpPatterns configuration;
  // configuration.bubbleExpand = true;
  configuration.bubbleCollapse = true;
  Value funcH = b.create<MatchOp>(variantH, func::FuncOp::getOperationName());
  b.create<ApplyPatternsOp>(funcH, configuration);

  // Step 4. Create the block/mapping tiling level and fuse.
  auto [fusionTargetH, fusionGroupH] =
      buildSelectFirstNonEmpty(b, maybeTrailingH, matmulH);
  ArrayRef<Attribute> allBlocksRef(strategy.allBlockAttrs);
  TileToForallAndFuseAndDistributeResult tileResult =
      buildTileFuseDistToForallAndWorkgroupCountWithTileSizes(
          /*builder=*/b,
          /*isolatedParentOpH=*/variantH,
          /*rootH=*/fusionTargetH,
          /*opsToFuseH=*/fusionGroupH,
          /*tileSizes=*/
          getAsOpFoldResult(b.getI64ArrayAttr(strategy.workgroupTileSizes)),
          /*threadDimMapping=*/
          b.getArrayAttr(allBlocksRef));

  /// The previous fill handle gets invalidated so we match it again.
  Value newFillH =
      b.create<MatchOp>(variantH, linalg::FillOp::getOperationName());
  maybeFillH =
      b.create<FuseIntoContainingOp>(newFillH, tileResult.forallH).getResult();
  auto tiledImg2colH =
      b.create<FuseIntoContainingOp>(img2colH, tileResult.forallH).getResult();

  LLVM_DEBUG(b.create<PrintOp>(variantH));

  /// Perform a pass of canonicalization + enabling after fusion.
  variantH = buildCanonicalizationAndEnablingTransforms(b, emptyConfiguration,
                                                        variantH);

  // Step 5. Normalize to reorder results irrespective of emptiness.
  auto [blockMatmulH, maybeBlockTrailingH] = buildSelectFirstNonEmpty(
      b, tileResult.resultingFusedOpsHandles.front(), tileResult.tiledOpH);

  // Step 6. Tile the reduction loop
  auto tileToScfForOp = b.create<transform::TileToScfForOp>(
      TypeRange{pdlOperationType, pdlOperationType}, blockMatmulH, ValueRange{},
      strategy.getInnerLoopTileSizes());
  auto innerLoopH = tileToScfForOp.getLoops()[0];
  auto matmulLoopK = tileToScfForOp.getTiledLinalgOp();

  tiledImg2colH =
      b.create<FuseIntoContainingOp>(tiledImg2colH, innerLoopH).getResult();
  // maybeFillH = b.create<FuseIntoContainingOp>(maybeFillH,
  // innerLoopH).getResult();
  variantH = buildCanonicalizationAndEnablingTransforms(b, emptyConfiguration,
                                                        variantH);

  LLVM_DEBUG(b.create<PrintOp>(variantH));

  // Step 7. Promote to shared memory
  auto promoteOperandsOp = b.create<PromoteOperandsOp>(
      TypeRange{pdlOperationType, pdlOperationType}, matmulLoopK,
      b.getDenseI64ArrayAttr(
          ArrayRef<int64_t>{strategy.getImplicitGemmFilterOperandIndex()}));
  Value promotedMatmulH = promoteOperandsOp.getResult()[0];

  // Step 8. Tile img2col, fill, and trailing elementwise to threads
  iree_compiler::buildTileFuseDistToForallWithNumThreads(
      /*b=*/b,
      /*isolatedParentOpH=*/variantH,
      /*rootH=*/tiledImg2colH,
      /*opsHToFuse=*/{},
      /*numThreads=*/
      getAsOpFoldResult(b.getI64ArrayAttr(strategy.getInputTileSizes())),
      /*threadDimMapping=*/b.getArrayAttr({strategy.allThreadAttrs.front()}));

  iree_compiler::buildTileFuseDistToForallWithNumThreads(
      /*b=*/b,
      /*isolatedParentOpH=*/variantH,
      /*rootH=*/maybeBlockTrailingH,
      /*opsHToFuse=*/{},
      /*numThreads=*/
      getAsOpFoldResult(b.getI64ArrayAttr(strategy.getOutputTileSizes())),
      /*threadDimMapping=*/b.getArrayAttr({strategy.allThreadAttrs.front()}));

  iree_compiler::buildTileFuseDistToForallWithNumThreads(
      /*b=*/b,
      /*isolatedParentOpH=*/variantH,
      /*rootH=*/maybeFillH,
      /*opsHToFuse=*/{},
      /*numThreads=*/
      getAsOpFoldResult(b.getI64ArrayAttr(strategy.getOutputTileSizes())),
      /*threadDimMapping=*/b.getArrayAttr({strategy.allThreadAttrs.front()}));

  // Step 9. Tile matmul to warps
  iree_compiler::buildTileFuseDistToForallWithNumThreads(
      /*b=*/b,
      /*isolatedParentOpH=*/variantH,
      /*rootH=*/promotedMatmulH,
      /*opsHToFuse=*/{},
      /*numWarps=*/
      getAsOpFoldResult(b.getI64ArrayAttr(strategy.getWarpsTileSizes())),
      /*threadDimMapping=*/b.getArrayAttr({strategy.allWarpAttrs.front()}));

  LLVM_DEBUG(b.create<PrintOp>(variantH));

  // Step 10. Vectorize and unroll to wmma sizes
  funcH = b.create<MatchOp>(variantH, func::FuncOp::getOperationName());

  ApplyPatternsOpPatterns vectorizeConfiguration;
  vectorizeConfiguration.rankReducingLinalg = true;
  vectorizeConfiguration.rankReducingVector = true;
  b.create<ApplyPatternsOp>(funcH, vectorizeConfiguration);
  funcH = b.create<VectorizeOp>(funcH, /*vectorizePadding=*/false,
                                /*vectorizeExtract=*/true);

  // Basically a hack to find the parent forall loop of the matmul for wmma
  // unrolling.
  auto forallOpsH =
      b.create<MatchOp>(variantH, scf::ForallOp::getOperationName());
  int numLoops = 3;
  int resultPos = 1;
  if (strategy.captures.maybeFillElementalTypeBitWidth > 0) {
    numLoops++;
    resultPos++;
  }
  if (strategy.captures.maybeTrailingOutputElementalTypeBitWidth > 0)
    numLoops++;
  Value matmulLoop =
      b.create<SplitHandlesOp>(forallOpsH, numLoops)->getResult(resultPos);

  ApplyPatternsOpPatterns unrollConfiguration;
  unrollConfiguration.unrollVectorsGpuWmma = true;
  b.create<ApplyPatternsToNestedOp>(matmulLoop, unrollConfiguration);

  LLVM_DEBUG(b.create<PrintOp>(variantH));

  // Step 11. Bufferize
  ApplyPatternsOpPatterns foldConfiguration;
  foldConfiguration.foldReassociativeReshapes = true;
  b.create<ApplyPatternsOp>(funcH, foldConfiguration);

  b.create<IREEEliminateEmptyTensorsOp>(variantH);
  funcH = b.create<MatchOp>(variantH, func::FuncOp::getOperationName());

  ApplyPatternsOpPatterns eraseConfiguration;
  eraseConfiguration.eraseUnnecessaryTensorOperands = true;
  b.create<ApplyPatternsOp>(funcH, eraseConfiguration);

  variantH = b.create<IREEBufferizeOp>(variantH, /*targetGPU=*/true);
  variantH = buildCanonicalizationAndEnablingTransforms(b, emptyConfiguration,
                                                        variantH);

  LLVM_DEBUG(b.create<PrintOp>(variantH));

  // Step 12. Post-bufferization mapping to blocks and threads
  funcH = b.create<MatchOp>(variantH, func::FuncOp::getOperationName());
  funcH = buildMapToBlockAndThreads(b, funcH, strategy.getNumThreadsInBlock());
  b.create<HoistStaticAllocOp>(funcH);
  b.create<GpuDistributeSharedMemoryCopyOp>(funcH);
  ApplyPatternsOpPatterns distributeConfiguration;
  eraseConfiguration.foldMemrefAliases = true;
  variantH = buildCanonicalizationAndEnablingTransforms(
      b, distributeConfiguration, variantH);

  // Note: Don't do this for SPIR-V
  b.create<IREEEraseHALDescriptorTypeFromMemRefOp>(funcH);

  b.create<VectorToMMAConversionOp>(funcH, /*useMmaSync=*/false,
                                    /*useWmma=*/true);
  variantH = buildCanonicalizationAndEnablingTransforms(b, emptyConfiguration,
                                                        variantH);

  LLVM_DEBUG(b.create<PrintOp>(variantH));
}

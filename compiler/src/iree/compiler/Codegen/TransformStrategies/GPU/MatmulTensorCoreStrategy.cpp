// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/TransformStrategies/GPU/MatmulTensorCoreStrategy.h"

#include "iree-dialects/Dialect/LinalgTransform/StructuredTransformOpsExt.h"
#include "iree/compiler/Codegen/Common/TransformExtensions/CommonExtensions.h"
#include "iree/compiler/Codegen/LLVMGPU/TransformExtensions/LLVMGPUExtensions.h"
#include "iree/compiler/Codegen/TransformStrategies/Common/Common.h"
#include "iree/compiler/Codegen/TransformStrategies/GPU/Common.h"
#include "iree/compiler/Codegen/TransformStrategies/GPU/MappingInfo.h"
#include "iree/compiler/Codegen/TransformStrategies/GPU/Strategies.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Dialect/Vector/TransformOps/VectorTransformOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

using namespace mlir;

#define DEBUG_TYPE "iree-transform-builder"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

// TODO: significantly better namespacing.
using iree_compiler::buildPad;
using iree_compiler::buildTileFuseDistToForallWithNumThreads;
using iree_compiler::buildTileFuseDistToForallWithTileSizes;
using iree_compiler::TileToForallAndFuseAndDistributeResult;
using iree_compiler::gpu::buildBufferize;
using iree_compiler::gpu::buildConvertToAsyncCopies;
using iree_compiler::gpu::buildConvertToTensorCoreOp;
using iree_compiler::gpu::buildDistributeMatmulCopies;
using iree_compiler::gpu::buildHoistOutputPaddingOp;
using iree_compiler::gpu::buildMatmulVectorization;
using iree_compiler::gpu::buildMultiBuffering;
using iree_compiler::gpu::buildPipelineSharedMemoryCopies;
using iree_compiler::gpu::kCudaWarpSize;
using iree_compiler::gpu::MappingInfo;
using iree_compiler::gpu::MatmulStrategy;
using iree_compiler::gpu::scaleUpByBitWidth;
using iree_compiler::IREE::transform_dialect::EliminateGpuBarriersOp;
using iree_compiler::IREE::transform_dialect::
    IREEPopulateWorkgroupCountRegionUsingNumThreadsSliceOp;
using transform::MatchOp;
using transform_ext::RegisterMatchCallbacksOp;

void MatmulStrategy::initDefaultValues() {
  // Set the configuration for padding the matmul.
  paddingValueTypes = {captures.lhsElementType, captures.rhsElementType,
                       captures.outputElementType};
  paddingDimensions = {0, 1, 2};
  packingDimensions = {1, 1, 1};

  lhsElementalBitWidth = captures.lhsElementType.getIntOrFloatBitWidth();
  rhsElementalBitWidth = captures.rhsElementType.getIntOrFloatBitWidth();
  resElementalBitWidth = captures.outputElementType.getIntOrFloatBitWidth();

  // Pull in tile configs from flags.
  AbstractGemmLikeStrategy::initDefaultValues();
}

LLVM_DUMP_METHOD void MatmulStrategy::dump() const { print(llvm::errs()); }

void MatmulStrategy::print(llvm::raw_ostream &os) const {
  os << "\n--- Matmul strategy ---\n";
  AbstractGemmLikeStrategy::print(os);
}

LogicalResult MatmulStrategy::validate(const GPUModel &gpuModel) const {
  // Validate the parent strategy.
  if (failed(AbstractGemmLikeStrategy::validate(gpuModel)))
    return failure();

  return success();
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
  MappingInfo blockMapping = strategy.getBlockMapping();
  TileToForallAndFuseAndDistributeResult tileResult =
      buildTileFuseDistToForallWithTileSizes(
          /*builder=*/b,
          /*variantH=*/variantH,
          /*rootH=*/matmulH,
          /*opsToFuseH=*/fillH,
          /*tileSizes=*/
          getAsOpFoldResult(b.getI64ArrayAttr(blockMapping.tileSizes)),
          /*threadDimMapping=*/
          b.getArrayAttr(blockMapping.threadMapping));

  // Handle the workgroup count region.
  b.create<IREEPopulateWorkgroupCountRegionUsingNumThreadsSliceOp>(
      tileResult.forallH);

  // TODO: handle trailing op.
  return std::make_tuple(tileResult.resultingFusedOpsHandles.front(),
                         tileResult.tiledOpH, Value(), tileResult.forallH);
}

void iree_compiler::gpu::buildMatmulTensorCoreStrategy(
    ImplicitLocOpBuilder &b, Value variantH, const MatmulStrategy &strategy) {
  LLVM_DEBUG(strategy.print(DBGS()));

  // Step 1. Apply block-level part of the strategy, keeps everything fused.
  auto [fillH, matmulH, maybeTiledTrailingHBlock, forall] =
      buildMatmulStrategyBlockDistribution(b, variantH, strategy);
  // Tile reduction loop.
  SmallVector<int64_t> tileSizes{0, 0, strategy.reductionTileSize};
  // Avoid canonicalizing before the pad to avoid folding away the extract_slice
  // on the output needed to hoist the output pad.
  auto tileReductionResult = buildTileFuseToScfFor(
      b, variantH, matmulH, {}, getAsOpFoldResult(b.getI64ArrayAttr(tileSizes)),
      /*canonicalize=*/false);

  // Step 2. Pad the matmul op.
  auto paddedMatmulOpH =
      buildPad(b, tileReductionResult.tiledOpH,
               strategy.getZeroPadAttrFromElementalTypes(b).getValue(),
               strategy.paddingDimensions, strategy.packingDimensions);

  // Step 3. Hoist the padding of the output operand above the reduction loop.
  // The resulting fillOp will be mapped with the contraction using an SIMD
  // programming model.
  Value fillOpH = fillH;
  if (!strategy.alignedRes()) {
    fillOpH = buildHoistOutputPaddingOp(b, variantH, paddedMatmulOpH);
  }

  // Running canonicalization is required here to enable aligned pads to become
  // linalg.copy ops when rewriting in DPS.
  Value funcH =
      b.create<transform::MatchOp>(variantH, func::FuncOp::getOperationName());
  iree_compiler::buildCanonicalizationAndEnablingTransforms(b, funcH);

  // Step 4. Distribute pad and copies: SIMT programming model.
  auto [lhsCopyOpH, rhsCopyOpH, copyBackOpH] =
      buildDistributeMatmulCopies(b, variantH, paddedMatmulOpH, strategy);

  // Step 5. Distribute to warps: SIMD programming model.
  // TODO: get the number of warps from strategy.
  MappingInfo computeMapping = strategy.computeMapping();
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
  variantH = buildBufferize(b, variantH);

  // Step 8. Post-bufferization mapping to blocks and threads.
  // Need to match again since bufferize invalidated all handles.
  // TODO: assumes a single func::FuncOp to transform, needs hardening.
  funcH = b.create<MatchOp>(variantH, func::FuncOp::getOperationName());
  funcH = buildMapToBlockAndThreads(b, funcH, strategy.numThreads,
                                    strategy.numWarps);
  funcH = b.create<EliminateGpuBarriersOp>(funcH);

  // Step 9. Convert to tensor core ops.
  // TODO: avoid consuming handles and returning here.
  funcH = buildConvertToTensorCoreOp(b, funcH, strategy);

  if (strategy.useAsyncCopies) {
    // Step 10. Multi-buffering.
    if (strategy.pipelineDepth > 1)
      buildMultiBuffering(b, funcH, strategy);

    // Step 11. Convert to async copies.
    // TODO: avoid consuming handles and returning here.
    funcH = buildConvertToAsyncCopies(b, funcH, strategy);

    // Step 12. Pipeline shared memory copies.
    if (strategy.pipelineDepth > 1)
      buildPipelineSharedMemoryCopies(b, funcH, strategy);
  }

  // Step 13. Late lowerings and cleanups.
  buildLowerVectorMasksAndCleanup(b, funcH);
}

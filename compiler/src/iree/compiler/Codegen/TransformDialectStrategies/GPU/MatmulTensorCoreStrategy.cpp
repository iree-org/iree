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
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/TransformOps/SCFTransformOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
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

static SmallVector<Attribute> getBlockAttrs(MLIRContext *context) {
  auto threadX =
      mlir::gpu::GPUThreadMappingAttr::get(context, mlir::gpu::Threads::DimX);
  auto threadY =
      mlir::gpu::GPUThreadMappingAttr::get(context, mlir::gpu::Threads::DimY);
  return SmallVector<Attribute>{threadY, threadX};
}

static SmallVector<Attribute> getWarpAttrs(MLIRContext *context) {
  auto warpX =
      mlir::gpu::GPUWarpMappingAttr::get(context, mlir::gpu::Warps::DimX);
  auto warpY =
      mlir::gpu::GPUWarpMappingAttr::get(context, mlir::gpu::Warps::DimY);
  return SmallVector<Attribute>{warpY, warpX};
}

static Value buildPadMatmul(ImplicitLocOpBuilder &b, Value isolatedParentOpH,
                            Value rootH) {
  SmallVector<float> paddingValues = {0.f, 0.f, 0.f};
  SmallVector<int64_t> paddingDimensions = {0, 1, 2};
  SmallVector<int64_t> packingDimensions = {1, 1, 1};
  Value padResult = b.create<transform::PadOp>(
      rootH.getType(), rootH, b.getF32ArrayAttr(paddingValues),
      b.getI64ArrayAttr(paddingDimensions),
      b.getI64ArrayAttr(packingDimensions), ArrayAttr());

  // Perform a pass of canonicalization + enabling after tiling.
  ApplyPatternsOpPatterns configuration;
  isolatedParentOpH =
      mlir::iree_compiler::buildCanonicalizationAndEnablingTransforms(
          b, configuration, isolatedParentOpH);
  return padResult;
}

static std::tuple<Value, Value> buildHoistOutputPaddingOp(
    ImplicitLocOpBuilder &b, Value variantH, Value paddedMatmulH) {
  // Find the output pad and hoist it.
  Value outputH = b.create<transform::GetProducerOfOperand>(
      paddedMatmulH.getType(), paddedMatmulH, b.getI64IntegerAttr(2));
  auto padOpType = transform::OperationType::get(
      b.getContext(), tensor::PadOp::getOperationName());
  outputH = b.create<transform::CastOp>(padOpType, outputH);
  Value hoistPad =
      b.create<transform::HoistPadOp>(paddedMatmulH.getType(), outputH, 1);
  // Convert the insert slice from the hoisting to a copy.
  Value insertSliceH = b.create<transform::MatchOp>(
      variantH, tensor::InsertSliceOp::getOperationName());
  Value copyH = b.create<transform::InsertSliceToCopyOp>(insertSliceH.getType(),
                                                         insertSliceH);

  // Perform a pass of canonicalization clean.
  ApplyPatternsOpPatterns configuration;
  variantH = mlir::iree_compiler::buildCanonicalizationAndEnablingTransforms(
      b, configuration, variantH);

  return std::make_tuple(hoistPad, copyH);
}

static SmallVector<Value> buildDistributeCopyOps(ImplicitLocOpBuilder &b,
                                                 Value variantH,
                                                 ArrayRef<Value> copyOps,
                                                 bool foldIfBranch = false) {
  SmallVector<Value> results;
  for (Value copyOp : copyOps) {
    auto tileOp = b.create<transform::TileToForallOp>(
        copyOp, ArrayRef<int64_t>({4, 32}), transform::TileSizesSpec(),
        b.getArrayAttr(getBlockAttrs(b.getContext())));
    results.push_back(tileOp.getResult(1));
    if (foldIfBranch) {
      Value ifOpH = b.create<transform::MatchOp>(tileOp.getResult(0),
                                                 scf::IfOp::getOperationName());
      b.create<transform::TakeAssumedBranchOp>(ifOpH, b.getUnitAttr());
    }
  }
  return results;
}

static std::tuple<Value, Value, Value, Value>
buildMamtulStrategyBlockDistribution(ImplicitLocOpBuilder &b, Value variantH,
                                     const MatmulStrategy &strategy) {
  // Step 1. Call the matcher. Note that this is the same matcher as used to
  // trigger this compilation path, so it must always apply.
  b.create<RegisterMatchCallbacksOp>();
  auto [fillH, matmulH, maybeTrailingH] = unpackRegisteredMatchCallback<3>(
      b, "matmul", transform::FailurePropagationMode::Propagate, variantH);

  // TODO: get from strategy.
  SmallVector<int64_t> tileSizes = {128, 128};
  // Step 2. Create the block/mapping tiling level and fusee.
  // auto [fusionTargetH, fusionGroupH] =
  //     buildSelectFirstNonEmpty(b, maybeTrailingH, matmulH);
  TileToForallAndFuseAndDistributeResult tileResult =
      buildTileFuseDistToForallAndWorkgroupCountWithTileSizes(
          /*builder=*/b,
          /*isolatedParentOpH=*/variantH,
          /*rootH=*/matmulH,
          /*opsToFuseH=*/fillH,
          /*tileSizes=*/
          getAsOpFoldResult(b.getI64ArrayAttr(tileSizes)),
          /*threadDimMapping=*/
          b.getArrayAttr(getBlockAttrs(b.getContext())));

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
  variantH = b.create<IREEBufferizeOp>(variantH, /*targetGpu=*/true);
  Value memrefFunc =
      b.create<MatchOp>(variantH, func::FuncOp::getOperationName());
  b.create<IREEEraseHALDescriptorTypeFromMemRefOp>(memrefFunc);
  b.create<ApplyBufferOptimizationsOp>(memrefFunc);
  return variantH;
}

void mlir::iree_compiler::gpu::buildMatmulTensorCoreStrategy(
    ImplicitLocOpBuilder &b, Value variantH, const MatmulStrategy &strategy) {
  // Step 1. Apply block-level part of the strategy, keeps everything fused.
  auto [fillH, matmulH, maybeTiledTrailingHBlock, forall] =
      buildMamtulStrategyBlockDistribution(b, variantH, strategy);

  // Tile reduction loop.
  SmallVector<int64_t> tileSizes(2, 0);
  tileSizes.push_back(16);
  auto res =
      buildTileFuseToScfFor(b, variantH, matmulH, {},
                            getAsOpFoldResult(b.getI64ArrayAttr(tileSizes)));

  // Step 2. Pad the matmul op.
  auto paddedMatmul = buildPadMatmul(b, variantH, res.tiledOpH);

  // Step 3. Hoist the padding of the output operand above the reduction loop.
  auto [hoistedPadOpH, copyH] =
      buildHoistOutputPaddingOp(b, variantH, paddedMatmul);

  // Step 4. Distribute pad ops.
  Value lhsH = b.create<transform::GetProducerOfOperand>(
      paddedMatmul.getType(), paddedMatmul, b.getI64IntegerAttr(0));
  Value rhsH = b.create<transform::GetProducerOfOperand>(
      paddedMatmul.getType(), paddedMatmul, b.getI64IntegerAttr(1));
  SmallVector<Value> distributedOps = buildDistributeCopyOps(
      b, variantH, {lhsH, rhsH, hoistedPadOpH}, /*foldIfBranch=*/true);
  distributedOps.append(buildDistributeCopyOps(b, variantH, {copyH}));
  for (Value v : distributedOps) {
    b.create<transform::MaskedVectorizeOp>(v, ValueRange(), false,
                                           ArrayRef<int64_t>({4, 4}));
  }

  // TODO: get the number of warps from strategy.
  std::array<int64_t, 2> numThreads = {2, 2};
  // Step 5. Distribute to warps.
  buildTileFuseDistToForallWithNumThreads(
      b, variantH, paddedMatmul, ValueRange(),
      getAsOpFoldResult(b.getI64ArrayAttr(numThreads)),
      b.getArrayAttr(getWarpAttrs(b.getContext())));

  // Step 6. Rank-reduce and vectorize.
  Value funcH =
      b.create<transform::MatchOp>(variantH, func::FuncOp::getOperationName());
  funcH = b.create<transform::LowerMaskedTransfersOp>(funcH.getType(), funcH);
  ApplyPatternsOpPatterns patterns;
  patterns.rankReducingLinalg = true;
  patterns.rankReducingVector = true;
  b.create<ApplyPatternsOp>(funcH, patterns);
  b.create<transform::VectorizeOp>(funcH);

  // Step 7. Bufferize and drop HAL descriptor from memref ops.
  variantH = buildBufferize(b, variantH);
}

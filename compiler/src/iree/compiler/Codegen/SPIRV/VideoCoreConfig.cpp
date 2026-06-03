// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- VideoCoreConfig.cpp - VideoCore CodeGen Configurations -------------===//
//
// This file contains CodeGen configurations for Broadcom VideoCore GPUs.
//
//===----------------------------------------------------------------------===//

#include <array>
#include <cstdint>

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/SPIRV/KernelConfig.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"

#define DEBUG_TYPE "iree-spirv-videocore-config"

namespace mlir::iree_compiler::detail {

/// Repeatedly halves `startValue` until it evenly divides `dim`, returning the
/// result (at least 1). When `startValue` is a power of two this is the largest
/// power-of-two <= `startValue` that divides `dim`. `startValue` is not
/// required to be a power of two.
static int64_t largestDivisorByHalving(int64_t dim, int64_t startValue) {
  if (dim < 0) {
    return 1;
  }
  while (dim % startValue != 0) {
    startValue >>= 1;
  }
  return std::max(startValue, int64_t(1));
}

/// Sets a SPIR-V "BaseVectorize" lowering config for a (batch) matmul on the
/// Broadcom VideoCore VII GPU.
///
/// The op is treated as the standard contraction (Bx)MxK * (Bx)KxN = (Bx)MxN,
/// with the B/M/N/K loop indices. Only static shapes of f32 are handled
/// everything else will be rejected.
///
/// Four levels of tiling are selected:
///   1. Workgroup size - aim for about ~256 invocations
///   2. Workgroup tile - the M tile is the largest power-of-two dividing M up
///   to
///      ~4x the workgroup X count. The N tile takes the rest of a 1024-element
///      budget. This biases sequential loads toward M.
///   3. Thread tile - each invocation handles 1/16th of the workgroup tile in
///      M and N (at least 1).
///   4. Reduction tile - the K loop is vectorized by the largest power-of-two
///      dividing K, capped at 4.
LogicalResult
setMatmulOpVideoCoreConfig(IREE::GPU::TargetAttr target, linalg::LinalgOp op,
                           std::array<int64_t, 2> bestWorkgroupSizeXY,
                           std::array<int64_t, 3> bestThreadTileSizeMNK) {
  LLVM_DEBUG(llvm::dbgs() << "trying to deduce config as matmul...\n");
  OpOperand *lhs = op.getDpsInputOperand(0);
  OpOperand *rhs = op.getDpsInputOperand(1);

  // The following tiling heuristic will ignore any operations that do not
  // have statically known shapes.
  auto lhsType = llvm::cast<ShapedType>(lhs->get().getType());
  auto rhsType = llvm::cast<ShapedType>(rhs->get().getType());
  // This routine can in principle handle 8/16/32-bit element types, but the
  // caller (setVideoCoreMatmulConfig) currently restricts inputs to f32.
  auto elementBits =
      static_cast<int>(IREE::Util::getTypeBitWidth(lhsType.getElementType()));
  if (!llvm::is_contained({8, 16, 32}, elementBits)) {
    return failure();
  }

  ArrayRef<int64_t> lhsShape = lhsType.getShape();
  ArrayRef<int64_t> rhsShape = rhsType.getShape();
  if (llvm::any_of(lhsShape, ShapedType::isDynamic)) {
    return failure();
  }
  if (llvm::any_of(rhsShape, ShapedType::isDynamic)) {
    return failure();
  }

  // Ensure that we only focus on batch matmuls or single matmuls, i.e. 2D or 3D
  assert(llvm::is_contained({2u, 3u}, op.getNumParallelLoops()));

  // Find the loop indices for the B, M, N, K dimensions. We tile the standard
  // matmul (Bx)MxK * (Bx)KxN = (Bx)MxN, where M is the LHS parallel dim, N the
  // RHS parallel dim and K the reduction dim (see getMatmulBMNKIndex).
  int lastParallelDim = -1;
  const auto [bIndex, mIndex, nIndex, kIndex] =
      getMatmulBMNKIndex(op, &lastParallelDim);
  if (mIndex < 0 || nIndex < 0 || kIndex < 0) {
    return failure();
  }
  const bool isBM = bIndex >= 0;

  // Get all the dimension sizes that we need for tiling.
  SmallVector<int64_t> loopRanges = op.getStaticLoopRanges();
  const unsigned numLoops = loopRanges.size();
  const int64_t dimM = loopRanges[mIndex];
  const int64_t dimN = loopRanges[nIndex];
  const int64_t dimK = loopRanges[kIndex];

  int64_t bestX = bestWorkgroupSizeXY[0], bestY = bestWorkgroupSizeXY[1];
  LLVM_DEBUG({
    llvm::dbgs() << "best thread tile size (M, N, K) = ("
                 << bestThreadTileSizeMNK[0] << ", " << bestThreadTileSizeMNK[1]
                 << ", " << bestThreadTileSizeMNK[2] << ")\n";
    llvm::dbgs() << "best workgroup size (X, Y) = (" << bestX << ", " << bestY
                 << ")\n";
  });

  // Aim for ~256 invocations per workgroup, balanced to the output shape: the Y
  // (N) dimension uses half of dimN capped at 16, and X (M) takes the rest of
  // the ~256 budget (capped at dimM). These larger workgroups are load-bearing
  // for the VideoCore VII -- sizing the workgroup down to match the tile
  // measured several times slower on the V3D.
  SmallVector<int64_t, 3> workgroupSize(3, 1); // (X, Y, Z)
  workgroupSize[1] = std::min(dimN >> 1, int64_t(16));
  workgroupSize[0] =
      std::min(std::max((bestX * bestY) / workgroupSize[1], int64_t(1)), dimM);

  SmallVector<int64_t> workgroupTileSizes(numLoops, 0);
  // Batch is simply tiled to 1 for now. Could be improved if batch is
  // large and the spatial dimensions are small compared to the amount
  // of available threads.
  if (isBM) {
    workgroupTileSizes[bIndex] = 1;
  }
  // To maximize the number of elements loaded sequentially we give the M
  // dimension the larger workgroup tile and hand the rest of the budget (1024
  // elements total) to the N dimension.
  workgroupTileSizes[mIndex] =
      largestDivisorByHalving(dimM, workgroupSize[0] * 4);
  workgroupTileSizes[nIndex] =
      largestDivisorByHalving(dimN, 1024 / workgroupTileSizes[mIndex]);

  // Thread Tiling
  SmallVector<int64_t> threadTileSizes(numLoops, 0);
  // Batch is simply tiled to 1 for now. Remember workgroup is 1 and tile size
  // is 1
  if (isBM) {
    threadTileSizes[bIndex] = workgroupTileSizes[bIndex] / workgroupSize[2];
  }
  // Each thread is given 1/16th of what the workgroup was given. Which should
  // be skewed in favour of the M dimension to allow for coalesced memory
  // accesses.
  threadTileSizes[mIndex] =
      std::max(workgroupTileSizes[mIndex] / 16, int64_t(1));
  threadTileSizes[nIndex] =
      std::max(workgroupTileSizes[nIndex] / 16, int64_t(1));

  // The reduction tiling determines the width of the vector multiply-add in the
  // inner loop. The largest power-of-two that divides K (so the K loop tiles
  // evenly), searched downward from maxVectorization and capped at 4.
  SmallVector<int64_t> reductionTileSizes(numLoops, 0);
  // 32 is an empirical upper bound from the original tuning. It has no
  // known correspondence to a specific hardware property.
  int64_t maxVectorization = 32;
  reductionTileSizes[kIndex] =
      std::min(largestDivisorByHalving(dimK, maxVectorization), int64_t(4));

  workgroupTileSizes.resize(lastParallelDim + 1);
  threadTileSizes.resize(lastParallelDim + 1);

  TileSizesListType tileSizes;
  llvm::append_values(tileSizes, workgroupTileSizes, threadTileSizes,
                      reductionTileSizes);

  LLVM_DEBUG({
    llvm::dbgs() << "workgroup size (X, Y, X) = (" << workgroupSize[0] << ", "
                 << workgroupSize[1] << ", " << workgroupSize[2] << ")\n";
    llvm::dbgs() << "workgroup tiling (M, N) = (" << workgroupTileSizes[mIndex]
                 << ", " << workgroupTileSizes[nIndex] << ")\n";
    llvm::dbgs() << "thread tiling (M, N) = (" << threadTileSizes[mIndex]
                 << ", " << threadTileSizes[nIndex] << ")\n";
    llvm::dbgs() << "reduction tiling (M, N, k) = ("
                 << reductionTileSizes[mIndex] << ", "
                 << reductionTileSizes[nIndex] << ','
                 << reductionTileSizes[kIndex] << ")\n";
  });

  // Sets the workgroup size on the dispatch function and adds the tiling for
  // the MatMul to the corresponding linalg operation.
  MLIRContext *ctx = op->getContext();
  auto config = IREE::Codegen::LoweringConfigAttr::get(ctx, tileSizes);
  auto pipelineAttr = IREE::GPU::SPIRVPipelineAttr::get(
      ctx, IREE::GPU::SPIRVLoweringPipeline::BaseVectorize);
  auto translationInfo = IREE::Codegen::TranslationInfoAttr::get(
      ctx, pipelineAttr, SymbolRefAttr(), workgroupSize, /*subgroupSize=*/0,
      DictionaryAttr());
  auto funcOp = op->getParentOfType<mlir::FunctionOpInterface>();
  return setOpConfigAndEntryPointFnTranslation(funcOp, op, config,
                                               translationInfo);
}

static LogicalResult setVideoCoreMatmulConfig(linalg::LinalgOp op,
                                              IREE::GPU::TargetAttr target) {
  auto inputType =
      llvm::cast<ShapedType>(op.getDpsInputOperand(0)->get().getType());
  // Restrict the tilings to just float32 as we have not investigated the
  // optimal tiling for f16 or other types yet.
  if (!inputType.getElementType().isF32()) {
    return failure();
  }
  const std::array<int64_t, 2> workgroupXY = {16, 16};
  const std::array<int64_t, 3> threadMNK = {4, 4, 4};
  return setMatmulOpVideoCoreConfig(target, op, workgroupXY, threadMNK);
}

static int64_t getWorkgroupTiling(int64_t &remainingThreads,
                                  int64_t dimensionSize) {
  // Find the largest power of two value that can evenly divide the dimension
  // and take that away from the pool of available threads.
  auto result = dimensionSize & (~(dimensionSize - 1));
  remainingThreads = std::max(result / remainingThreads, int64_t(1));
  return result;
}

static LogicalResult setConvOpForVideoCoreConfig(linalg::LinalgOp op,
                                                 int64_t threadsPerWorkGroup,
                                                 int64_t optimalThreadTiling) {
  auto convDimsOrFailure = linalg::inferConvolutionDims(op);
  if (failed(convDimsOrFailure)) {
    return failure();
  }
  auto inputType =
      llvm::cast<ShapedType>(op.getDpsInputOperand(0)->get().getType());
  // Restrict the tilings to just float32 as we have not investigated the
  // optimal tiling for f16 or other types yet.
  if (!inputType.getElementType().isF32()) {
    return failure();
  }
  const mlir::linalg::ConvolutionDimensions &convDims = *convDimsOrFailure;

  LLVM_DEBUG({
    llvm::dbgs() << "conv: " << op;
    llvm::dbgs() << "\nconv batch dim: ";
    llvm::interleaveComma(convDims.batch, llvm::dbgs());
    llvm::dbgs() << "\nconv output window dims: ";
    llvm::interleaveComma(convDims.outputImage, llvm::dbgs());
    llvm::dbgs() << "\nconv output channel dim: ";
    llvm::interleaveComma(convDims.outputChannel, llvm::dbgs());
    llvm::dbgs() << "\nconv filter window dims: ";
    llvm::interleaveComma(convDims.filterLoop, llvm::dbgs());
    llvm::dbgs() << "\nconv input channel dims: ";
    llvm::interleaveComma(convDims.inputChannel, llvm::dbgs());
    llvm::dbgs() << "\nconv depth multiplier: ";
    llvm::interleaveComma(convDims.depth, llvm::dbgs());
    llvm::dbgs() << "\n";
  });

  SmallVector<int64_t> loopRanges = op.getStaticLoopRanges();
  const int ohIndex = convDims.outputImage.front();
  const int owIndex = convDims.outputImage.back();
  const int64_t oh = loopRanges[ohIndex];
  const int64_t ow = loopRanges[owIndex];

  int ocIndex;
  if (!convDims.outputChannel.empty()) {
    assert(convDims.outputChannel.size() == 1);
    ocIndex = convDims.outputChannel.front();
  } else if (!convDims.depth.empty()) {
    // For depthwise convolution ops with multiplier 1, we have the same
    // input/filter/output channel size, which is being categorized as the
    // multiplier.
    assert(convDims.depth.size() == 1);
    ocIndex = convDims.depth.front();
  } else {
    // For pooling ops, the input/output channel size will be categorized
    // as the additional batch dimension.
    assert(convDims.batch.size() == 2);
    ocIndex = convDims.batch.back();
  }

  Type outputType = op.getDpsInitOperand(0)->get().getType();
  ArrayRef<int64_t> outputShape = llvm::cast<ShapedType>(outputType).getShape();
  if ((convDims.inputChannel.empty() ||
       ShapedType::isDynamic(convDims.inputChannel.front())) ||
      llvm::any_of(outputShape.drop_front(), ShapedType::isDynamic)) {
    return failure();
  }

  // Output tilings for the SPIRV Vectorize Pipeline. There are 4 stages to
  // tiling with this particular pipeline. Workgroup, Thread, Reduction and
  // Vector(Window).
  TileSizesListType tileSizes;
  const int64_t outputChannelSize = loopRanges[ocIndex];

  const bool isNCHW = ocIndex < ohIndex;
  if (!isNCHW) {
    // TODO: implement the NHWC case
    return failure();
  }

  // Workgroup tiling is calculated by finding the largest power-of-two value
  // that is an exact multiple of that output dimension size. As we are
  // performing a convolution, the input elements required for one output is 2D
  // with a size of FxF, meaning the filter size. The algorithm below will try
  // to find the largest tile in the OW dimension first and subtract from the
  // available threads for the later dimensions. The idea being all threads will
  // need generally the same row from the input tensor when calculating their
  // output. Generating a sequential access pattern.
  SmallVector<int64_t> workgroupTiling(4, 1); // (N, OC, OH, OW)
  workgroupTiling[3] = getWorkgroupTiling(threadsPerWorkGroup, ow);
  workgroupTiling[2] = getWorkgroupTiling(threadsPerWorkGroup, oh);
  workgroupTiling[1] =
      getWorkgroupTiling(threadsPerWorkGroup, outputChannelSize);
  tileSizes.push_back(workgroupTiling);

  // Remember OC->Z, OW->Y and OH->X
  // The rule for the workgroup size is as follows:
  //    X = WG_TILING_X / 4
  //    Y = WG_TILING_Y
  //    Z = WG_TILING_Z / 4
  // This configuration seems to work best for the GPU. There is no concrete
  // reasoning behind it.
  SmallVector<int64_t, 3> workgroupSize(3, 1); // (X, Y, Z)
  workgroupSize[0] = std::max(workgroupTiling[3] / 4, int64_t(1));
  workgroupSize[1] = workgroupTiling[2];
  workgroupSize[2] = std::max(workgroupTiling[1] / 4, int64_t(1));

  // For each thread we want to maximize the number of output elements in OW
  // that each thread calculates. That is why we do not calculate the output
  // elements of the other dimensions but simply the ones in the OW row.
  // Attempting to keep the memory accesses from the input matrix
  // in-line for all the output elements calculated in a thread.
  SmallVector<int64_t> threadTiling = {1, 1, 1,
                                       optimalThreadTiling}; // (N, OC, OH, OW)
  tileSizes.push_back(threadTiling);

  // The reduction tiling is the operation performing the convolution and is
  // usually where the vectorization is applied. I.e. the vector-multiply-add
  // operation.
  SmallVector<int64_t> reductionTiling(loopRanges.size(), 0);
  reductionTiling[convDims.inputChannel.front()] = 4;
  reductionTiling[convDims.filterLoop.front()] = 1;
  reductionTiling[convDims.filterLoop.back()] = 1;
  tileSizes.push_back(reductionTiling);

  // Tile along OH by size 1 to enable downsizing 2-D convolution to 1-D.
  // Meaning [N, OC, OH, OW] will be likely something like [1, 1, 1, OW]
  // So the loop performing the convolution will traverse only over a single
  // row. Keeping the memory accesses coalesced.
  SmallVector<int64_t> windowTileSizes(4, 0);
  windowTileSizes[ohIndex] = 1;
  tileSizes.push_back(windowTileSizes);

  MLIRContext *ctx = op->getContext();
  auto config = IREE::Codegen::LoweringConfigAttr::get(ctx, tileSizes);
  auto pipelineAttr = IREE::GPU::SPIRVPipelineAttr::get(
      ctx, IREE::GPU::SPIRVLoweringPipeline::BaseVectorize);
  auto translationInfo = IREE::Codegen::TranslationInfoAttr::get(
      ctx, pipelineAttr, SymbolRefAttr(), workgroupSize, /*subgroupSize=*/0,
      DictionaryAttr());
  auto funcOp = op->getParentOfType<mlir::FunctionOpInterface>();
  // This will annotate the dispatch function with the workgroup size and the
  // linalg.conv operation with the tilings we have chosen.
  return setOpConfigAndEntryPointFnTranslation(funcOp, op, config,
                                               translationInfo);
}

//===----------------------------------------------------------------------===//
// Entry Point
//===----------------------------------------------------------------------===//

LogicalResult setVideoCoreCodeGenConfig(IREE::GPU::TargetAttr target,
                                        Operation *rootOp) {
  if (!isa<linalg::LinalgOp>(rootOp)) {
    return failure();
  }

  auto linalgOp = cast<linalg::LinalgOp>(rootOp);
  if (isMatmulOrBatchMatmul(linalgOp) || isa<linalg::MatmulOp>(linalgOp) ||
      isa<linalg::BatchMatmulOp>(linalgOp)) {
    return setVideoCoreMatmulConfig(linalgOp, target);
  }

  // The IREE heuristic got us quite good performance around 60s for ResNet18
  // using the below settings.
  // With this heuristic we get around 3.2s with ResNet18 on the RPI5 GPU
  if (auto convOp = dyn_cast<linalg::ConvolutionOpInterface>(rootOp)) {
    return setConvOpForVideoCoreConfig(cast<linalg::LinalgOp>(rootOp), 256, 4);
  }

  return failure();
}

} // namespace mlir::iree_compiler::detail

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
  return setMatmulOpConfig(target, op, workgroupXY, threadMNK);
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

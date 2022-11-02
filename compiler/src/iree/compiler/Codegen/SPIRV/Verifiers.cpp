// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/SPIRV/KernelConfig.h"
#include "iree/compiler/Codegen/SPIRV/Utils.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"

#define DEBUG_TYPE "iree-spirv-verifier"

namespace mlir {
namespace iree_compiler {

constexpr unsigned kWorkgroupTileLevel = 0;
constexpr unsigned kThreadTileLevel = 1;
constexpr unsigned kReductionTileLevel = 2;

LogicalResult verifySPIRVMatmulPromoteVectorizePassPipeline(
    Operation *op, IREE::Codegen::LoweringConfigAttr loweringConfig,
    IREE::Codegen::TranslationInfoAttr translationInfo,
    ArrayRef<int64_t> workgroupSize) {
  // Verify that the translation info is using the right pipeline
  if (translationInfo.getDispatchLoweringPassPipeline() !=
      IREE::Codegen::DispatchLoweringPassPipeline::
          SPIRVMatmulPromoteVectorize) {
    return op->emitOpError("expected pipeline in translation_info to be ")
           << stringifyEnum(IREE::Codegen::DispatchLoweringPassPipeline::
                                SPIRVMatmulPromoteVectorize);
  }

  if (!isa<linalg::MatmulOp, linalg::BatchMatmulOp>(op)) {
    return success();
  }
  LLVM_DEBUG({
    llvm::dbgs() << "verifying op: " << *op << "\n";
    llvm::dbgs() << "chosen workgroup size: [";
    llvm::interleaveComma(workgroupSize, llvm::dbgs());
    llvm::dbgs() << "]\n";
  });

  // Get spirv.target_env attributes
  const spirv::TargetEnvAttr targetEnvAttr = getSPIRVTargetEnvAttr(op);
  const spirv::TargetEnv targetEnv(targetEnvAttr);
  const auto limits = targetEnv.getResourceLimits();
  LLVM_DEBUG(llvm::dbgs() << "target environment: " << targetEnvAttr << "\n");

  const int subgroupSize = limits.getSubgroupSize();
  const int maxSharedMemory = limits.getMaxComputeSharedMemorySize();
  const int maxThreads = limits.getMaxComputeWorkgroupInvocations();
  const auto maxWorkGroupSize = llvm::to_vector<3>(llvm::map_range(
      limits.getMaxComputeWorkgroupSize().getAsValueRange<IntegerAttr>(),
      [](const APInt &dim) { return dim.getSExtValue(); }));

  // Verify each dimension of workgroupSize should be power of two
  if (!llvm::isPowerOf2_64(workgroupSize[0]) ||
      !llvm::isPowerOf2_64(workgroupSize[1]) ||
      !llvm::isPowerOf2_64(workgroupSize[2])) {
    return op->emitOpError(
        "expected each workgroup size dimension to be power of two");
  }

  // Verify each dimension of workgroup size should not exceed the corresponding
  // limit of maxWorkGroupSize
  if (workgroupSize[0] > maxWorkGroupSize[0] ||
      workgroupSize[1] > maxWorkGroupSize[1] ||
      workgroupSize[2] > maxWorkGroupSize[2]) {
    return op->emitOpError("expected workgroup size dimensions not exceeding ")
           << "[" << maxWorkGroupSize[0] << ", " << maxWorkGroupSize[1] << ", "
           << maxWorkGroupSize[2] << "]";
  }

  // Verify the total workgroup size should not exceed maxThreads
  int64_t totalWorkgroupSize =
      workgroupSize[0] * workgroupSize[1] * workgroupSize[2];
  if (totalWorkgroupSize > maxThreads) {
    return op->emitOpError(
               "expected total invocation count in workgroup to be <= ")
           << maxThreads << ", got " << totalWorkgroupSize;
  }

  // Verify the total workgroup size should be multiple of subgroupSize
  if (totalWorkgroupSize % subgroupSize != 0) {
    return op->emitOpError("expected total workgroup size to be multiple of ")
           << subgroupSize;
  }

  Type inputType = op->getOperand(0).getType();
  ArrayRef<int64_t> lhsShape =
      op->getOperand(0).getType().cast<ShapedType>().getShape();
  ArrayRef<int64_t> rhsShape =
      op->getOperand(1).getType().cast<ShapedType>().getShape();

  SmallVector<int64_t> firstLevelTileSizes =
      loweringConfig.getTileSizeVals(kWorkgroupTileLevel);
  SmallVector<int64_t> secondLevelTileSizes =
      loweringConfig.getTileSizeVals(kThreadTileLevel);
  SmallVector<int64_t> thirdLevelTileSizes =
      loweringConfig.getTileSizeVals(kReductionTileLevel);

  if (loweringConfig.getTileSizes().size() != 3) {
    return op->emitOpError("expected 3 levels of tiling sizes, got ")
           << loweringConfig.getTileSizes().size();
  }

  // For BatchMatmul, the first dimension is the batch dimension.
  // We don't check the batch.
  if (isa<linalg::BatchMatmulOp>(op)) {
    lhsShape = lhsShape.drop_front(1);
    rhsShape = rhsShape.drop_front(1);
    firstLevelTileSizes.erase(firstLevelTileSizes.begin());
    secondLevelTileSizes.erase(secondLevelTileSizes.begin());
    thirdLevelTileSizes.erase(thirdLevelTileSizes.begin());
  }

  // Verify the tile size divides the matmul inputs A [M x K] & B [K x N]
  if (lhsShape[0] % firstLevelTileSizes[0] != 0 ||
      lhsShape[1] % thirdLevelTileSizes[2] != 0) {
    return op->emitOpError("LHS shape is indivisible by first level tile size");
  }
  if (rhsShape[0] % thirdLevelTileSizes[2] != 0 ||
      rhsShape[1] % firstLevelTileSizes[1] != 0) {
    return op->emitOpError("RHS shape is indivisible by first level tile size");
  }

  // Verify that workgroup_tile_size = thread_tile_size * workgroup_size
  if (secondLevelTileSizes[0] * workgroupSize[1] != firstLevelTileSizes[0] ||
      secondLevelTileSizes[1] * workgroupSize[0] != firstLevelTileSizes[1]) {
    return op->emitOpError(
        "expected workgroup tile sizes to be the product of thread tile "
        "sizes and workgroup sizes");
  }

  // Verify shared memory usage of operands after tiling <= maxSharedMemory.
  unsigned tilingSharedMemSizeBytes = getTileBytes(
      firstLevelTileSizes[0], firstLevelTileSizes[1], thirdLevelTileSizes[2],
      inputType.cast<ShapedType>().getElementType().getIntOrFloatBitWidth());
  unsigned totalSharedMemSizeBytes = getMultiBufferMemoryUsage(
      tilingSharedMemSizeBytes, translationInfo.getSoftwarePipelineDepth());

  if (totalSharedMemSizeBytes > maxSharedMemory) {
    return op->emitOpError("expected shared memory usage <= ")
           << maxSharedMemory << ", got " << totalSharedMemSizeBytes;
  }
  return success();
}

LogicalResult verifySPIRVBaseVectorizePassPipeline(
    Operation *op, IREE::Codegen::LoweringConfigAttr loweringConfig,
    IREE::Codegen::TranslationInfoAttr translationInfo,
    ArrayRef<int64_t> workgroupSize) {
  // Verify that the translation info is using the right pipeline
  if (translationInfo.getDispatchLoweringPassPipeline() !=
      IREE::Codegen::DispatchLoweringPassPipeline::SPIRVBaseVectorize) {
    return op->emitOpError("expected pipeline in translation_info to be ")
           << stringifyEnum(IREE::Codegen::DispatchLoweringPassPipeline::
                                SPIRVBaseVectorize);
  }

  if (!isa<linalg::Conv2DNhwcHwcfOp, linalg::DepthwiseConv2DNhwcHwcOp>(op)) {
    return success();
  }

  const int numTileSizeLevels = loweringConfig.getTileSizes().size();
  SmallVector<int64_t> firstLevelTileSizes =
      loweringConfig.getTileSizeVals(kWorkgroupTileLevel);
  SmallVector<int64_t> secondLevelTileSizes =
      loweringConfig.getTileSizeVals(kThreadTileLevel);
  SmallVector<int64_t> thirdLevelTileSizes =
      loweringConfig.getTileSizeVals(kReductionTileLevel);

  if (numTileSizeLevels != 4) {
    return op->emitOpError("expected 4 levels of tiling sizes, got ")
           << numTileSizeLevels;
  }

  ArrayRef<int64_t> outputShape =
      op->getOperand(2).getType().cast<ShapedType>().getShape();
  const int64_t oh = outputShape[1], ow = outputShape[2], oc = outputShape[3];

  // Verify the first level tile size divides the Convolution
  // output size [OH, OW, OC]
  if (oh % firstLevelTileSizes[1] != 0 || ow % firstLevelTileSizes[2] != 0 ||
      oc % firstLevelTileSizes[3] != 0) {
    return op->emitOpError(
        "expected first level tile size divides the output size [OH, OW, "
        "OC]");
  }

  // Verify that workgroup_tile_size = thread_tile_size * workgroup_size
  if (secondLevelTileSizes[1] * workgroupSize[2] != firstLevelTileSizes[1] ||
      secondLevelTileSizes[2] * workgroupSize[1] != firstLevelTileSizes[2] ||
      secondLevelTileSizes[3] * workgroupSize[0] != firstLevelTileSizes[3]) {
    return op->emitOpError(
        "expected workgroup tile sizes to be the product of thread tile size "
        "and workgroup size");
  }

  // Verify that the tile sizes for KH and KW should be 1
  if (thirdLevelTileSizes[4] != 1 || thirdLevelTileSizes[5] != 1) {
    return op->emitOpError("expected tile sizes for KH and KW to be 1");
  }

  // Verify the fourth level of tile size
  SmallVector<int64_t> fourthLevelTileSizes = loweringConfig.getTileSizeVals(3);
  if (fourthLevelTileSizes[0] != 0 || fourthLevelTileSizes[1] != 1 ||
      fourthLevelTileSizes[2] != 0 || fourthLevelTileSizes[3] != 0) {
    return op->emitOpError(
        "expected the fourth level of tile size to be [0, 1, 0, 0]");
  }
  return success();
}

}  // namespace iree_compiler
}  // namespace mlir

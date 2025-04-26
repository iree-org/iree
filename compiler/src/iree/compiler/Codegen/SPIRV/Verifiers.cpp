// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/SPIRV/Passes.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/InterleavedRange.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#define DEBUG_TYPE "iree-spirv-verifier"

namespace mlir::iree_compiler {
using CodeGenPipeline = IREE::Codegen::DispatchLoweringPassPipeline;

constexpr unsigned kWorkgroupTileLevel = 0;
constexpr unsigned kThreadTileLevel = 1;
constexpr unsigned kReductionTileLevel = 2;

LogicalResult verifySPIRVMatmulPromoteVectorizePassPipeline(
    Operation *op, IREE::Codegen::LoweringConfigAttr loweringConfig,
    IREE::Codegen::TranslationInfoAttr translationInfo,
    ArrayRef<int64_t> workgroupSize) {
  // Verify that the translation info is using the right pipeline.
  if (translationInfo.getDispatchLoweringPassPipeline() !=
      CodeGenPipeline::SPIRVMatmulPromoteVectorize) {
    return op->emitOpError("expected pipeline in translation_info to be ")
           << stringifyEnum(CodeGenPipeline::SPIRVMatmulPromoteVectorize);
  }

  if (!isa<linalg::MatmulOp, linalg::BatchMatmulOp>(op))
    return success();

  LLVM_DEBUG(llvm::dbgs() << "verifying op: " << *op << "\n"
                          << "chosen workgroup size: "
                          << llvm::interleaved_array(workgroupSize) << "\n");

  FailureOr<int64_t> maybeDepth =
      getSoftwarePipelineDepth(translationInfo.getConfiguration());
  FailureOr<int64_t> maybeStage =
      getSoftwarePipelineStoreStage(translationInfo.getConfiguration());
  if (failed(maybeDepth) || failed(maybeStage)) {
    return op->emitOpError(
        "invalid matmul configuration without pipelining config");
  }

  IREE::GPU::TargetAttr target = getGPUTargetAttr(op);
  LLVM_DEBUG(llvm::dbgs() << "target: " << target << "\n");

  auto funcOp = op->getParentOfType<mlir::FunctionOpInterface>();
  std::optional<int> subgroupSize = getGPUSubgroupSize(funcOp);
  if (!subgroupSize)
    return funcOp->emitError("failed to query subgroup size");
  const int maxThreads = target.getWgp().getMaxThreadCountPerWorkgroup();
  const auto maxWorkGroupSize =
      target.getWgp().getMaxWorkgroupSizes().asArrayRef();

  if (workgroupSize.size() < 3) {
    return funcOp->emitOpError("expected workgroup size to have three "
                               "dimensions for SPIR-V pipelines");
  }

  // Verify each dimension of workgroupSize should be power of two.
  if (!llvm::isPowerOf2_64(workgroupSize[0]) ||
      !llvm::isPowerOf2_64(workgroupSize[1]) ||
      !llvm::isPowerOf2_64(workgroupSize[2])) {
    return op->emitOpError(
        "expected each workgroup size dimension to be power of two");
  }

  // Verify each dimension of workgroup size should not exceed the corresponding
  // limit of maxWorkGroupSize.
  if (workgroupSize[0] > maxWorkGroupSize[0] ||
      workgroupSize[1] > maxWorkGroupSize[1] ||
      workgroupSize[2] > maxWorkGroupSize[2]) {
    return op->emitOpError("expected workgroup size dimensions not exceeding ")
           << "[" << maxWorkGroupSize[0] << ", " << maxWorkGroupSize[1] << ", "
           << maxWorkGroupSize[2] << "]";
  }

  // Verify the total workgroup size should not exceed maxThreads.
  int64_t totalWorkgroupSize =
      workgroupSize[0] * workgroupSize[1] * workgroupSize[2];
  if (totalWorkgroupSize > maxThreads) {
    return op->emitOpError(
               "expected total invocation count in workgroup to be <= ")
           << maxThreads << ", got " << totalWorkgroupSize;
  }

  // Verify the total workgroup size should be multiple of subgroupSize.
  if (totalWorkgroupSize % *subgroupSize != 0) {
    return op->emitOpError("expected total workgroup size to be multiple of ")
           << *subgroupSize;
  }

  ArrayRef<int64_t> lhsShape =
      llvm::cast<ShapedType>(op->getOperand(0).getType()).getShape();
  ArrayRef<int64_t> rhsShape =
      llvm::cast<ShapedType>(op->getOperand(1).getType()).getShape();

  if (loweringConfig.getTilingLevels().size() != 1) {
    return op->emitOpError("expected 1 levels of tiling sizes, got ")
           << loweringConfig.getTilingLevels().size();
  }

  SmallVector<int64_t> tileSizes =
      loweringConfig.getTileSizeVals(kWorkgroupTileLevel);

  // For BatchMatmul, the first dimension is the batch dimension.
  // We don't check the batch.
  if (isa<linalg::BatchMatmulOp>(op)) {
    lhsShape = lhsShape.drop_front(1);
    rhsShape = rhsShape.drop_front(1);
    tileSizes.erase(tileSizes.begin());
  }

  // Verify the tile size divides the matmul inputs A [M x K] & B [K x N].
  const int64_t dimM = lhsShape[0], dimN = rhsShape[1], dimK = lhsShape[1];
  if (dimM % tileSizes[0] != 0 || dimK % tileSizes[2] != 0) {
    return op->emitOpError("LHS shape is indivisible by first level tile size");
  }
  if (dimK % tileSizes[2] != 0 || dimN % tileSizes[1] != 0) {
    return op->emitOpError("RHS shape is indivisible by first level tile size");
  }

  return success();
}

LogicalResult verifySPIRVCooperativeMatrixVectorizePassPipeline(
    Operation *op, IREE::Codegen::LoweringConfigAttr loweringConfig,
    IREE::Codegen::TranslationInfoAttr translationInfo,
    ArrayRef<int64_t> workgroupSize) {
  // Verify that the translation info is using the right pipeline.
  if (translationInfo.getDispatchLoweringPassPipeline() !=
      CodeGenPipeline::SPIRVCooperativeMatrixVectorize) {
    return op->emitOpError("expected pipeline in translation_info to be ")
           << stringifyEnum(CodeGenPipeline::SPIRVCooperativeMatrixVectorize);
  }

  if (!isa<linalg::MatmulOp, linalg::BatchMatmulOp>(op)) {
    return success();
  }
  LLVM_DEBUG(llvm::dbgs() << "verifying op: " << *op << "\n"
                          << "chosen workgroup size: "
                          << llvm::interleaved_array(workgroupSize) << "\n");

  FailureOr<int64_t> maybeDepth =
      getSoftwarePipelineDepth(translationInfo.getConfiguration());
  FailureOr<int64_t> maybeStage =
      getSoftwarePipelineStoreStage(translationInfo.getConfiguration());
  if (failed(maybeDepth) || failed(maybeStage)) {
    return op->emitOpError(
        "invalid cooperative matrix configuration without pipelining config");
  }

  IREE::GPU::TargetAttr target = getGPUTargetAttr(op);
  LLVM_DEBUG(llvm::dbgs() << "target: " << target << "\n");

  auto funcOp = op->getParentOfType<mlir::FunctionOpInterface>();
  std::optional<int> subgroupSize = getGPUSubgroupSize(funcOp);
  if (!subgroupSize)
    return funcOp->emitError("failed to query subgroup size");
  const int maxThreads = target.getWgp().getMaxThreadCountPerWorkgroup();
  const auto maxWorkGroupSize =
      target.getWgp().getMaxWorkgroupSizes().asArrayRef();

  // Verify each dimension of workgroupSize should be power of two.
  if (!llvm::isPowerOf2_64(workgroupSize[0]) ||
      !llvm::isPowerOf2_64(workgroupSize[1]) ||
      !llvm::isPowerOf2_64(workgroupSize[2])) {
    return op->emitOpError(
        "expected each workgroup size dimension to be power of two");
  }

  // Verify each dimension of workgroup size should not exceed the corresponding
  // limit of maxWorkGroupSize.
  if (workgroupSize[0] > maxWorkGroupSize[0] ||
      workgroupSize[1] > maxWorkGroupSize[1] ||
      workgroupSize[2] > maxWorkGroupSize[2]) {
    return op->emitOpError("expected workgroup size dimensions not exceeding ")
           << llvm::interleaved_array(maxWorkGroupSize);
  }

  // Verify the total workgroup size should not exceed maxThreads.
  int64_t totalWorkgroupSize =
      workgroupSize[0] * workgroupSize[1] * workgroupSize[2];
  if (totalWorkgroupSize > maxThreads) {
    return op->emitOpError(
               "expected total invocation count in workgroup to be <= ")
           << maxThreads << ", got " << totalWorkgroupSize;
  }

  // Verify the total workgroup size should be multiple of subgroupSize.
  if (totalWorkgroupSize % *subgroupSize != 0) {
    return op->emitOpError("expected total workgroup size to be multiple of ")
           << *subgroupSize;
  }

  // Verify that there are four level of tile sizes.
  if (loweringConfig.getTilingLevels().size() != 4) {
    return op->emitOpError("expected 4 levels of tiling sizes, got ")
           << loweringConfig.getTilingLevels().size();
  }

  ArrayRef<int64_t> lhsShape =
      llvm::cast<ShapedType>(op->getOperand(0).getType()).getShape();
  ArrayRef<int64_t> rhsShape =
      llvm::cast<ShapedType>(op->getOperand(1).getType()).getShape();

  SmallVector<int64_t> workgroupTileSizes =
      loweringConfig.getTileSizeVals(kWorkgroupTileLevel);
  SmallVector<int64_t> subgroupTileSizes =
      loweringConfig.getTileSizeVals(kThreadTileLevel);
  SmallVector<int64_t> reductionTileSizes =
      loweringConfig.getTileSizeVals(kReductionTileLevel);
  SmallVector<int64_t> nativeVectorSizes = loweringConfig.getTileSizeVals(3);

  // For BatchMatmul, the first dimension is the batch dimension.
  // We don't check the batch.
  if (isa<linalg::BatchMatmulOp>(op)) {
    lhsShape = lhsShape.drop_front(1);
    rhsShape = rhsShape.drop_front(1);
    workgroupTileSizes.erase(workgroupTileSizes.begin());
    subgroupTileSizes.erase(subgroupTileSizes.begin());
    reductionTileSizes.erase(reductionTileSizes.begin());
    nativeVectorSizes.erase(nativeVectorSizes.begin());
  }

  auto getElementType = [](Value v) {
    return llvm::cast<ShapedType>(v.getType()).getElementType();
  };

  Type lhsType = getElementType(op->getOperand(0));
  Type rhsType = getElementType(op->getOperand(1));
  Type resultType = getElementType(op->getOperand(2));

  // Verify that the fourth level tile sizes match cooperative matrix,
  // and subgroup tile sizes should be multiple of cooperative matrix (M, N, K)
  // sizes.
  bool isNativeVectorSizeAccepted = false;
  for (IREE::GPU::MMAAttr mma : target.getWgp().getMma()) {
    auto [mSize, nSize, kSize] = mma.getMNKShape();
    auto [aType, bType, cType] = mma.getABCElementTypes();

    if (aType == lhsType && bType == rhsType && cType == resultType &&
        mSize == nativeVectorSizes[0] && nSize == nativeVectorSizes[1] &&
        kSize == nativeVectorSizes[2]) {
      isNativeVectorSizeAccepted = true;
      if (subgroupTileSizes[0] % mSize != 0 ||
          subgroupTileSizes[1] % nSize != 0 ||
          reductionTileSizes[2] % kSize != 0) {
        return op->emitOpError(
                   "expected subgroup tile sizes to be multiple of ")
               << "[" << mSize << ", " << nSize << ", " << kSize << "]";
      }
    }
  }

  if (!isNativeVectorSizeAccepted) {
    return op->emitOpError(
        "expected the fourth level tile sizes to match cooperative matrix "
        "sizes");
  }

  // Verify the tile size divides the matmul inputs A [M x K] & B [K x N].
  const int64_t dimM = lhsShape[0], dimN = rhsShape[1], dimK = lhsShape[1];
  if (dimM % workgroupTileSizes[0] != 0 || dimK % reductionTileSizes[2] != 0) {
    return op->emitOpError("LHS shape is indivisible by first level tile size");
  }
  if (dimK % reductionTileSizes[2] != 0 || dimN % workgroupTileSizes[1] != 0) {
    return op->emitOpError("RHS shape is indivisible by first level tile size");
  }

  // Verify workgroup_size_x = warp_size * wg_tile_n / subgroup_tile_n.
  if (workgroupSize[0] * subgroupTileSizes[1] !=
      *subgroupSize * workgroupTileSizes[1]) {
    return op->emitOpError(
        "expected workgroup x component equals to (warp_size * wg_tile_n / "
        "subgroup_tile_n)");
  }

  // Verify workgroup_size_y = wg_tile_m / subgroup_tile_m.
  if (workgroupSize[1] * subgroupTileSizes[0] != workgroupTileSizes[0]) {
    return op->emitOpError(
        "expected workgroup y component equals to (wg_tile_m / "
        "subgroup_tile_m)");
  }

  return success();
}

LogicalResult verifySPIRVBaseVectorizePassPipeline(
    Operation *op, IREE::Codegen::LoweringConfigAttr loweringConfig,
    IREE::Codegen::TranslationInfoAttr translationInfo,
    ArrayRef<int64_t> workgroupSize) {
  // Verify that the translation info is using the right pipeline.
  if (translationInfo.getDispatchLoweringPassPipeline() !=
      CodeGenPipeline::SPIRVBaseVectorize) {
    return op->emitOpError("expected pipeline in translation_info to be ")
           << stringifyEnum(CodeGenPipeline::SPIRVBaseVectorize);
  }

  if (!isa<linalg::Conv2DNhwcHwcfOp, linalg::DepthwiseConv2DNhwcHwcOp>(op)) {
    return success();
  }

  const int numTileSizeLevels = loweringConfig.getTilingLevels().size();
  SmallVector<int64_t> workgroupTileSizes =
      loweringConfig.getTileSizeVals(kWorkgroupTileLevel);
  SmallVector<int64_t> threadTileSizes =
      loweringConfig.getTileSizeVals(kThreadTileLevel);
  SmallVector<int64_t> reductionTileSizes =
      loweringConfig.getTileSizeVals(kReductionTileLevel);

  if (numTileSizeLevels != 4) {
    return op->emitOpError("expected 4 levels of tiling sizes, got ")
           << numTileSizeLevels;
  }

  ArrayRef<int64_t> outputShape =
      llvm::cast<ShapedType>(op->getOperand(2).getType()).getShape();
  const int64_t oh = outputShape[1], ow = outputShape[2], oc = outputShape[3];

  // Verify the first level tile size divides the Convolution
  // output size [OH, OW, OC].
  if (oh % workgroupTileSizes[1] != 0 || ow % workgroupTileSizes[2] != 0 ||
      oc % workgroupTileSizes[3] != 0) {
    return op->emitOpError(
        "expected first level tile size divides the output size [OH, OW, "
        "OC]");
  }

  // Verify that workgroup_tile_size = thread_tile_size * workgroup_size.
  if (threadTileSizes[1] * workgroupSize[2] != workgroupTileSizes[1] ||
      threadTileSizes[2] * workgroupSize[1] != workgroupTileSizes[2] ||
      threadTileSizes[3] * workgroupSize[0] != workgroupTileSizes[3]) {
    return op->emitOpError(
        "expected workgroup tile sizes to be the product of thread tile size "
        "and workgroup size");
  }

  // Verify that the tile sizes for KH and KW should be 1.
  if (reductionTileSizes[4] != 1 || reductionTileSizes[5] != 1) {
    return op->emitOpError("expected tile sizes for KH and KW to be 1");
  }

  // Verify the fourth level of tile size.
  SmallVector<int64_t> fourthLevelTileSizes = loweringConfig.getTileSizeVals(3);
  if (fourthLevelTileSizes[0] != 0 || fourthLevelTileSizes[1] != 1 ||
      fourthLevelTileSizes[2] != 0 || fourthLevelTileSizes[3] != 0) {
    return op->emitOpError(
        "expected the fourth level of tile size to be [0, 1, 0, 0]");
  }
  return success();
}

} // namespace mlir::iree_compiler

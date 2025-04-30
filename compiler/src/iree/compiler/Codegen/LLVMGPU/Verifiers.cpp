// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Interfaces/PartitionableLoopsInterface.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"

namespace mlir::iree_compiler {

using CodeGenPipeline = IREE::Codegen::DispatchLoweringPassPipeline;

////////////////////////////////////////////////////////////////////////////////
// Constants used in the matmul lowering verifiers.
constexpr unsigned kWorkgroupTileLevel = 0;

// Use the constexpr to convey the meaning of the indices.
// Dimenstions identifiers for: workgroup size (x, y, z), and thread (x, y, z).
constexpr int kDimX = 0;
constexpr int kDimY = 1;
constexpr int kDimZ = 2;

// Dimensions identifiers for: matmul problem shapes (m, n, k), thread block
// shape (m, n, k), warp shape, and instruction shape (m, n, k).
constexpr int kM = 0;
constexpr int kN = 1;
constexpr int kK = 2;
////////////////////////////////////////////////////////////////////////////////

/// Returns the shape of the math instruction for the given pipeline and input
/// element type.
static LogicalResult
getInstructionShape(Operation *op, CodeGenPipeline pipeline,
                    Type inputElementType,
                    SmallVector<int64_t> &instructionShape) {
  switch (pipeline) {
  case CodeGenPipeline::LLVMGPUMatmulTensorCore:
    // Tensor Core Pipeline / WMMA API
    if (inputElementType.isF16() || inputElementType.isBF16()) {
      instructionShape = {16, 16, 16};
    } else if (inputElementType.isF32()) {
      instructionShape = {16, 16, 8};
    } else {
      return op->emitError(
          "Expected f16, bf16 or f32 for Tensor Core (WMMA) pipeline");
    }
    break;
  case CodeGenPipeline::LLVMGPUMatmulTensorCoreMmaSync:
    // Tensor Core Pipeline / MMA.SYNC
    if (inputElementType.isF16() || inputElementType.isBF16()) {
      instructionShape = {16, 8, 16};
    } else if (inputElementType.isF32()) {
      instructionShape = {16, 8, 8};
    } else {
      return op->emitError(
          "Expected f16, bf16 or f32 for Tensor Core (MMA.SYNC) pipeline");
    }
    break;
  default:
    return op->emitError(
        "Expected matmul SIMT, TensorCore(WMMA), or TensorCore(MMA.SYNC), "
        "compilation pipeline");
  }
  return success();
}

/// Verifies launch configuration for matmul and batchmatmul on a GPU for CUDA
/// and Tensor Core pipelines.
LogicalResult
verifyGPUMatmulPipeline(Operation *op,
                        IREE::Codegen::LoweringConfigAttr loweringConfig,
                        IREE::Codegen::TranslationInfoAttr translationInfo,
                        ArrayRef<int64_t> workgroupSize) {
  CodeGenPipeline pipeline = translationInfo.getDispatchLoweringPassPipeline();

  if (pipeline != CodeGenPipeline::LLVMGPUMatmulTensorCore &&
      pipeline != CodeGenPipeline::LLVMGPUMatmulTensorCoreMmaSync) {
    return success();
  }

  // Only verify batched and unbatched matmul.
  if (!isa<linalg::MatmulOp, linalg::BatchMatmulOp>(op)) {
    return success();
  }

  // Early exit if the workgroup size is not set.
  if (workgroupSize.empty()) {
    return op->emitOpError("expected workgroup size for GPU pipelines");
  }

  FailureOr<int64_t> maybeDepth =
      getSoftwarePipelineDepth(translationInfo.getConfiguration());
  FailureOr<int64_t> maybeStage =
      getSoftwarePipelineStoreStage(translationInfo.getConfiguration());
  if (failed(maybeDepth) || failed(maybeStage)) {
    return op->emitOpError(
        "invalid matmul configuration without pipelining config");
  }

  if (*maybeStage != 1) {
    return op->emitError(
        "store to workgroup memory currently expected to happen in stage 1 of "
        "software pipeline.");
  }

  // Get compilation pipeline.
  StringRef pipelineName = stringifyEnum(pipeline);

  // Get Operand/Result types.
  mlir::Type lhsType = op->getOperand(0).getType();
  mlir::Type rhsType = op->getOperand(1).getType();
  assert(cast<ShapedType>(lhsType).getElementType() ==
             cast<ShapedType>(rhsType).getElementType() &&
         "expected lhs and rhs to have same type. Mixed input types are not "
         "supported yet in IREE Codegen.");

  // Get lhs and rhs shapes.
  ArrayRef<int64_t> lhsShape = llvm::cast<ShapedType>(lhsType).getShape();
  ArrayRef<int64_t> rhsShape = llvm::cast<ShapedType>(rhsType).getShape();

  // Tile shapes in number of elements.
  SmallVector<int64_t> tileShape =
      loweringConfig.getTileSizeVals(kWorkgroupTileLevel);
  SmallVector<int64_t> threadBlockShape{tileShape};

  if (auto batchMatmulOp = dyn_cast<linalg::BatchMatmulOp>(op)) {
    // Inspect the batch tile dimensions separately for batch. The batch tile
    // dim should be strictly greater than 1 for parallelizable loops and 0
    // for non-parallelizable.
    if (cast<PartitionableLoopsInterface>(op).getPartitionableLoops(
            kNumMaxParallelDims)[0] == 0) {
      if (tileShape[0] > 1) {
        return op->emitError("Received batch tile dimension of ")
               << tileShape[0]
               << " instead of 1 or lower for partitionable loops with "
               << "compilation pipeline " << pipelineName;
      }
    } else {
      if (tileShape[0] != 0) {
        return op->emitError("Received batch tile dimension of ")
               << tileShape[0]
               << " instead of 0 for non-partitionable loops with compilation"
               << " pipeline " << pipelineName;
      }
    }

    // Remove the batch dimension from the threadBlockShape, lhsShape, and
    // rhsShape.
    threadBlockShape = {tileShape[1], tileShape[2], tileShape[3]};
    lhsShape = lhsShape.drop_front();
    rhsShape = rhsShape.drop_front();
  }

  //
  // Begin verification for CUDA and Tensor Core pipelines.
  //

  // Verify the total number of threads in a thread block.
  int totalNumThreads = workgroupSize[0] * workgroupSize[1] * workgroupSize[2];

  if (totalNumThreads > 1024) {
    return op->emitError("Total number of threads in a thread block ")
           << totalNumThreads
           << " exceeds the limit of 1024 with compilation pipeline "
           << pipelineName;
  }

  // Verify the number of threads in z-dim is 1.
  if (workgroupSize[kDimZ] != 1) {
    return op->emitError("Expected workgroup size in z-dim = 1, but got ")
           << workgroupSize[kDimZ] << " with compilation pipeline "
           << pipelineName;
  }

  //
  // Additional verification Tensor Core pipelines.
  //

  // Verify that x-dim has multiple of kWarpSize threads or has integer units of
  // warps in x-dim.
  if (workgroupSize[kDimX] % kWarpSize != 0) {
    return op->emitError("Number of threads in x-dim ")
           << workgroupSize[kDimX] << " is not a multiple of warp size ("
           << kWarpSize
           << ") or integer units of warps in x-dim with compilation pipeline "
           << pipelineName;
  }

  // Number of warps in x, y, and z dim.
  SmallVector<int64_t> numWarps{workgroupSize[kDimX] / kWarpSize,
                                workgroupSize[kDimY], workgroupSize[kDimZ]};

  // Matrix-multiply problem shape in number of elements in M, N, and K dim.
  SmallVector<int64_t> matmulShape{lhsShape[0], rhsShape[1], lhsShape[1]};

  // Warp tile shape in number of elements in M, N, and K dim.
  // Note that num warp in (x, y, z) dim are mapped to problem (M, N, K) dim as:
  // DimY -> ProblemDimM, DimX -> ProblemDimN, DimZ -> ProblemDimK.
  SmallVector<int64_t> warpShape{threadBlockShape[kM] / numWarps[kDimY],
                                 threadBlockShape[kN] / numWarps[kDimX],
                                 threadBlockShape[kK] / numWarps[kDimZ]};

  // Instruction shape in number of elements in M, N, and K dim.
  SmallVector<int64_t> instructionShape;
  if (failed(getInstructionShape(
          op, pipeline, llvm::cast<ShapedType>(lhsType).getElementType(),
          instructionShape))) {
    return failure();
  }

  // Verify that matmul problem shape can be tiled with the thread block shape.
  // TODO: This check should be relaxed as we allow unaligned matmul shapes.
  if (matmulShape[kM] % threadBlockShape[kM] != 0 ||
      matmulShape[kN] % threadBlockShape[kN] != 0 ||
      matmulShape[kK] % threadBlockShape[kK] != 0) {
    return op->emitError("Thread block shape ")
           << threadBlockShape << " cannot be tiled on matmul shape "
           << matmulShape << " with compilation pipeline " << pipelineName;
  }

  // Verify that if warp shape can be tiled using warp-level Tensor core
  // instruction shape.
  if (warpShape[kM] % instructionShape[kM] != 0 ||
      warpShape[kN] % instructionShape[kN] != 0 ||
      warpShape[kK] % instructionShape[kK] != 0) {
    return op->emitError("Tensor Core instruction shape ")
           << instructionShape << " cannot be tiled on warp shape " << warpShape
           << " with compilation pipeline " << pipelineName;
  }

  return success();
}

/// Verifies pipelines that use iree_gpu.lowering_config attributes.
LogicalResult
verifyGPUMatmulPipeline(Operation *op,
                        IREE::GPU::LoweringConfigAttr loweringConfig,
                        IREE::Codegen::TranslationInfoAttr translationInfo) {

  CodeGenPipeline pipeline = translationInfo.getDispatchLoweringPassPipeline();
  // TODO: add verification for other pipelines
  if (pipeline != CodeGenPipeline::LLVMGPUVectorDistribute) {
    return success();
  }

  // Only verify batched and unbatched matmul.
  if (!isa<linalg::MatmulOp, linalg::BatchMatmulOp>(op)) {
    return success();
  }

  unsigned reduction = static_cast<uint32_t>(IREE::GPU::TilingLevel::Reduction);
  unsigned numLoops = llvm::cast<linalg::LinalgOp>(op).getNumLoops();
  size_t size = 0;

  SmallVector<int64_t> reductionTileSizes =
      loweringConfig.getStaticTilingLevelSizes(reduction, op);

  size = reductionTileSizes.size();

  if (size > numLoops) {
    return op->emitOpError("expected number of reduction tile size is equal "
                           "or less than number of loops");
  }
  for (size_t i = 0; i < size; ++i) {
    if (reductionTileSizes[i] > 0 &&
        llvm::cast<linalg::LinalgOp>(op).getIteratorTypesArray()[i] !=
            utils::IteratorType::reduction) {
      return op->emitOpError(
          "expected to non-zero reduction tile has reduction iterator");
    }
  }

  SmallVector<int64_t> workgroupTileSizes =
      loweringConfig.getWorkgroupTileSizes();
  size = workgroupTileSizes.size();
  for (size_t i = 0; i < size; ++i) {
    if (workgroupTileSizes[i] > 0 &&
        llvm::cast<linalg::LinalgOp>(op).getIteratorTypesArray()[i] !=
            utils::IteratorType::parallel) {
      return op->emitOpError(
          "expected to non-zero workgroup tile has parallel iterator");
    }
  }

  return success();
}

} // namespace mlir::iree_compiler

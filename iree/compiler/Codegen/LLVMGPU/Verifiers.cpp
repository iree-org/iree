// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/LLVMGPUUtils.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"

namespace mlir {
namespace iree_compiler {

constexpr unsigned kWorkgroupTileLevel = 0;
constexpr int kSharedMemSizeBytes = 64 * 1024;

LogicalResult verifyGPUMatmulSimtPassPipeline(
    Operation *op, IREE::Codegen::LoweringConfigAttr loweringConfig,
    IREE::Codegen::TranslationInfoAttr translationInfo,
    ArrayRef<int64_t> workgroupSize) {
  auto pipeline =
      IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUMatmulSimt;
  StringRef pipelineName = stringifyEnum(pipeline);
  if (workgroupSize.empty()) {
    return op->emitOpError("expected workgroup size for GPU pipelines");
  }

  // Verify the total workgroup size is <= 1024
  int64_t totalWorkgroupSize =
      workgroupSize[0] * workgroupSize[1] * workgroupSize[2];
  if (totalWorkgroupSize > 1024) {
    return op->emitOpError("expected workgroup size to be <=1024 for ")
           << pipelineName << ", got " << totalWorkgroupSize;
  }

  // Verify the workgroup.z component should always be 1
  if (workgroupSize[2] != 1) {
    return op->emitOpError("expected workgroup z component to be 1 for ")
           << pipelineName << ", got " << workgroupSize[2];
  }

  // For linalg::MatmulOp, verify shared memory usage of operands after tiling
  // requires <= 64Kb combined space.
  linalg::MatmulOp matmulOp = dyn_cast<linalg::MatmulOp>(op);
  if (matmulOp) {
    SmallVector<int64_t> firstLevelTileSizes =
        loweringConfig.getTileSizeVals(kWorkgroupTileLevel);
    MemRefType type =
        matmulOp.getInputOperand(0)->get().getType().cast<mlir::MemRefType>();
    unsigned bytesSize = type.getElementType().getIntOrFloatBitWidth() / 8;

    // Input shape sizes: A [ M x K],  B [ K x N]
    unsigned totalSharedMemSizeBytes =
        (firstLevelTileSizes[0] * firstLevelTileSizes[2] +
         firstLevelTileSizes[1] * firstLevelTileSizes[2]) *
        bytesSize;

    if (totalSharedMemSizeBytes > kSharedMemSizeBytes) {
      return op->emitOpError("expected shared memory usage <= 64Kb for ")
             << pipelineName << ", got " << totalSharedMemSizeBytes;
    }
  }
  return success();
}

LogicalResult verifyGPUMatmulTensorCorePipeline(
    Operation *op, IREE::Codegen::LoweringConfigAttr loweringConfig,
    IREE::Codegen::TranslationInfoAttr translationInfo,
    ArrayRef<int64_t> workgroupSize) {
  auto pipeline =
      IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUMatmulTensorCore;
  StringRef pipelineName = stringifyEnum(pipeline);
  if (workgroupSize.empty()) {
    return op->emitOpError("expected workgroup size for GPU pipelines");
  }

  // Verify the total workgroup size is <= 1024
  int64_t totalWorkgroupSize =
      workgroupSize[0] * workgroupSize[1] * workgroupSize[2];
  if (totalWorkgroupSize > 1024) {
    return op->emitOpError("expected workgroup size to be <=1024 for ")
           << pipelineName << ", got " << totalWorkgroupSize;
  }

  // Verify that the workgroup X dimension is 32 aligned
  if (workgroupSize[0] % 32 != 0) {
    return op->emitOpError("workgroup size is not 32 aligned for ")
           << pipelineName << ", got " << workgroupSize[0];
  }

  // Verify the workgroup.z component should always be 1
  if (workgroupSize[2] != 1) {
    return op->emitOpError("expected workgroup z component to be 1 for ")
           << pipelineName << ", got " << workgroupSize[2];
  }

  // Verify the second level of tiling = first level tile size divided by the
  // warps per workgroup size
  SmallVector<int64_t> firstLevelTileSizes =
      loweringConfig.getTileSizeVals(kWorkgroupTileLevel);
  std::array<int64_t, 3> warpsPerWorkgroup = {
      workgroupSize[0] / kWarpSize, workgroupSize[1], workgroupSize[2]};
  SmallVector<int64_t, 3> secondLevelTileSizes;
  for (int i = 0; i < firstLevelTileSizes.size(); ++i) {
    secondLevelTileSizes.push_back(firstLevelTileSizes[i] /
                                   warpsPerWorkgroup[i]);
  }

  // Verify the TensorCore size divides the second level tile size
  SmallVector<int64_t, 3> tensorCoreSize({16, 16, 8});
  if (secondLevelTileSizes[0] % tensorCoreSize[0] != 0 ||
      secondLevelTileSizes[1] % tensorCoreSize[1] != 0 ||
      secondLevelTileSizes[2] % tensorCoreSize[2] != 0) {
    return op->emitOpError(
               "tensorcore size doesn't factor into second level tile size "
               "for ")
           << pipelineName;
  }

  // For linalg::MatmulOp, verify the first level tile size divides the matmul
  // inputs A [M x K] & B [K x N]
  linalg::MatmulOp matmulOp = dyn_cast<linalg::MatmulOp>(op);
  if (matmulOp) {
    ArrayRef<int64_t> lhsShape =
        getUntiledShape(matmulOp.getInputOperand(0)->get());
    ArrayRef<int64_t> rhsShape =
        getUntiledShape(matmulOp.getInputOperand(1)->get());

    if (lhsShape[0] % firstLevelTileSizes[0] != 0 ||
        lhsShape[1] % firstLevelTileSizes[2] != 0) {
      return op->emitOpError(
                 "lhsShape doesn't factor into first level tile size for ")
             << pipelineName;
    }

    if (rhsShape[0] % firstLevelTileSizes[2] != 0 ||
        rhsShape[1] % firstLevelTileSizes[1] != 0) {
      return op->emitOpError(
                 "rhsShape doesn't factor into first level tile size for ")
             << pipelineName;
    }

    unsigned bytesSize;
    if (MemRefType type = matmulOp.getInputOperand(0)
                              ->get()
                              .getType()
                              .dyn_cast<mlir::MemRefType>()) {
      bytesSize = type.getElementType().getIntOrFloatBitWidth() / 8;
    } else if (RankedTensorType type = matmulOp.getInputOperand(0)
                                           ->get()
                                           .getType()
                                           .dyn_cast<RankedTensorType>()) {
      bytesSize = type.getElementType().getIntOrFloatBitWidth() / 8;
    } else if (UnrankedTensorType type = matmulOp.getInputOperand(0)
                                             ->get()
                                             .getType()
                                             .dyn_cast<UnrankedTensorType>()) {
      bytesSize = type.getElementType().getIntOrFloatBitWidth() / 8;
    } else {
      // Unable to determine type, skip rest of verification.
      return success();
    }

    // Input shape sizes: A [ M x K],  B [ K x N]
    unsigned totalSharedMemSizeBytes =
        (firstLevelTileSizes[0] * firstLevelTileSizes[2] +
         firstLevelTileSizes[1] * firstLevelTileSizes[2]) *
        bytesSize;

    if (totalSharedMemSizeBytes > kSharedMemSizeBytes) {
      return op->emitOpError("expected shared memory usage <= 64Kb for ")
             << pipelineName << ", got " << totalSharedMemSizeBytes;
    }
  }
  return success();
}

}  // namespace iree_compiler
}  // namespace mlir

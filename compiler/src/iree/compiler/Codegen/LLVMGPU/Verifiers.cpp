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

/// Verifies pipelines that use iree_gpu.lowering_config attributes.
LogicalResult verifyLLVMGPUVectorDistributePipeline(
    Operation *op, IREE::GPU::LoweringConfigAttr loweringConfig) {
  // Only verify batched and unbatched matmul.
  if (!isa<linalg::MatmulOp, linalg::BatchMatmulOp>(op)) {
    return success();
  }

  unsigned reduction = static_cast<uint32_t>(IREE::GPU::TilingLevel::Reduction);
  unsigned numLoops = cast<linalg::LinalgOp>(op).getNumLoops();
  size_t size = 0;

  SmallVector<int64_t> reductionTileSizes =
      loweringConfig.getStaticTilingLevelSizes(reduction, op);

  size = reductionTileSizes.size();

  if (size > numLoops) {
    return op->emitOpError("expected no more than ")
           << numLoops << " tile sizes in the reduction tiling level, but "
           << size << " were set";
  }
  for (size_t i = 0; i < size; ++i) {
    if (reductionTileSizes[i] > 0 &&
        cast<linalg::LinalgOp>(op).getIteratorTypesArray()[i] !=
            utils::IteratorType::reduction) {
      return op->emitOpError(
                 "expected only reduction dims to be set in the reduction "
                 "tiling level, but tile size at index (")
             << i << ") was also set";
    }
  }

  SmallVector<int64_t> workgroupTileSizes =
      loweringConfig.getWorkgroupTileSizes();
  size = workgroupTileSizes.size();

  if (size > numLoops) {
    return op->emitOpError("expected no more than ")
           << numLoops << " tile sizes in the workgroup tiling level, but "
           << size << " were set";
  }
  for (size_t i = 0; i < size; ++i) {
    if (workgroupTileSizes[i] > 0 &&
        cast<linalg::LinalgOp>(op).getIteratorTypesArray()[i] !=
            utils::IteratorType::parallel) {
      return op->emitOpError(
                 "expected only parallel dims to be set in the workgroup "
                 "tiling level, but tile size at index (")
             << i << ") was also set";
    }
  }

  return success();
}

} // namespace mlir::iree_compiler

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
    return op->emitOpError("expected number of reduction tile size is equal "
                           "or less than number of loops");
  }
  for (size_t i = 0; i < size; ++i) {
    if (reductionTileSizes[i] > 0 &&
        cast<linalg::LinalgOp>(op).getIteratorTypesArray()[i] !=
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
        cast<linalg::LinalgOp>(op).getIteratorTypesArray()[i] !=
            utils::IteratorType::parallel) {
      return op->emitOpError(
          "expected to non-zero workgroup tile has parallel iterator");
    }
  }

  return success();
}

} // namespace mlir::iree_compiler

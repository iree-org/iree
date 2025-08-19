// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- NVIDIAConfig.h - NVIDIA CodeGen Configurations ---------------------===//
//
// This file contains CodeGen configurations for NVIDIA GPUs.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/SPIRV/KernelConfig.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/ADT/APInt.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

#define DEBUG_TYPE "iree-spirv-nvidia-config"

using llvm::APIntOps::GreatestCommonDivisor;

constexpr unsigned NVIDIANumSubgroupsPerWorkgroup = 4;
// The number of tiles along M and N dimensions per workgroup.
constexpr unsigned NVIDIANumMNTilesPerSubgroup = 4;

namespace mlir::iree_compiler::detail {

static LogicalResult setNVIDIAMatmulConfig(linalg::LinalgOp op,
                                           IREE::GPU::TargetAttr target) {
  // First try to see if we can use tensor cores.
  /*
  if (succeeded(setCooperativeMatrixConfig(target, op,
                                           NVIDIANumSubgroupsPerWorkgroup,
                                           NVIDIANumMNTilesPerSubgroup)))
    return success();
                                            */
  const int subgroupSize = target.getPreferredSubgroupSize();
  const std::array<int64_t, 2> workgroupXY = {subgroupSize, 8};
  std::array<int64_t, 3> threadMNK;
  auto inputType =
      llvm::cast<ShapedType>(op.getDpsInputOperand(0)->get().getType());
  if (IREE::Util::getTypeBitWidth(inputType.getElementType()) == 16) {
    threadMNK = {8, 8, 32};
  } else {
    threadMNK = {4, 4, 32};
  }
  return setMatmulOpConfig(target, op, workgroupXY, threadMNK,
                           /*enablePromotion=*/false);
}

// Volta architecture:
// https://docs.nvidia.com/cuda/volta-tuning-guide/index.html#sm-occupancy
//
// * 64K 32-bit registers per SM
// * 96KB shared memory per SM
// * Max 32 thread blocks per SM
// * Max 64 concurrent warps per SM
// * Max 255 registers per thread

// Turing architecture:
// https://docs.nvidia.com/cuda/turing-tuning-guide/index.html#sm-occupancy
//
// * 64K 32-bit registers per SM
// * 64KB shared memory per SM
// * Max 16 thread blocks per SM
// * Max 32 concurrent warps per SM
// * Max 255 registers per thread

// Ampere architecture:
// https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html#sm-occupancy
//
// * 64K 32-bit registers per SM
// * 164KB/96KB shared memory for compute capability 8.0/8.6
// * Max 32/16 thread blocks per SM for compute capability 8.0/8.6
// * Max 64 concurrent warps per SM
// * Max 255 registers per thread

// Note that the above numbers are from CUDA docs; for Vulkan the drivder can
// expose slightly different numbers, e.g., max shared memory size is smaller.

LogicalResult setNVIDIACodeGenConfig(IREE::GPU::TargetAttr target,
                                     Operation *rootOp) {
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(rootOp)) {
    if (isMatmulOrBatchMatmul(linalgOp))
      return setNVIDIAMatmulConfig(linalgOp, target);
  }

  return failure();
}

} // namespace mlir::iree_compiler::detail

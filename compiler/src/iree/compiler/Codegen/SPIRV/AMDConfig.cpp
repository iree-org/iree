// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- AMDConfig.h - AMD CodeGen Configurations ---------------------------===//
//
// This file contains CodeGen configurations for AMD GPUs.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/Dialect/LoweringConfig.h"
#include "iree/compiler/Codegen/SPIRV/KernelConfig.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"

#define DEBUG_TYPE "iree-spirv-amd-config"

namespace mlir {
namespace iree_compiler {
namespace detail {

constexpr unsigned AMDSimtSoftwarePipelineDepth = 2;
constexpr unsigned AMDSimtSoftwarePipelineStoreStage = 0;

constexpr unsigned AMDCoopMatrixSoftwarePipelineDepth = 1;
constexpr unsigned AMDCoopMatrixSoftwarePipelineStoreStage = 0;

constexpr unsigned AMDNumSubgroupsPerWorkgroup = 4;
// The number of tiles along M and N dimensions per workgroup.
constexpr unsigned AMDNumMNTilesPerSubgroup = 8;

static LogicalResult setAMDMatmulConfig(linalg::LinalgOp op,
                                        const spirv::TargetEnv &targetEnv) {
  if (succeeded(setCooperativeMatrixConfig(
          targetEnv, op, AMDNumSubgroupsPerWorkgroup, AMDNumMNTilesPerSubgroup,
          AMDCoopMatrixSoftwarePipelineDepth,
          AMDCoopMatrixSoftwarePipelineStoreStage)))
    return success();

  spirv::ResourceLimitsAttr limits = targetEnv.getResourceLimits();
  const int subgroupSize = limits.getSubgroupSize();
  const std::array<int64_t, 2> workgroupXY = {subgroupSize / 2, 8};
  std::array<int64_t, 3> threadMNK;
  auto inputType = op.getDpsInputOperand(0)->get().getType().cast<ShapedType>();
  if (inputType.getElementType().getIntOrFloatBitWidth() == 16) {
    threadMNK = {8, 8, 32};
  } else {
    threadMNK = {8, 4, 16};
  }
  return setMatmulOpConfig(limits, op, workgroupXY, threadMNK,
                           /*enablePromotion=*/true,
                           AMDSimtSoftwarePipelineDepth,
                           AMDSimtSoftwarePipelineStoreStage);
}

// RDNA architecture:
// https://gpuopen.com/wp-content/uploads/2019/08/RDNA_Architecture_public.pdf
//
// Workgroup Processor (WGP) is the block for workgroups in RDNA; it has its own
// instruction/constant cache, L0 cache x2, Local Data Share (LDS, a.k.a. shared
// memory), SALU x4, SIMD32 x4.
//
// * 1024 registers per SIMD32
// * 128KB LDS per WGP
// * Max 20 waves per SIMD32
// * Max 64KB LDS per workgroup

LogicalResult setAMDCodeGenConfig(const spirv::TargetEnv &targetEnv,
                                  Operation *rootOp) {
  spirv::ResourceLimitsAttr limits = targetEnv.getResourceLimits();
  int subgroupSize = limits.getSubgroupSize();

  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(rootOp)) {
    if (isMatmulOrBatchMatmul(linalgOp))
      return setAMDMatmulConfig(linalgOp, targetEnv);
  }

  if (auto convOp = dyn_cast<linalg::ConvolutionOpInterface>(rootOp)) {
    bool hasPaddedInput = convOp.image().getDefiningOp<tensor::PadOp>();
    int bestTilingFactor = hasPaddedInput ? 16 : 32;
    return setConvOpConfig(rootOp, subgroupSize, bestTilingFactor);
  }

  return failure();
}

}  // namespace detail
}  // namespace iree_compiler
}  // namespace mlir

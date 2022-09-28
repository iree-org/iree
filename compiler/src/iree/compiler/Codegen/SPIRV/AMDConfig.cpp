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
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"

#define DEBUG_TYPE "iree-spirv-amd-config"

namespace mlir {
namespace iree_compiler {
namespace detail {

static LogicalResult setAMDMatmulConfig(linalg::LinalgOp op, int subgroupSize) {
  const std::array<int64_t, 2> workgroupXY = {subgroupSize / 2, 8};
  const std::array<int64_t, 3> threadMNK = {8, 4, 16};
  return setMatmulOpConfig(op, subgroupSize, workgroupXY, threadMNK,
                           /*useWorkgroupMemory=*/true);
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
  int subgroupSize = targetEnv.getResourceLimits().getSubgroupSize();

  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(rootOp)) {
    if (isMatmulOrBatchMatmul(linalgOp))
      return setAMDMatmulConfig(linalgOp, subgroupSize);
  }

  return TypeSwitch<Operation *, LogicalResult>(rootOp)
      .Case<linalg::BatchMatmulOp, linalg::MatmulOp>([subgroupSize](auto op) {
        return setAMDMatmulConfig(op, subgroupSize);
      })
      .Case<linalg::Conv2DNchwFchwOp, linalg::Conv2DNhwcHwcfOp>(
          [subgroupSize](auto op) {
            bool hasPaddedInput =
                op.image().template getDefiningOp<tensor::PadOp>();
            int bestTilingFactor = hasPaddedInput ? 16 : 32;
            return setConvOpConfig(op, subgroupSize, bestTilingFactor);
          })
      .Case<linalg::DepthwiseConv2DNhwcHwcOp>([subgroupSize](auto op) {
        bool hasPaddedInput =
            op.image().template getDefiningOp<tensor::PadOp>();
        int bestTilingFactor = hasPaddedInput ? 16 : 32;
        return setConvOpConfig(op, subgroupSize, bestTilingFactor);
      })
      .Default([](Operation *) { return success(); });
}

}  // namespace detail
}  // namespace iree_compiler
}  // namespace mlir

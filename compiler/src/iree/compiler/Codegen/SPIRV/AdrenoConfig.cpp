// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- AdrenoConfig.h - Adreno CodeGen Configurations ---------------------===//
//
// This file contains CodeGen configurations for Adreno GPUs.
//
//===----------------------------------------------------------------------===//

#include <array>

#include "iree/compiler/Codegen/SPIRV/KernelConfig.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace iree_compiler {
namespace detail {

static LogicalResult setAdrenoMatmulConfig(linalg::LinalgOp op,
                                           spirv::ResourceLimitsAttr limits) {
  const int subgroupSize = limits.getSubgroupSize();
  const std::array<int64_t, 2> workgroupXY = {subgroupSize / 2, 2};
  std::array<int64_t, 3> threadMNK;
  auto inputType = op.getDpsInputOperand(0)->get().getType().cast<ShapedType>();
  if (inputType.getElementType().getIntOrFloatBitWidth() == 16) {
    threadMNK = {16, 8, 8};
  } else {
    threadMNK = {16, 4, 4};
  }
  return setMatmulOpConfig(limits, op, workgroupXY, threadMNK);
}

//===----------------------------------------------------------------------===//
// Entry Point
//===----------------------------------------------------------------------===//

LogicalResult setAdrenoCodeGenConfig(const spirv::TargetEnv &targetEnv,
                                     Operation *rootOp) {
  spirv::ResourceLimitsAttr limits = targetEnv.getResourceLimits();
  int subgroupSize = limits.getSubgroupSize();

  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(rootOp)) {
    if (isMatmulOrBatchMatmul(linalgOp))
      return setAdrenoMatmulConfig(linalgOp, limits);
  }

  return TypeSwitch<Operation *, LogicalResult>(rootOp)
      .Case<linalg::BatchMatmulOp, linalg::MatmulOp>(
          [limits](auto op) { return setAdrenoMatmulConfig(op, limits); })
      .Case<linalg::Conv2DNchwFchwOp, linalg::Conv2DNhwcHwcfOp>(
          [subgroupSize](auto op) {
            return setConvOpConfig(op, subgroupSize,
                                   /*bestTilingFactor=*/32);
          })
      .Case<linalg::DepthwiseConv2DNhwcHwcOp>([subgroupSize](auto op) {
        return setConvOpConfig(op, subgroupSize,
                               /*bestTilingFactor=*/16);
      })
      .Default([](Operation *) { return failure(); });
}

}  // namespace detail
}  // namespace iree_compiler
}  // namespace mlir

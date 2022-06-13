// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- AppleConfig.h - Apple CodeGen Configurations -----------------------===//
//
// This file contains CodeGen configurations for Apple GPUs.
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

//===----------------------------------------------------------------------===//
// Entry Point
//===----------------------------------------------------------------------===//

LogicalResult setAppleCodeGenConfig(const spirv::TargetEnv &targetEnv,
                                    Operation *rootOp) {
  int subgroupSize = targetEnv.getResourceLimits().getSubgroup_size();
  return TypeSwitch<Operation *, LogicalResult>(rootOp)
      .Case<linalg::BatchMatmulOp, linalg::MatmulOp>([subgroupSize](auto op) {
        std::array<int64_t, 2> workgroupXY = {256, 1};
        std::array<int64_t, 3> threadMNK = {4, 4, 4};
        return setMatmulOpConfig(op, subgroupSize, workgroupXY, threadMNK);
      })
      .Case<linalg::Conv2DNhwcHwcfOp>([subgroupSize](auto op) {
        return setConvOpConfig(op, subgroupSize,
                               /*bestTilingFactor=*/16);
      })
      .Case<linalg::DepthwiseConv2DNhwcHwcOp>([subgroupSize](auto op) {
        return setConvOpConfig(op, subgroupSize,
                               /*bestTilingFactor=*/16);
      })
      .Default([](Operation *) { return success(); });
}

}  // namespace detail
}  // namespace iree_compiler
}  // namespace mlir

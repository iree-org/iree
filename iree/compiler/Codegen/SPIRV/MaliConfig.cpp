// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- MaliConfig.h - Mali CodeGen Configurations -------------------------===//
//
// This file contains CodeGen configurations for Mali GPUs.
//
//===----------------------------------------------------------------------===//

#include <array>

#include "iree/compiler/Codegen/SPIRV/KernelConfig.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"

namespace mlir {
namespace iree_compiler {
namespace detail {

//===----------------------------------------------------------------------===//
// Entry Point
//===----------------------------------------------------------------------===//

LogicalResult setMaliCodeGenConfig(const spirv::TargetEnv &targetEnv,
                                   Operation *rootOp) {
  int64_t subgroupSize = targetEnv.getResourceLimits().subgroup_size().getInt();
  return TypeSwitch<Operation *, LogicalResult>(rootOp)
      .Case<linalg::BatchMatmulOp, linalg::MatmulOp>([](auto op) {
        std::array<int64_t, 2> workgroupXY = {8, 2};
        std::array<int64_t, 3> threadMNK;
        auto inputType = op.inputs()[0].getType().template cast<ShapedType>();
        if (inputType.getElementType().isF16()) {
          threadMNK = {8, 8, 4};
        } else {
          threadMNK = {6, 4, 4};
        }
        return setMatmulOpConfig(op, workgroupXY, threadMNK);
      })
      .Case<linalg::Conv2DNhwcHwcfOp>([subgroupSize](auto op) {
        return setConvOpConfig(op, subgroupSize,
                               /*bestTilingFactor=*/16);
      })
      .Case<linalg::DepthwiseConv2DNhwOp>([subgroupSize](auto op) {
        return setConvOpConfig(op, subgroupSize,
                               /*bestTilingFactor=*/16);
      })
      .Default([](Operation *) { return success(); });
}

}  // namespace detail
}  // namespace iree_compiler
}  // namespace mlir

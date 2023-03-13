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
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace mlir {
namespace iree_compiler {
namespace detail {

static LogicalResult setMaliMatmulConfig(linalg::LinalgOp op,
                                         spirv::ResourceLimitsAttr limits) {
  const int subgroupSize = limits.getSubgroupSize();
  const std::array<int64_t, 2> workgroupXY = {subgroupSize / 2, 2};
  std::array<int64_t, 3> threadMNK;
  Type inputType = op.getDpsInputOperand(0)->get().getType();
  Type elementType = inputType.cast<ShapedType>().getElementType();
  if (elementType.getIntOrFloatBitWidth() == 16) {
    threadMNK = {2, 8, 8};
  } else if (elementType.isInteger(8)) {
    threadMNK = {4, 4, 4};
  } else {
    threadMNK = {6, 4, 4};
  }
  return setMatmulOpConfig(limits, op, workgroupXY, threadMNK);
}

//===----------------------------------------------------------------------===//
// Entry Point
//===----------------------------------------------------------------------===//

LogicalResult setMaliCodeGenConfig(const spirv::TargetEnv &targetEnv,
                                   Operation *rootOp) {
  spirv::ResourceLimitsAttr limits = targetEnv.getResourceLimits();
  int subgroupSize = limits.getSubgroupSize();

  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(rootOp)) {
    if (isMatmulOrBatchMatmul(linalgOp))
      return setMaliMatmulConfig(linalgOp, limits);
  }

  return TypeSwitch<Operation *, LogicalResult>(rootOp)
      .Case<linalg::BatchMatmulOp, linalg::MatmulOp>(
          [limits](auto op) { return setMaliMatmulConfig(op, limits); })
      .Case<linalg::Conv2DNchwFchwOp, linalg::Conv2DNhwcHwcfOp>(
          [subgroupSize](auto op) {
            bool hasPaddedInput =
                op.image().template getDefiningOp<tensor::PadOp>();
            int bestTilingFactor = hasPaddedInput ? 8 : 16;
            return setConvOpConfig(op, subgroupSize, bestTilingFactor);
          })
      .Case<linalg::DepthwiseConv2DNhwcHwcOp>([subgroupSize](auto op) {
        bool hasPaddedInput =
            op.image().template getDefiningOp<tensor::PadOp>();
        int bestTilingFactor = hasPaddedInput ? 8 : 16;
        return setConvOpConfig(op, subgroupSize, bestTilingFactor);
      })
      .Default([](Operation *) { return failure(); });
}

}  // namespace detail
}  // namespace iree_compiler
}  // namespace mlir

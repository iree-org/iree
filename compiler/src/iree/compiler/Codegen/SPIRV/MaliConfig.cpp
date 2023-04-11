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
    threadMNK = {4, 4, 16};
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

  if (auto convOp = dyn_cast<linalg::ConvolutionOpInterface>(rootOp)) {
    auto type = cast<ShapedType>(convOp.image().getType());
    const int bitwidth = type.getElementTypeBitWidth();
    if (bitwidth > 32) return failure();
    const int multipler = 32 / bitwidth;
    bool hasPaddedInput = convOp.image().getDefiningOp<tensor::PadOp>();
    const int bestTilingFactor = (hasPaddedInput ? 8 : 16) * multipler;
    return setConvOpConfig(rootOp, subgroupSize, bestTilingFactor);
  }

  return failure();
}

}  // namespace detail
}  // namespace iree_compiler
}  // namespace mlir

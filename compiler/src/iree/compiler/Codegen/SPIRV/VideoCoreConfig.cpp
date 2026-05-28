// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- VideoCoreConfig.cpp - VideoCore CodeGen Configurations -------------===//
//
// This file contains CodeGen configurations for Broadcom VideoCore GPUs.
//
//===----------------------------------------------------------------------===//

#include <array>

#include "iree/compiler/Codegen/SPIRV/KernelConfig.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"

namespace mlir::iree_compiler::detail {

static LogicalResult setVideoCoreMatmulConfig(linalg::LinalgOp op,
                                              IREE::GPU::TargetAttr target) {
  auto inputType =
      llvm::cast<ShapedType>(op.getDpsInputOperand(0)->get().getType());
  // Restrict the tilings to just float32 as we have not investigated the
  // optimal tiling for f16 or other types yet.
  if (!inputType.getElementType().isF32()) {
    return failure();
  }
  const std::array<int64_t, 2> workgroupXY = {16, 16};
  const std::array<int64_t, 3> threadMNK = {4, 4, 4};
  return setMatmulOpConfig(target, op, workgroupXY, threadMNK);
}

//===----------------------------------------------------------------------===//
// Entry Point
//===----------------------------------------------------------------------===//

LogicalResult setVideoCoreCodeGenConfig(IREE::GPU::TargetAttr target,
                                        Operation *rootOp) {
  if (!isa<linalg::LinalgOp>(rootOp)) {
    return failure();
  }

  auto linalgOp = cast<linalg::LinalgOp>(rootOp);
  if (isMatmulOrBatchMatmul(linalgOp) || isa<linalg::MatmulOp>(linalgOp) ||
      isa<linalg::BatchMatmulOp>(linalgOp)) {
    return setVideoCoreMatmulConfig(linalgOp, target);
  }

  // TODO: add optimized tilings for Conv2d.

  return failure();
}

} // namespace mlir::iree_compiler::detail

// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMCPU/Utils.h"

#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#define DEBUG_TYPE "iree-llvmcpu-utils"

namespace mlir {
namespace iree_compiler {

bool preferIntrinsicsOverAsm(IREE::HAL::ExecutableTargetAttr targetAttr) {
  auto intrinsicsAttr =
      getConfigBoolAttr(targetAttr, "prefer_intrinsics_over_asm");
  return intrinsicsAttr && intrinsicsAttr->getValue();
}

bool hasAVX2Feature(IREE::HAL::ExecutableTargetAttr targetAttr) {
  return hasFeature(targetAttr, "+avx2");
}

bool hasAVX512fFeature(IREE::HAL::ExecutableTargetAttr targetAttr) {
  return hasFeature(targetAttr, "+avx512f");
}

bool hasVFeature(IREE::HAL::ExecutableTargetAttr targetAttr) {
  return hasFeature(targetAttr, "+v");
}

bool hasZve32xFeature(IREE::HAL::ExecutableTargetAttr targetAttr) {
  return hasFeature(targetAttr, "+zve32x");
}

bool hasZve32fFeature(IREE::HAL::ExecutableTargetAttr targetAttr) {
  return hasFeature(targetAttr, "+zve32f");
}

bool hasZve64xFeature(IREE::HAL::ExecutableTargetAttr targetAttr) {
  return hasFeature(targetAttr, "+zve64x");
}

bool hasAnySVEFeature(IREE::HAL::ExecutableTargetAttr targetAttr) {
  return hasFeature(targetAttr, "+sve") || hasFeature(targetAttr, "+sve2");
}

bool hasSMEFeature(IREE::HAL::ExecutableTargetAttr targetAttr) {
  return hasFeature(targetAttr, "+sme");
}

FailureOr<Operation *> getRootOperation(ArrayRef<Operation *> computeOps) {
  Operation *rootOperation = nullptr;
  for (auto op : llvm::reverse(computeOps)) {
    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
      // Do not treat linalg ops that are all parallel as root operations in
      // this sweep.
      if (linalgOp.getNumLoops() == linalgOp.getNumParallelLoops())
        continue;

      // All other linalg ops are root ops.
      rootOperation = op;
      break;
    }

    if (isa<TilingInterface>(op) &&
        !isa<tensor::PadOp, tensor::PackOp, tensor::UnPackOp>(op)) {
      // All other operations that implement this interface are root ops.
      rootOperation = op;
      break;
    }
  }

  if (!rootOperation) {
    // Check for elementwise operations.
    for (auto op : llvm::reverse(computeOps)) {
      if (isa<linalg::LinalgOp>(op)) {
        rootOperation = op;
        break;
      }
    }
  }

  if (!rootOperation) {
    // Check for pad/pack/unpack ops by themselves.
    for (auto op : llvm::reverse(computeOps)) {
      if (isa<tensor::PadOp, tensor::PackOp, tensor::UnPackOp>(op)) {
        rootOperation = op;
        break;
      }
    }
  }

  return rootOperation;
}

bool hasByteAlignedElementTypes(linalg::LinalgOp linalgOp) {
  return llvm::all_of(linalgOp->getOperands(), [](Value operand) {
    auto bitwidth =
        getElementTypeOrSelf(operand.getType()).getIntOrFloatBitWidth();
    return bitwidth % 8 == 0;
  });
}

} // namespace iree_compiler
} // namespace mlir

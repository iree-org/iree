// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Utils/CPUUtils.h"

#include <numeric>

#include "iree/builtins/ukernel/exported_bits.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#define DEBUG_TYPE "iree-codegen-cpu-utils"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

namespace mlir::iree_compiler {

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

uint32_t findMmt4dUkernelType(Type lhsElemType, Type rhsElemType,
                              Type outElemType) {
  if (lhsElemType.isSignlessInteger(8) && rhsElemType.isSignlessInteger(8) &&
      outElemType.isSignlessInteger(32)) {
    return IREE_UK_FLAG_MMT4D_TYPE_S8S8S32;
  } else if (lhsElemType.isSignlessInteger(8) && rhsElemType.isSignlessInteger(4) &&
      outElemType.isSignlessInteger(32)) {
    return IREE_UK_FLAG_MMT4D_TYPE_S8S4S32;
  } else if (lhsElemType.isSignlessInteger(16) &&
             rhsElemType.isSignlessInteger(16) &&
             outElemType.isSignlessInteger(32)) {
    return IREE_UK_FLAG_MMT4D_TYPE_S16S16S32;
  } else if (lhsElemType.isSignlessInteger(16) &&
             rhsElemType.isUnsignedInteger(4) &&
             outElemType.isSignlessInteger(32)) {
    return IREE_UK_FLAG_MMT4D_TYPE_S16U4S32;
  } else if (lhsElemType.isSignlessInteger(16) &&
             rhsElemType.isSignlessInteger(8) &&
             outElemType.isSignlessInteger(32)) {
    return IREE_UK_FLAG_MMT4D_TYPE_S16S8S32;
  } else if (lhsElemType.isF32() && rhsElemType.isF32() &&
             outElemType.isF32()) {
    return IREE_UK_FLAG_MMT4D_TYPE_F32F32F32;
  } else if (lhsElemType.isF16() && rhsElemType.isF16() &&
             outElemType.isF32()) {
    return IREE_UK_FLAG_MMT4D_TYPE_F16F16F32;
  } else if (lhsElemType.isF16() && rhsElemType.isF16() &&
             outElemType.isF16()) {
    return IREE_UK_FLAG_MMT4D_TYPE_F16F16F16;
  } else if (lhsElemType.isBF16() && rhsElemType.isBF16() &&
             outElemType.isF32()) {
    return IREE_UK_FLAG_MMT4D_TYPE_BF16BF16F32;
  } else if (lhsElemType.isBF16() && rhsElemType.isBF16() &&
             outElemType.isBF16()) {
    return IREE_UK_FLAG_MMT4D_TYPE_BF16BF16BF16;
  }
  return IREE_UK_FLAG_MMT4D_TYPE_NONE;
}

} // namespace mlir::iree_compiler

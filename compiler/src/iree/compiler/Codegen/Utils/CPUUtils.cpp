// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Utils/CPUUtils.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"

#include <numeric>

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
        !isa<tensor::PadOp, linalg::PackOp, linalg::UnPackOp>(op)) {
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
      if (isa<tensor::PadOp, linalg::PackOp, linalg::UnPackOp>(op)) {
        rootOperation = op;
        break;
      }
    }
  }

  return rootOperation;
}

static const char kDecompositionAttrName[] = "enable_decomposition";
StringAttr getEnableDecompositionAttrName(MLIRContext *ctx) {
  return StringAttr::get(ctx, kDecompositionAttrName);
}
std::string getEnableDecompositionStr() { return kDecompositionAttrName; }

static const char kLoopPeelingAttrName[] = "enable_loop_peeling";
StringAttr getEnableLoopPeelingAttrName(MLIRContext *ctx) {
  return StringAttr::get(ctx, kLoopPeelingAttrName);
}
std::string getEnableLoopPeelingStr() { return kLoopPeelingAttrName; }

bool isOptEnabled(FunctionOpInterface funcOp, StringRef label) {
  DictionaryAttr config = getTranslationInfo(funcOp).getConfiguration();
  return config && config.contains(label);
}

} // namespace mlir::iree_compiler

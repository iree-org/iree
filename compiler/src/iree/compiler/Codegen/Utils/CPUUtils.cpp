// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Utils/CPUUtils.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"

#include <numeric>

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#define DEBUG_TYPE "iree-codegen-cpu-utils"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

namespace mlir::iree_compiler {

static llvm::cl::opt<bool> clEnableScalableVectorization(
    "iree-llvmcpu-enable-scalable-vectorization",
    llvm::cl::desc("Enable scalable vectorization if it is supported by the "
                   "target (e.g., +sve, +sve2 and/or +sme feature flags)"),
    llvm::cl::init(false));

static llvm::cl::opt<int> clVscaleFromUser(
    "iree-experimental-vscale-value",
    llvm::cl::desc(
        "The runtime value of vscale. This will _only_ be used for host-side "
        "code, e.g. to calculate storage sizes and workgroup counts. This is "
        "due to a current limitation of the host-side code not being able to "
        "properly query this value at runtime, see #21317 and #21590. Codegen "
        "will be vector-length agnostic and will be querying the value of "
        "vscale at runtime, as intended. For scalable vector code that "
        "propagates vscale ops into the host-side code, this value has to be "
        "explicitly set by the user, e.g. for SVE data-tiling on the AArch64 "
        "backend."),
    llvm::cl::Hidden, llvm::cl::init(-1));

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

bool isScalableVectorizationEnabled() { return clEnableScalableVectorization; }

unsigned getUserVscaleValue() {
  assert(clVscaleFromUser >= 1 && "Currently, vscale needs to be specified by "
                                  "the user for host-side code!");
  return clVscaleFromUser;
}

} // namespace mlir::iree_compiler

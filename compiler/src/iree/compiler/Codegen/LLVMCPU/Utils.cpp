// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMCPU/Utils.h"

#include "iree/compiler/Codegen/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/ADT/STLExtras.h"

#define DEBUG_TYPE "iree-llvmcpu-utils"

namespace mlir {
namespace iree_compiler {

std::optional<StringRef>
getCpuFeatures(IREE::HAL::ExecutableTargetAttr targetAttr) {
  auto cpuFeatures = getConfigStringAttr(targetAttr, "cpu_features");
  if (!cpuFeatures)
    return std::nullopt;
  return cpuFeatures->getValue();
}

bool isX86(IREE::HAL::ExecutableTargetAttr targetAttr) {
  std::optional<llvm::Triple> triple = getTargetTriple(targetAttr);
  return triple && triple.value().isX86();
}

bool isX86_64(IREE::HAL::ExecutableTargetAttr targetAttr) {
  std::optional<llvm::Triple> triple = getTargetTriple(targetAttr);
  return triple && triple.value().getArch() == llvm::Triple::x86_64;
}

bool isAArch64(IREE::HAL::ExecutableTargetAttr targetAttr) {
  std::optional<llvm::Triple> triple = getTargetTriple(targetAttr);
  return triple && triple.value().isAArch64();
}

bool isRISCV(IREE::HAL::ExecutableTargetAttr targetAttr) {
  std::optional<llvm::Triple> triple = getTargetTriple(targetAttr);
  return triple && triple.value().isRISCV();
}

bool preferIntrinsicsOverAsm(IREE::HAL::ExecutableTargetAttr targetAttr) {
  auto intrinsicsAttr =
      getConfigBoolAttr(targetAttr, "prefer_intrinsics_over_asm");
  return intrinsicsAttr && intrinsicsAttr->getValue();
}

// TODO(dcaballe): If we have to check for a significantly large number of
// features in the future, we may want to consider a persistent state to carry
// over processed HAL information or keeping the TTI instance alive and query
// subtarget features data structure.
bool hasFeature(IREE::HAL::ExecutableTargetAttr targetAttr, StringRef feature) {
  std::optional<StringRef> features = getCpuFeatures(targetAttr);
  if (!features) {
    return false;
  }

  // Find feature string in list of features, making sure that we don't match a
  // sub-string.
  std::stringstream sstream(features->str());
  std::string str;
  while (std::getline(sstream, str, ',')) {
    if (str == feature) {
      return true;
    }
  }

  return false;
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

} // namespace iree_compiler
} // namespace mlir

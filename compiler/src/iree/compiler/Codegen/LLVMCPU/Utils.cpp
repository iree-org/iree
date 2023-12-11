// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMCPU/Utils.h"

#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

#define DEBUG_TYPE "iree-llvmcpu-utils"

namespace mlir::iree_compiler {

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

} // namespace mlir::iree_compiler

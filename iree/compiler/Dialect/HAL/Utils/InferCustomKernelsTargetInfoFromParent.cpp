// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Utils/InferCustomKernelsTargetInfoFromParent.h"

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Utils/CustomKernelsTargetInfo.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace iree_compiler {

LogicalResult InferCustomKernelsTargetInfoFromParent(
    FuncOp entryPointFn, CustomKernelsTargetInfo &targetInfo) {
  // Set the out-value to defaults early so that early returns produce
  // consistent results and so that we can write simpler code below
  // (for loop OR-ing booleans, assuming initial 'false' value).
  targetInfo = CustomKernelsTargetInfo();

  // Try to find the parent ExecutableVariantOp and its relevant attributes.
  auto variantOp =
      entryPointFn->getParentOfType<IREE::HAL::ExecutableVariantOp>();
  if (!variantOp) {
    return failure();
  }
  IREE::HAL::ExecutableTargetAttr targetAttr = variantOp.target();
  if (!targetAttr) {
    return failure();
  }
  auto config = targetAttr.getConfiguration();
  if (!config) {
    return failure();
  }
  auto tripleAttr = config.getAs<StringAttr>("target_triple");
  if (!tripleAttr) {
    return failure();
  }
  auto cpuFeaturesAttr = config.getAs<StringAttr>("cpu_features");
  if (!cpuFeaturesAttr) {
    return failure();
  }

  // Exactly the implementation of llvm::Triple::getArchName, skipping all the
  // parsing work of constructing a llvm::Triple from a string.
  llvm::StringRef archName(tripleAttr.getValue().split('-').first);
  llvm::StringRef featuresStr(cpuFeaturesAttr.getValue());
  return ParseCustomKernelsTargetInfo(archName, featuresStr, targetInfo);
}

}  // namespace iree_compiler
}  // namespace mlir

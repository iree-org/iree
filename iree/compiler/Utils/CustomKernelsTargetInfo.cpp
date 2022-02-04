// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Utils/CustomKernelsTargetInfo.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"

namespace mlir {
namespace iree_compiler {

LogicalResult ParseCustomKernelTargetFeaturesForAarch64(
    const llvm::SmallVector<llvm::StringRef> &features,
    CustomKernelsTargetInfo &target_info) {
  for (auto f : features) {
    if (f == "+dotprod") {
      target_info.add(CustomKernelTargetFeature::Aarch64Dotprod);
    } else {
      return failure();
    }
  }
  return success();
}

LogicalResult ParseCustomKernelsTargetInfo(
    llvm::StringRef archStr, llvm::StringRef featuresStr,
    CustomKernelsTargetInfo &target_info) {
  // Set the out-value to defaults early so that early returns produce
  // consistent results and so that we can write simpler code below.
  target_info = CustomKernelsTargetInfo();

  if (archStr.empty()) {
    return success();
  }

  llvm::SmallVector<llvm::StringRef> features;
  featuresStr.split(features, ',');

  if (archStr == "aarch64") {
    target_info.init(CustomKernelTargetArch::Aarch64);
    return ParseCustomKernelTargetFeaturesForAarch64(features, target_info);
  }

  return failure();
}

}  // namespace iree_compiler
}  // namespace mlir

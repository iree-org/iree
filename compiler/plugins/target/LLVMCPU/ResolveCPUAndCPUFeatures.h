// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_PLUGINS_TARGET_LLVMCPU_RESOLVECPUANDCPUFEATURES_H_
#define IREE_COMPILER_PLUGINS_TARGET_LLVMCPU_RESOLVECPUANDCPUFEATURES_H_

#include <string>
#include <string_view>

namespace mlir::iree_compiler::IREE::HAL {

enum class ResolveCPUAndCPUFeaturesStatus {
  OK,
  InconsistentHost,
  UnimplementedMapping,
  ImplicitGenericFallback
};

// Given an input `triple` and the input-output parameters `cpu` and
// `cpuFeatures`, which may be empty or the special "host" value, this function
// populates `cpu` and `cpuFeatures` with all the information that is known.
ResolveCPUAndCPUFeaturesStatus
resolveCPUAndCPUFeatures(std::string_view triple, std::string &cpu,
                         std::string &cpuFeatures);

std::string getMessage(ResolveCPUAndCPUFeaturesStatus status,
                       std::string_view triple);

} // namespace mlir::iree_compiler::IREE::HAL

#endif // IREE_COMPILER_PLUGINS_TARGET_LLVMCPU_RESOLVECPUANDCPUFEATURES_H_

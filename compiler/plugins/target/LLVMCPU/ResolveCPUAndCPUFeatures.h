// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_PLUGINS_TARGET_LLVMCPU_RESOLVECPUANDCPUFEATURES_H_
#define IREE_COMPILER_PLUGINS_TARGET_LLVMCPU_RESOLVECPUANDCPUFEATURES_H_

#include <string_view>

#include "llvm/TargetParser/Triple.h"

namespace mlir::iree_compiler::IREE::HAL {

bool resolveCPUAndCPUFeatures(const llvm::Triple &triple, std::string &cpu,
                              std::string &cpuFeatures,
                              std::string_view loggingUnspecifiedTargetCPU);

} // namespace mlir::iree_compiler::IREE::HAL

#endif // IREE_COMPILER_PLUGINS_TARGET_LLVMCPU_RESOLVECPUANDCPUFEATURES_H_

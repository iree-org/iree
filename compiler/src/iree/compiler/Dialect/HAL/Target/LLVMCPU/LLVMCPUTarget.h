// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef IREE_COMPILER_DIALECT_HAL_TARGET_LLVMCPU_LLVMCPUTARGET_H_
#define IREE_COMPILER_DIALECT_HAL_TARGET_LLVMCPU_LLVMCPUTARGET_H_

#include "iree/compiler/Dialect/HAL/Target/LLVMCPU/LLVMTargetOptions.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

// Registers the LLVM CPU target backends.
void registerLLVMCPUTargetBackends(
    std::function<LLVMTargetOptions()> queryOptions);

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_HAL_TARGET_LLVMCPU_LLVMCPUTARGET_H_

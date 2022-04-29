// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef IREE_COMPILER_DIALECT_HAL_TARGET_LLVM_LLVMAOTTARGET_H_
#define IREE_COMPILER_DIALECT_HAL_TARGET_LLVM_LLVMAOTTARGET_H_

#include "iree/compiler/Dialect/HAL/Target/LLVM/LLVMTargetOptions.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

// Registers the LLVM Ahead-Of-Time (AOT) target backends.
void registerLLVMAOTTargetBackends(
    std::function<LLVMTargetOptions()> queryOptions);

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_HAL_TARGET_LLVM_LLVMAOTTARGET_H_

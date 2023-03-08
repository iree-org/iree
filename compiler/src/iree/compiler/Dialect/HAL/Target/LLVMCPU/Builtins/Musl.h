// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_HAL_TARGET_LLVMCPU_BUILTINS_MUSL_H_
#define IREE_COMPILER_DIALECT_HAL_TARGET_LLVMCPU_BUILTINS_MUSL_H_

#include "llvm/IR/Module.h"
#include "llvm/Target/TargetMachine.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

llvm::Expected<std::unique_ptr<llvm::Module>> loadMuslBitcode(
    llvm::TargetMachine *targetMachine, llvm::LLVMContext &context);

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_HAL_TARGET_LLVMCPU_BUILTINS_MUSL_H_

// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_HAL_UTILS_LLVMCODEGENUTILS_H_
#define IREE_COMPILER_DIALECT_HAL_UTILS_LLVMCODEGENUTILS_H_

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace llvm {
class TargetMachine;
} // namespace llvm

namespace mlir::iree_compiler::IREE::HAL {

// Propagate the target features and target cpu to individual LLVMFuncOp within
// this module if their corresponding attributes haven't been set there.
void populateLLVMFuncTargetAttrs(ModuleOp moduleOp,
                                 const llvm::TargetMachine &targetMachine);

} // namespace mlir::iree_compiler::IREE::HAL

#endif // IREE_COMPILER_DIALECT_HAL_UTILS_LLVMCODEGENUTILS_H_

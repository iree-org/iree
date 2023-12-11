// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_HAL_TARGET_LLVMCPU_BUILTINS_DEVICE_H_
#define IREE_COMPILER_DIALECT_HAL_TARGET_LLVMCPU_BUILTINS_DEVICE_H_

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "llvm/IR/Module.h"
#include "llvm/Target/TargetMachine.h"

namespace mlir::iree_compiler::IREE::HAL {

// Loads the libdevice bitcode file and specializes it for |targetMachine|.
llvm::Expected<std::unique_ptr<llvm::Module>>
loadDeviceBitcode(llvm::TargetMachine *targetMachine,
                  llvm::LLVMContext &context);

// Specializes |module| using |targetMachine|.
void specializeDeviceModule(IREE::HAL::ExecutableVariantOp variantOp,
                            llvm::Module &module,
                            llvm::TargetMachine &targetMachine);

} // namespace mlir::iree_compiler::IREE::HAL

#endif // IREE_COMPILER_DIALECT_HAL_TARGET_LLVMCPU_BUILTINS_DEVICE_H_

// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_HAL_TARGET_LLVMCPU_LLVMIRPASSES_H_
#define IREE_COMPILER_DIALECT_HAL_TARGET_LLVMCPU_LLVMIRPASSES_H_

#include <memory>

#include "iree/compiler/Dialect/HAL/Target/LLVMCPU/LLVMTargetOptions.h"
#include "llvm/IR/Module.h"
#include "llvm/Target/TargetMachine.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::iree_compiler::IREE::HAL {

// Creates and runs LLVMIR optimization passes defined in LLVMTargetOptions.
LogicalResult runLLVMIRPasses(const LLVMTarget &target,
                              llvm::TargetMachine *machine,
                              llvm::Module *module);

// Emits compiled module obj for the target machine.
LogicalResult runEmitObjFilePasses(llvm::TargetMachine *machine,
                                   llvm::Module *module,
                                   llvm::CodeGenFileType fileType,
                                   std::string *objData);

} // namespace mlir::iree_compiler::IREE::HAL

#endif // IREE_COMPILER_DIALECT_HAL_TARGET_LLVMCPU_LLVMIRPASSES_H_

// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_HAL_TARGET_LLVMLINKERUTILS_H_
#define IREE_COMPILER_DIALECT_HAL_TARGET_LLVMLINKERUTILS_H_

#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "llvm/IR/Module.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Target/TargetMachine.h"

namespace mlir::iree_compiler::IREE::HAL {

// Returns true if there are any external symbols in |module| with |prefix|.
bool anyRequiredSymbols(const llvm::Module &module, StringRef prefix);

// User callback to inject custom global values or functions into |module| prior
// to linking.
using ModuleSpecializationCallback = std::function<void(llvm::Module &module)>;

// Verifies a bitcode library is loaded correctly and appends it to |linker|.
// Only symbols used by libraries already added to the |linker| will be linked.
// A |specializationCallback| is provided to allow for substituting global
// values or injecting target-specific specializations prior to linking.
//
// Example:
//  if (failed(linkBitcodeModule(loc, linker, linkerFlags, targetMachine,
//                               "libfoo", loadLibFoo(...))))
LogicalResult linkBitcodeModule(
    Location loc, llvm::Linker &linker, unsigned linkerFlags,
    llvm::TargetMachine &targetMachine, StringRef name,
    llvm::Expected<std::unique_ptr<llvm::Module>> bitcodeModuleValue,
    ModuleSpecializationCallback specializationCallback = {});

// Loads a bitcode file specified by the |objectAttr| and specializes it for
// |targetMachine|.
llvm::Expected<std::unique_ptr<llvm::Module>>
loadBitcodeObject(IREE::HAL::ExecutableObjectAttr objectAttr,
                  llvm::LLVMContext &context);

// Links all .bc objects in |objectAttrs| into |linker|.
LogicalResult
linkBitcodeObjects(Location loc, llvm::Linker &linker, unsigned linkerFlags,
                   llvm::TargetMachine &targetMachine, ArrayAttr objectAttrs,
                   llvm::LLVMContext &context,
                   ModuleSpecializationCallback specializationCallback = {});

LogicalResult linkCmdlineBitcodeFiles(Location loc, llvm::Linker &linker,
                                      unsigned linkerFlags,
                                      llvm::TargetMachine &targetMachine,
                                      llvm::LLVMContext &context);

} // namespace mlir::iree_compiler::IREE::HAL

#endif //  IREE_COMPILER_DIALECT_HAL_TARGET_LLVMLINKERUTILS_H_

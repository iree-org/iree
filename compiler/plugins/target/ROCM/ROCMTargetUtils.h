// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_PLUGINS_TARGET_ROCM_ROCMTARGETUTILS_H_
#define IREE_COMPILER_PLUGINS_TARGET_ROCM_ROCMTARGETUTILS_H_

#include "iree/compiler/Dialect/HAL/Target/TargetBackend.h"
#include "llvm/IR/Module.h"
#include "llvm/Target/TargetMachine.h"

namespace mlir::iree_compiler::IREE::HAL {

// Sets HIP platform globals based on the target architecture.
LogicalResult setHIPGlobals(Location loc, llvm::Module *module,
                            StringRef targetChip);

// Links HIP device bitcode if the module uses any symbols from it.
LogicalResult linkHIPBitcodeIfNeeded(Location loc, llvm::Module *module,
                                     StringRef targetChip,
                                     StringRef bitcodePath);

// Links optimized Ukernel module.
LogicalResult linkUkernelBitcodeFiles(Location loc, llvm::Module *module,
                                      StringRef enabledUkernelsStr,
                                      StringRef targetChip,
                                      StringRef bitcodePath,
                                      unsigned linkerFlags,
                                      llvm::TargetMachine &targetMachine);
// Compiles ISAToHsaco Code
std::string createHsaco(Location loc, const std::string isa, StringRef name);

// Returns true if the rocm archtecture target is supported for ukernels.
bool hasUkernelSupportedRocmArch(IREE::HAL::ExecutableTargetAttr targetAttr);

} // namespace mlir::iree_compiler::IREE::HAL

#endif // IREE_COMPILER_PLUGINS_TARGET_ROCM_ROCMTARGETUTILS_H_

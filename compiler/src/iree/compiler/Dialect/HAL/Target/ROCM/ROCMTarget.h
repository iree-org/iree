// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_HAL_TARGET_ROCM_ROCMTARGET_H_
#define IREE_COMPILER_DIALECT_HAL_TARGET_ROCM_ROCMTARGET_H_

#include "iree/compiler/Dialect/HAL/Target/TargetBackend.h"
#include "llvm/IR/Module.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

// Registers the ROCM backend.
void registerROCMTargetBackends();

// Links LLVM module to ROC Device Library Bit Code
void linkROCDLIfNecessary(llvm::Module *module, std::string targetChip,
                          std::string bitCodeDir);

// Compiles ISAToHsaco Code
std::string createHsaco(Location loc, const std::string isa, StringRef name);

} // namespace HAL
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir

#endif // IREE_COMPILER_DIALECT_HAL_TARGET_ROCM_ROCMTARGET_H_

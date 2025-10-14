// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef COMPILER_PLUGINS_TARGET_LLVMCPU_DIALECT_PASSES_H_
#define COMPILER_PLUGINS_TARGET_LLVMCPU_DIALECT_PASSES_H_

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::iree_compiler::IREE::HAL {

//===----------------------------------------------------------------------===//
// Register all Passes
//===----------------------------------------------------------------------===//

#define GEN_PASS_DECL
#include "compiler/plugins/target/LLVMCPU/Passes.h.inc" // IWYU pragma: keep

void registerLLVMCPUTargetPasses();

} // namespace mlir::iree_compiler::IREE::HAL

#endif // COMPILER_PLUGINS_TARGET_LLVMCPU_PASSES_H_

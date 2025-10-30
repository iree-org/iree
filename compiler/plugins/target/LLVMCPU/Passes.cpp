// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/target/LLVMCPU/Passes.h"

namespace mlir::iree_compiler::IREE::HAL {

namespace {
#define GEN_PASS_REGISTRATION
#include "compiler/plugins/target/LLVMCPU/Passes.h.inc" // IWYU pragma: export
} // namespace

void registerLLVMCPUTargetPasses() { registerPasses(); }

} // namespace mlir::iree_compiler::IREE::HAL

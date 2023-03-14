// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This file defines a helper to trigger the registration of passes to
// the system.
//
// Based on MLIR's InitAllPasses but for IREE input passes.

#ifndef IREE_COMPILER_TOOLS_INIT_INPUT_PASSES_H_
#define IREE_COMPILER_TOOLS_INIT_INPUT_PASSES_H_

namespace mlir {
namespace iree_compiler {

// Registers IREE input conversion passes with the global registry.
void registerInputPasses();

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_TOOLS_INIT_INPUT_PASSES_H_

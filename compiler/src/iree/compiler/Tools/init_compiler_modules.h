// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_TOOLS_INIT_COMPILER_MODULES_H_
#define IREE_COMPILER_TOOLS_INIT_COMPILER_MODULES_H_

#include "iree/compiler/Modules/Check/IR/CheckDialect.h"

namespace mlir::iree_compiler {

// Add all the IREE compiler module dialects to the provided registry.
inline void registerIreeCompilerModuleDialects(DialectRegistry &registry) {
  // clang-format off
  registry.insert<IREE::Check::CheckDialect>();
  // clang-format on
}

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_TOOLS_INIT_COMPILER_MODULES_H_

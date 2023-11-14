// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This files defines a helper to trigger the registration of dialects to
// the system.
//
// Based on MLIR's InitAllDialects but for IREE dialects.

#ifndef IREE_COMPILER_TOOLS_INIT_DIALECTS_H_
#define IREE_COMPILER_TOOLS_INIT_DIALECTS_H_

#include "iree/compiler/Tools/init_compiler_modules.h"
#include "iree/compiler/Tools/init_iree_dialects.h"
#include "iree/compiler/Tools/init_mlir_dialects.h"

namespace mlir {
namespace iree_compiler {

inline void registerAllDialects(DialectRegistry &registry) {
  registerMlirDialects(registry);
  registerIreeDialects(registry);

  mlir::iree_compiler::registerIreeCompilerModuleDialects(registry);
}

} // namespace iree_compiler
} // namespace mlir

#endif // IREE_COMPILER_TOOLS_INIT_DIALECTS_H_

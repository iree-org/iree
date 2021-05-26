// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This files defines a helper to trigger the registration of dialects to
// the system.
//
// Based on MLIR's InitAllDialects but for IREE dialects.

#ifndef IREE_TOOLS_INIT_DIALECTS_H_
#define IREE_TOOLS_INIT_DIALECTS_H_

#include "iree/tools/init_compiler_modules.h"
#include "iree/tools/init_iree_dialects.h"
#include "iree/tools/init_mlir_dialects.h"
#include "iree/tools/init_xla_dialects.h"

#ifdef IREE_HAVE_EMITC_DIALECT
#include "emitc/InitDialect.h"
#endif  // IREE_HAVE_EMITC_DIALECT

namespace mlir {
namespace iree_compiler {

inline void registerAllDialects(DialectRegistry &registry) {
  registerIreeDialects(registry);
  registerMlirDialects(registry);
  mlir::registerXLADialects(registry);
  mlir::iree_compiler::registerIreeCompilerModuleDialects(registry);

#ifdef IREE_HAVE_EMITC_DIALECT
  mlir::registerEmitCDialect(registry);
#endif  // IREE_HAVE_EMITC_DIALECT
}

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_TOOLS_INIT_DIALECTS_H_

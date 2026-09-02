// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Touching MLIRContext forces the host to resolve renamed MLIR symbols at
// dlopen. This fails to load if either side got the rename wrong.

#include <cstdio>

#include "iree/compiler/PluginAPI/PluginEntryPoint.h"
#include "mlir/IR/MLIRContext.h"

static bool
registerDynamicTestPlugin(mlir::iree_compiler::PluginRegistrar *registrar) {
  (void)registrar;
  mlir::MLIRContext context;
  (void)context.isMultithreadingEnabled();
  std::fprintf(stderr, "DYNAMIC_TEST_PLUGIN: renamed MLIRContext ok\n");
  return true;
}

IREE_DEFINE_COMPILER_PLUGIN(dynamic_test_plugin, registerDynamicTestPlugin)

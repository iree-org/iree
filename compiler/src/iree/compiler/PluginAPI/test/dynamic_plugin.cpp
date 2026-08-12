// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Verification plugin for the renamed compiler ABI: instantiating
// mlir::MLIRContext forces resolution of renamed MLIR C++ symbols
// (_ZN6IREE184mlir...) from the host process at dlopen time. A plugin built
// without the rename, or a host library that fails to export the renamed
// symbols, makes this fail to load.

#include <cstdio>

#include "mlir/IR/MLIRContext.h"

extern "C" bool
iree_register_compiler_plugin_dynamic_test_plugin(void *registrar) {
  (void)registrar;
  mlir::MLIRContext context;
  (void)context.isMultithreadingEnabled();
  std::fprintf(stderr, "DYNAMIC_TEST_PLUGIN: renamed MLIRContext ok\n");
  return true;
}

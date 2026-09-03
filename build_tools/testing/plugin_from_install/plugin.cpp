// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Built against an IREE install tree, with no IREE sources in reach. Touching
// MLIRContext forces the host to resolve renamed MLIR symbols at dlopen, so a
// prefix or header hash the install tree got wrong fails the load.

#include <cstdio>

#include "helper.h"
#include "iree/compiler/PluginAPI/PluginEntryPoint.h"
#include "mlir/IR/MLIRContext.h"

static bool registerInstallTreePlugin(
    mlir::iree_compiler::PluginRegistrar* registrar) {
  (void)registrar;
  mlir::MLIRContext context;
  if (!helperTouchesContext(&context)) {
    return false;
  }
  std::fprintf(stderr, "INSTALL_TREE_PLUGIN: renamed MLIRContext ok\n");
  return true;
}

IREE_DEFINE_COMPILER_PLUGIN(install_tree_probe, registerInstallTreePlugin)

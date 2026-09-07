// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Exercise the public C++ plugin API using only installed IREE headers.

#include <cstdio>

#include "helper.h"
#include "iree/compiler/PluginAPI/Client.h"
#include "iree/compiler/PluginAPI/PluginEntryPoint.h"
#include "mlir/IR/MLIRContext.h"

namespace {
struct InstallTreeSession
    : mlir::iree_compiler::PluginSession<InstallTreeSession> {
  mlir::LogicalResult onActivate() override {
    if (!helperTouchesContext(context)) {
      return mlir::failure();
    }
    std::fprintf(stderr, "INSTALL_TREE_PLUGIN: session activated\n");
    return mlir::success();
  }
};
}  // namespace

static bool registerInstallTreePlugin(
    mlir::iree_compiler::PluginRegistrar* registrar) {
  registrar->registerPlugin<InstallTreeSession>("install_tree_probe");
  return true;
}

IREE_DEFINE_COMPILER_PLUGIN(install_tree_probe, registerInstallTreePlugin,
                            "0.1")

// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Built with -fvisibility=hidden, as an out-of-tree vendor would, so the macro
// has to export the entry point itself.

#include "iree/compiler/PluginAPI/PluginEntryPoint.h"

static bool registerHiddenPlugin(mlir::iree_compiler::PluginRegistrar *r) {
  (void)r;
  return true;
}

IREE_DEFINE_COMPILER_PLUGIN(hidden_plugin, registerHiddenPlugin, "test")

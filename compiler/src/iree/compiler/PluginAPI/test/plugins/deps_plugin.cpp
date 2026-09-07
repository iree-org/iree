// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// A plugin of more than one library, which is what any real one is.

#include "deps_helper.h"
#include "iree/compiler/PluginAPI/PluginEntryPoint.h"

static bool registerDepsPlugin(mlir::iree_compiler::PluginRegistrar *r) {
  (void)r;
  return iree_plugin_test::helperSucceeded();
}

IREE_DEFINE_COMPILER_PLUGIN(deps_plugin, registerDepsPlugin, "test")

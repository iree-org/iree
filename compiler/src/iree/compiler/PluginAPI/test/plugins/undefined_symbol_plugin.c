// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Calls a function nothing provides. Lazy binding would let this load and fail
// mid-compilation instead.

#include "iree/compiler/PluginAPI/PluginEntryPoint.h"

extern bool iree_test_absent_function(void);

static bool register_undefined_plugin(IreeCompilerPluginRegistrar *registrar) {
  (void)registrar;
  return iree_test_absent_function();
}

static const IreeCompilerPluginInfo info = {
    IREE_COMPILER_PLUGIN_API_VERSION, IREE_COMPILER_PLUGIN_ABI_HASH,
    "undefined_plugin", "test", register_undefined_plugin};

IREE_COMPILER_PLUGIN_EXPORT const IreeCompilerPluginInfo *
iree_get_compiler_plugin_info(void) {
  return &info;
}

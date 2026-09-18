// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// A plugin claiming a header surface this compiler does not have. It must be
// refused before its registration function is reached.

#include "iree/compiler/PluginAPI/PluginEntryPoint.h"

static bool
register_other_headers_plugin(IreeCompilerPluginRegistrar *registrar) {
  (void)registrar;
  return true;
}

static const IreeCompilerPluginInfo info = {
    IREE_COMPILER_PLUGIN_API_VERSION, "0000000000000000", "from_other_headers",
    "test", register_other_headers_plugin};

IREE_COMPILER_PLUGIN_EXPORT const IreeCompilerPluginInfo *
iree_get_compiler_plugin_info(void) {
  return &info;
}

// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/PluginAPI/PluginEntryPoint.h"

static bool register_failure(IreeCompilerPluginRegistrar *registrar) {
  (void)registrar;
  return false;
}

static const IreeCompilerPluginInfo info = {
    IREE_COMPILER_PLUGIN_API_VERSION, IREE_COMPILER_PLUGIN_ABI_HASH,
    "registration_failure", "test", register_failure};

IREE_COMPILER_PLUGIN_EXPORT const IreeCompilerPluginInfo *
iree_get_compiler_plugin_info(void) {
  return &info;
}

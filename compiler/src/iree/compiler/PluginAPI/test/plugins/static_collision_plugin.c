// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdlib.h>

#include "iree/compiler/PluginAPI/PluginEntryPoint.h"

static const char *const static_ids[] = {
#define HANDLE_PLUGIN_ID(id) #id,
#include "iree/compiler/PluginAPI/Config/StaticLinkedPlugins.inc"
#undef HANDLE_PLUGIN_ID
    "",
};

static bool register_collision(IreeCompilerPluginRegistrar *registrar) {
  // The loader must detect the collision before calling us.
  abort();
}

IREE_COMPILER_PLUGIN_EXPORT const IreeCompilerPluginInfo *
iree_get_compiler_plugin_info(void) {
  static IreeCompilerPluginInfo info = {IREE_COMPILER_PLUGIN_API_VERSION,
                                        IREE_COMPILER_PLUGIN_ABI_HASH, NULL,
                                        "test", register_collision};
  info.pluginId = static_ids[0];
  return &info;
}

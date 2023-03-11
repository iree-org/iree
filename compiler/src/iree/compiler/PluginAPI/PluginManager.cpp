// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/PluginAPI/PluginManager.h"

#include <stdio.h>

// Declare entrypoints for each statically registered plugin.
#define HANDLE_PLUGIN_ID(plugin_id)                          \
  extern "C" bool iree_register_compiler_plugin_##plugin_id( \
      mlir::iree_compiler::PluginRegistrar *);
#include "iree/compiler/PluginAPI/Config/StaticLinkedPlugins.inc"
#undef HANDLE_PLUGIN_ID

namespace mlir::iree_compiler {

bool PluginManager::initialize() {
// Initialize static plugins.
#define HANDLE_PLUGIN_ID(plugin_id)                                           \
  if (!registerPlugin(#plugin_id, iree_register_compiler_plugin_##plugin_id)) \
    return false;
#include "iree/compiler/PluginAPI/Config/StaticLinkedPlugins.inc"
#undef HANDLE_PLUGIN_ID
  return true;
}

bool PluginManager::registerPlugin(const char *pluginId,
                                   PluginRegistrationFunction f) {
  PluginRegistrar registrar(dialectRegistry);
  fprintf(stderr, "Registering plugin %s (%p)\n", pluginId, f);
  return true;
}

}  // namespace mlir::iree_compiler
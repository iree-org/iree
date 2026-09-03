// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_PLUGINAPI_PLUGINENTRYPOINT_H_
#define IREE_COMPILER_PLUGINAPI_PLUGINENTRYPOINT_H_

#include "iree/compiler/PluginAPI/PluginABIHash.h"

namespace mlir::iree_compiler {
class PluginRegistrar;
} // namespace mlir::iree_compiler

extern "C" {

// Bump on any change to IreeCompilerPluginInfo or to what a registration
// function may do. A plugin built against another value is rejected at load
// rather than left to fail on its first virtual call.
#define IREE_COMPILER_PLUGIN_API_VERSION 2

struct IreeCompilerPluginInfo {
  int apiVersion;
  // Hash of the headers the plugin compiled against. The version above covers
  // this struct alone, so nothing else would catch a changed Client.h.
  const char *abiHash;
  // Both owned by the plugin and valid for the life of the process.
  const char *pluginId;
  bool (*registerPlugin)(mlir::iree_compiler::PluginRegistrar *registrar);
};

// A dynamically loaded plugin is found through this one fixed name, so the id
// need not be known before the library is open. Weak because a static link
// holds many plugins and would otherwise see duplicate definitions; nothing
// calls it there.
#define IREE_COMPILER_PLUGIN_INFO_SYMBOL_NAME "iree_get_compiler_plugin_info"

#if defined(_MSC_VER)
#define IREE_COMPILER_PLUGIN_WEAK
#define IREE_COMPILER_PLUGIN_EXPORT __declspec(dllexport)
#else
#define IREE_COMPILER_PLUGIN_WEAK __attribute__((weak))
// A plugin is normally built with hidden visibility, which would leave dlsym
// nothing to find.
#define IREE_COMPILER_PLUGIN_EXPORT __attribute__((visibility("default")))
#endif

} // extern "C"

// Declares both entry points a plugin needs: the per-id one a static link
// calls by name, and the fixed one a dlopen resolves.
//
// Indirect so that a caller may pass a macro as the id: ## does not expand its
// own argument.
#define IREE_DEFINE_COMPILER_PLUGIN(plugin_id, register_fn)                    \
  IREE_DEFINE_COMPILER_PLUGIN_IMPL(plugin_id, register_fn)

#define IREE_DEFINE_COMPILER_PLUGIN_IMPL(plugin_id, register_fn)               \
  extern "C" IREE_COMPILER_PLUGIN_EXPORT bool                                  \
  iree_register_compiler_plugin_##plugin_id(                                   \
      mlir::iree_compiler::PluginRegistrar *registrar) {                       \
    return register_fn(registrar);                                             \
  }                                                                            \
  extern "C" IREE_COMPILER_PLUGIN_EXPORT IREE_COMPILER_PLUGIN_WEAK             \
      IreeCompilerPluginInfo                                                   \
      iree_get_compiler_plugin_info(void) {                                    \
    IreeCompilerPluginInfo info = {                                            \
        IREE_COMPILER_PLUGIN_API_VERSION,                                      \
        IREE_COMPILER_PLUGIN_ABI_HASH,                                         \
        #plugin_id,                                                            \
        &iree_register_compiler_plugin_##plugin_id,                            \
    };                                                                         \
    return info;                                                               \
  }

#endif // IREE_COMPILER_PLUGINAPI_PLUGINENTRYPOINT_H_

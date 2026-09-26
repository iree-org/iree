// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_PLUGINAPI_PLUGINENTRYPOINT_H_
#define IREE_COMPILER_PLUGINAPI_PLUGINENTRYPOINT_H_

// Plain C up to the macro, so a C plugin can include it.
#include <stdbool.h>
#include <stdint.h>

#include "iree/compiler/PluginAPI/PluginABIHash.h"

// Bump on any change to IreeCompilerPluginInfo or to what a registration
// function may do.
#define IREE_COMPILER_PLUGIN_API_VERSION 4

// One fixed name, so the loader needs no id before the library is open.
#define IREE_COMPILER_PLUGIN_INFO_SYMBOL_NAME "iree_get_compiler_plugin_info"

#if defined(_MSC_VER)
#define IREE_COMPILER_PLUGIN_WEAK
#define IREE_COMPILER_PLUGIN_EXPORT __declspec(dllexport)
#else
#define IREE_COMPILER_PLUGIN_WEAK __attribute__((weak))
// Plugins are normally built with hidden visibility, which hides this from
// dlsym.
#define IREE_COMPILER_PLUGIN_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
namespace mlir::iree_compiler {
class PluginRegistrar;
} // namespace mlir::iree_compiler
using IreeCompilerPluginRegistrar = mlir::iree_compiler::PluginRegistrar;
extern "C" {
#else
// Opaque to C; only the compiler dereferences it.
typedef void IreeCompilerPluginRegistrar;
#endif

// Shaped after llvm::PassPluginLibraryInfo and mlir::DialectPluginLibraryInfo.
//
// Returned by pointer: the compiler reads apiVersion first and the rest only
// once it matches, so a struct of another shape is never copied.
//
// The strings are owned by the plugin and live for the process.
typedef struct IreeCompilerPluginInfo {
  uint32_t apiVersion;
  // Hash of the selected IREE plugin API headers, not a full ABI fingerprint.
  // LLVM/MLIR revisions and ABI-affecting build settings must also match.
  const char *abiHash;
  // Unique ID advertised by this library. Use the same ID when registering
  // its session with PluginRegistrar and selecting it with --iree-plugin.
  const char *pluginId;
  // Free-form, only echoed in diagnostics.
  const char *pluginVersion;
  // Registers sessions and returns true on success. The registrar is borrowed
  // only for this call; do not retain its address. Dynamic registrations are
  // committed only if this callback succeeds and their IDs do not collide
  // with existing registrations. Other global side effects cannot be rolled
  // back, so the callback must clean them up on failure. Each ID may be
  // registered only once within a callback.
  bool (*registerPlugin)(IreeCompilerPluginRegistrar *registrar);
} IreeCompilerPluginInfo;

typedef const IreeCompilerPluginInfo *(*IreeCompilerPluginInfoGetter)(void);

#ifdef __cplusplus
} // extern "C"

// The per-id entry point serves a static link, the fixed one a dlopen. The
// fixed one is weak: a static link holds many plugins and calls none by that
// name.
//
// Indirect so the id may be a macro: ## does not expand its argument.
#define IREE_DEFINE_COMPILER_PLUGIN(plugin_id, register_fn, plugin_version)    \
  IREE_DEFINE_COMPILER_PLUGIN_IMPL(plugin_id, register_fn, plugin_version)

#define IREE_DEFINE_COMPILER_PLUGIN_IMPL(plugin_id, register_fn,               \
                                         plugin_version)                       \
  extern "C" IREE_COMPILER_PLUGIN_EXPORT bool                                  \
  iree_register_compiler_plugin_##plugin_id(                                   \
      mlir::iree_compiler::PluginRegistrar *registrar) {                       \
    return register_fn(registrar);                                             \
  }                                                                            \
  extern "C" IREE_COMPILER_PLUGIN_EXPORT                                       \
      IREE_COMPILER_PLUGIN_WEAK const IreeCompilerPluginInfo *                 \
      iree_get_compiler_plugin_info(void) {                                    \
    static const IreeCompilerPluginInfo info = {                               \
        IREE_COMPILER_PLUGIN_API_VERSION,                                      \
        IREE_COMPILER_PLUGIN_ABI_HASH,                                         \
        #plugin_id,                                                            \
        plugin_version,                                                        \
        &iree_register_compiler_plugin_##plugin_id,                            \
    };                                                                         \
    return &info;                                                              \
  }
#endif // __cplusplus

#endif // IREE_COMPILER_PLUGINAPI_PLUGINENTRYPOINT_H_

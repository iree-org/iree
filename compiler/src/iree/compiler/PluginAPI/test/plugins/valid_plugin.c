// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Spells the ABI by hand rather than through IREE_DEFINE_COMPILER_PLUGIN, so
// that a plugin written in plain C stays provably loadable.

#include <stdbool.h>
#include <stdint.h>

// A bare #define, so plain C can take the one value it cannot know.
#include "iree/compiler/PluginAPI/PluginABIHash.h"

struct IreeCompilerPluginInfo {
  uint32_t apiVersion;
  const char *abiHash;
  const char *pluginId;
  const char *pluginVersion;
  bool (*registerPlugin)(void *registrar);
};

static bool register_sample_plugin(void *registrar) {
  (void)registrar;
  return true;
}

struct IreeCompilerPluginInfo iree_get_compiler_plugin_info(void) {
  struct IreeCompilerPluginInfo info = {3, IREE_COMPILER_PLUGIN_ABI_HASH,
                                        "sample_plugin", "test",
                                        register_sample_plugin};
  return info;
}

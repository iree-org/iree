// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// A plugin from a future ABI. It must be refused before its registration
// function is reached.

#include <stdbool.h>

struct IreeCompilerPluginInfo {
  int apiVersion;
  const char *pluginId;
  bool (*registerPlugin)(void *registrar);
};

static bool register_future_plugin(void *registrar) {
  (void)registrar;
  return true;
}

struct IreeCompilerPluginInfo iree_get_compiler_plugin_info(void) {
  struct IreeCompilerPluginInfo info = {999, "from_the_future",
                                        register_future_plugin};
  return info;
}

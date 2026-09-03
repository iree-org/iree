// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// A plugin claiming a header surface this compiler does not have. It must be
// refused before its registration function is reached.

#include <stdbool.h>

struct IreeCompilerPluginInfo {
  int apiVersion;
  const char *abiHash;
  const char *pluginId;
  bool (*registerPlugin)(void *registrar);
};

static bool register_other_headers_plugin(void *registrar) {
  (void)registrar;
  return true;
}

struct IreeCompilerPluginInfo iree_get_compiler_plugin_info(void) {
  struct IreeCompilerPluginInfo info = {2, "0000000000000000",
                                        "from_other_headers",
                                        register_other_headers_plugin};
  return info;
}

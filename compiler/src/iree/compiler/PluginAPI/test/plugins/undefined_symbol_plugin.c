// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Calls a function nothing defines. Lazy binding would let this load and fail
// mid-compilation instead.

#include <stdbool.h>
#include <stdint.h>

struct IreeCompilerPluginInfo {
  uint32_t apiVersion;
  const char *abiHash;
  const char *pluginId;
  const char *pluginVersion;
  bool (*registerPlugin)(void *registrar);
};

extern bool iree_test_absent_function(void);

static bool register_undefined_plugin(void *registrar) {
  (void)registrar;
  return iree_test_absent_function();
}

struct IreeCompilerPluginInfo iree_get_compiler_plugin_info(void) {
  struct IreeCompilerPluginInfo info = {3, "", "undefined_plugin", "test",
                                        register_undefined_plugin};
  return info;
}

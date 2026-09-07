// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// A plugin from a future ABI: one field longer than the compiler's struct. It
// must be refused on apiVersion alone, before anything else is read.

#include <stdbool.h>
#include <stdint.h>

struct FutureIreeCompilerPluginInfo {
  uint32_t apiVersion;
  const char *abiHash;
  const char *pluginId;
  const char *pluginVersion;
  bool (*registerPlugin)(void *registrar);
  const char *addedInTheFuture;
};

static bool register_future_plugin(void *registrar) {
  (void)registrar;
  return true;
}

static const struct FutureIreeCompilerPluginInfo info = {
    999,    "ffffffffffffffff",     "from_the_future",
    "test", register_future_plugin, "extra"};

const struct FutureIreeCompilerPluginInfo *iree_get_compiler_plugin_info(void) {
  return &info;
}

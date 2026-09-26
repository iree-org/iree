// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Right version, null hash. Must be refused before the hash is read.

#include <stddef.h>

#include "iree/compiler/PluginAPI/PluginEntryPoint.h"

static bool register_null_field_plugin(IreeCompilerPluginRegistrar *registrar) {
  (void)registrar;
  return true;
}

static const IreeCompilerPluginInfo info = {IREE_COMPILER_PLUGIN_API_VERSION,
                                            NULL, "null_field", "test",
                                            register_null_field_plugin};

IREE_COMPILER_PLUGIN_EXPORT const IreeCompilerPluginInfo *
iree_get_compiler_plugin_info(void) {
  return &info;
}

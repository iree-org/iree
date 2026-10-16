// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Claims other headers. Must be refused before registration.

#include "iree/compiler/PluginAPI/PluginEntryPoint.h"

static bool
register_other_headers_plugin(IreeCompilerPluginRegistrar *registrar) {
  (void)registrar;
  return true;
}

static const IreeCompilerPluginInfo info = {
    UINT64_C(0), "from_other_headers", "test", register_other_headers_plugin};

IREE_COMPILER_PLUGIN_EXPORT const IreeCompilerPluginInfo *
iree_get_compiler_plugin_info(void) {
  return &info;
}

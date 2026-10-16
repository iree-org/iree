// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// An incompatible ABI with only a compatibility ID. The loader must reject it
// before reading fields from the current layout.

#include <stdint.h>

struct IncompatibleIreeCompilerPluginInfo {
  uint64_t abiHash;
};

static const struct IncompatibleIreeCompilerPluginInfo info = {UINT64_C(0)};

const struct IncompatibleIreeCompilerPluginInfo *
iree_get_compiler_plugin_info(void) {
  return &info;
}

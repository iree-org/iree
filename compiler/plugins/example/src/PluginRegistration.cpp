// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/PluginAPI/Registration.h"

extern "C" {
bool iree_register_compiler_plugin_example(
    mlir::iree_compiler::PluginRegistrar *registrar) {
  return false;
}
}
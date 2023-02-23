// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Tools/iree_compile_lib.h"
#include "iree/compiler/tool_entry_points_api.h"

int ireeCompilerRunMain(int argc, char **argv) {
  // TODO: Inline the actual runIreecMain here as the single place to use it.
  return mlir::iree_compiler::runIreecMain(argc, argv);
}

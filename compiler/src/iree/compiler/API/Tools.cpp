// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/API/Tools.h"

#include "iree/compiler/Tools/iree_compile_lib.h"

int ireeCompilerRunMain(int argc, char **argv) {
  return mlir::iree_compiler::runIreecMain(argc, argv);
}

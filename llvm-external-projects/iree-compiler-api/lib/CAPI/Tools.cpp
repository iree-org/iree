// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-compiler-c/Tools.h"

#include "iree/tools/iree_translate_lib.h"

int ireeCompilerRunMain(int argc, char **argv) {
  return mlir::iree_compiler::runIreeTranslateMain(argc, argv);
}

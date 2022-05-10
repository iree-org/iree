// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_TOOLS_IREE_TRANSLATE_LIB_H
#define IREE_COMPILER_TOOLS_IREE_TRANSLATE_LIB_H

namespace mlir {
namespace iree_compiler {

int runIreeTranslateMain(int argc, char **argv);

// NOTE: We are transitioning from the main compiler being based on
// the MLIR translation library (i.e. iree-translate) to a dedicated tool
// called iree-compile. When this is done, the above should go away and this
// file should be renamed to iree_compile_lib.h.
int runIreecMain(int argc, char **argv);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_TOOLS_IREE_TRANSLATE_LIB_H

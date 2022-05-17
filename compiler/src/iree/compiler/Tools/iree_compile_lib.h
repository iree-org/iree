// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_TOOLS_IREE_COMPILE_LIB_H
#define IREE_COMPILER_TOOLS_IREE_COMPILE_LIB_H

namespace mlir {
namespace iree_compiler {

int runIreecMain(int argc, char **argv);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_TOOLS_IREE_COMPILE_LIB_H
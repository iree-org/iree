// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_TOOLS_VERSION_H
#define IREE_COMPILER_TOOLS_VERSION_H

#include <string>

namespace mlir::iree_compiler {

// Returns the IREE compiler version. Empty if built without IREE_VERSION
// defined.
std::string getIreeRevision();

}  // namespace mlir::iree_compiler

#endif  // IREE_COMPILER_TOOLS_VERSION_H

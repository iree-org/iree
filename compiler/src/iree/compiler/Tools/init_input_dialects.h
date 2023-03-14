// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This files defines a helper to trigger the registration of dialects to
// the system.
//
// Based on MLIR's InitAllDialects but for IREE input dialects.

#ifndef IREE_COMPILER_TOOLS_INIT_INPUT_DIALECTS_H_
#define IREE_COMPILER_TOOLS_INIT_INPUT_DIALECTS_H_

#include "mlir/IR/Dialect.h"

namespace mlir {
namespace iree_compiler {

void registerInputDialects(DialectRegistry &registry);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_TOOLS_INIT_INPUT_DIALECTS_H_

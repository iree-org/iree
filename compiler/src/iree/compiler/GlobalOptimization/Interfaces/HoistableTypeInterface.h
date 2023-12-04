// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_GLOBALOPTIMIZATION_INTERFACES_HOISTABLETYPEINTERFACE_H_
#define IREE_COMPILER_GLOBALOPTIMIZATION_INTERFACES_HOISTABLETYPEINTERFACE_H_

#include "mlir/IR/Dialect.h"

namespace mlir {
namespace iree_compiler {

// Register all interfaces needed for hoisting constant expressions.
void registerHoistableTypeInterfaces(DialectRegistry &registry);

} // namespace iree_compiler
} // namespace mlir

#endif // IREE_COMPILER_GLOBALOPTIMIZATION_INTERFACES_HOISTABLETYPEINTERFACE_H_

// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_GLOBALOPTIMIZATION_INTERFACES_INTERFACES_H_
#define IREE_COMPILER_GLOBALOPTIMIZATION_INTERFACES_INTERFACES_H_

#include "mlir/IR/Dialect.h"

namespace mlir::iree_compiler {

/// Register all global optimization related interfaces.
void registerGlobalOptimizationInterfaces(DialectRegistry &registry);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_GLOBALOPTIMIZATION_INTERFACES_INTERFACES_H_

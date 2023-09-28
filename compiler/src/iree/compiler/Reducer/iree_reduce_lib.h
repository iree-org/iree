// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_TOOLS_IREE_REDUCER_LIB_H
#define IREE_COMPILER_TOOLS_IREE_REDUCER_LIB_H

#include "mlir/IR/OwningOpRef.h"
#include "mlir/Support/LLVM.h"

namespace mlir {

class Operation;

namespace iree_compiler::Reducer {

Operation *ireeRunReducingStrategies(OwningOpRef<Operation *> module,
                                     StringRef testScript);

} // namespace iree_compiler::Reducer
} // namespace mlir

#endif // IREE_COMPILER_TOOLS_IREE_REDUCER_LIB_H

// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_GLOBALOPTIMIZATION_PASSDETAIL_H_
#define IREE_COMPILER_GLOBALOPTIMIZATION_PASSDETAIL_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace GlobalOptimization {

#define GEN_PASS_CLASSES
#include "iree/compiler/GlobalOptimization/Passes.h.inc"

} // namespace GlobalOptimization
} // namespace iree_compiler
} // namespace mlir

#endif // IREE_COMPILER_GLOBALOPTIMIZATION_PASSDETAIL_H_

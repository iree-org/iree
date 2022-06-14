// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CONSTEVAL_PASSES_H_
#define IREE_COMPILER_CONSTEVAL_PASSES_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace ConstEval {

/// Creates a pass which uses the compiler and runtime to Jit global
/// initializers eligible for optimization and uses the actual results to
/// simplify the globals in the module.
std::unique_ptr<OperationPass<ModuleOp>> createJitGlobalsPass();

void registerConstEvalPasses();

}  // namespace ConstEval
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CONSTEVAL_PASSES_H_

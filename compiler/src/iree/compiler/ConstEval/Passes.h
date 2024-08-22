// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CONSTEVAL_PASSES_H_
#define IREE_COMPILER_CONSTEVAL_PASSES_H_

#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::ConstEval {

#define GEN_PASS_DECL
#include "iree/compiler/ConstEval/Passes.h.inc"

void registerConstEvalPasses();

} // namespace mlir::iree_compiler::ConstEval

#endif // IREE_COMPILER_CONSTEVAL_PASSES_H_

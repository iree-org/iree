// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_JITEVAL_PASSES
#define IREE_COMPILER_JITEVAL_PASSES

include "mlir/Pass/PassBase.td"

def JitGlobals :
  Pass<"iree-consteval-jit-globals", "ModuleOp"> {
  let summary = "Jits global initializers and evaluates them into concrete values";
  let constructor = "mlir::iree_compiler::ConstEval::createJitGlobalsPass()";
  let options = [
    Option<
      "targetRegistry", "target-registry",
      "llvm::cl::TargetRegistryRef", "",
      "Target backend registry containing the list of available backends."
    >,
  ];
}

#endif // IREE_COMPILER_JITEVAL_PASSES

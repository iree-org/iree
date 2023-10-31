// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_INPUTCONVERSION_TOSA_PASSES
#define IREE_COMPILER_INPUTCONVERSION_TOSA_PASSES

include "mlir/Pass/PassBase.td"

def TosaToLinalgExt :
    Pass<"iree-tosa-to-linalg-ext", "mlir::func::FuncOp"> {
  let summary = "Convert TOSA operations to their equivalent linalg-ext operations.";
  let constructor = "mlir::iree_compiler::createTosaToLinalgExt()";
  let dependentDialects = [
    "arith::ArithDialect",
    "linalg::LinalgDialect",
    "tensor::TensorDialect",
    "IREE::LinalgExt::IREELinalgExtDialect",
  ];
}

def VerifyCompilerTOSAInputLegality :
    Pass<"iree-tosa-verify-compiler-input-legality", "ModuleOp"> {
  let summary = "Verifies that only supported IR constructs are passed to the compiler.";
  let constructor = "mlir::iree_compiler::createVerifyCompilerTOSAInputLegality()";
}

def Converti48Toi64 :
    Pass<"iree-convert-i48-to-i64", "mlir::func::FuncOp"> {
  let summary = "Converts all i48s to i64s";
  let constructor = "mlir::iree_compiler::createConverti48Toi64()";
}

#endif // IREE_COMPILER_INPUTCONVERSION_TOSA_PASSES

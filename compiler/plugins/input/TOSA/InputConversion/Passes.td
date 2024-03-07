// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_PLUGINS_INPUT_TOSA_INPUTCONVERSION_PASSES
#define IREE_COMPILER_PLUGINS_INPUT_TOSA_INPUTCONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def Converti48Toi64 :
    InterfacePass<"iree-tosa-convert-i48-to-i64", "mlir::FunctionOpInterface"> {
  let summary = "Converts all i48s to i64s";
  let constructor = "mlir::iree_compiler::createConverti48Toi64()";
}

def StripSignedness :
    InterfacePass<"iree-tosa-strip-signedness", "mlir::FunctionOpInterface"> {
  let summary = "Legalizes ui tensors constants to uis";
  let constructor = "mlir::iree_compiler::createStripSignednessPass()";
}

def TosaToLinalgExt :
    InterfacePass<"iree-tosa-to-linalg-ext", "mlir::FunctionOpInterface"> {
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

#endif // IREE_COMPILER_PLUGINS_INPUT_TOSA_INPUTCONVERSION_PASSES

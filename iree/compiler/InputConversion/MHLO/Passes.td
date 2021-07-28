// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_INPUTCONVERSION_MHLO_PASSES
#define IREE_COMPILER_INPUTCONVERSION_MHLO_PASSES

include "mlir/Pass/PassBase.td"

def ConvertMHLOToLinalgOnTensors :
    Pass<"iree-mhlo-to-linalg-on-tensors", "FuncOp"> {
  let summary = "Convert from XLA-HLO ops to Linalg ops on tensors";
  let constructor = "mlir::iree_compiler::createMHLOToLinalgOnTensorsPass()";
}

def ConvertMHLOToLinalgExt
    : Pass<"iree-mhlo-to-linalg-ext", "FuncOp"> {
  let summary =
      "Convert from XLA-HLO ops to LinalgExt ops and distribute to Flow ops";
  let constructor =
      "mlir::iree_compiler::createConvertMHLOToLinalgExtPass()";
}

def LegalizeInputTypes :
    Pass<"iree-mhlo-legalize-input-types", "ModuleOp"> {
  let summary = "Legalizes input types to ones supported by the IREE flow dialect";
  let constructor = "mlir::iree_compiler::createLegalizeInputTypesPass()";
}

def MHLOToMHLOPreprocessing :
    Pass<"iree-mhlo-to-mhlo-preprocessing", "FuncOp"> {
  let summary = "Apply mhlo to mhlo transformations for some mhlo ops";
  let constructor = "mlir::iree_compiler::createMHLOToMHLOPreprocessingPass()";
  let options = [
    Option<"extractPadFromConv", "extract-pad-from-conv", "bool", /*default=*/"true",
           "Extract padding attributes from conv op">,
    Option<"orderConvFeatures", "order-conv-features", "bool", /*default=*/"true",
           "Guarantees input/output features ordered from conv kernel">
  ];
}

def VerifyCompilerMHLOInputLegality :
    Pass<"iree-mhlo-verify-compiler-input-legality", "ModuleOp"> {
  let summary = "Verifies that only supported IR constructs are passed to the compiler.";
  let constructor = "mlir::iree_compiler::createVerifyCompilerMHLOInputLegality()";
}

//------------------------------------------------------------------------------
// Test passes
//------------------------------------------------------------------------------

def TestMHLOConvertComplexToReal :
    Pass<"iree-test-mhlo-convert-complex-to-real", "FuncOp"> {
  let summary = "Test pass that does an MHLO->MHLO conversion of just complex arithmetic ops.";
  let constructor = "mlir::iree_compiler::createTestMHLOConvertComplexToRealPass()";
}

#endif // IREE_COMPILER_INPUTCONVERSION_MHLO_PASSES

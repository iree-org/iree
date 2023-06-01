// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_INPUTCONVERSION_MHLO_PASSES
#define IREE_COMPILER_INPUTCONVERSION_MHLO_PASSES

include "mlir/Pass/PassBase.td"

def ConvertMHLOToLinalgOnTensors :
    Pass<"iree-mhlo-to-linalg-on-tensors", "ModuleOp"> {
  let summary = "Convert from XLA-HLO ops to Linalg ops on tensors";
  let constructor = "mlir::iree_compiler::MHLO::createMHLOToLinalgOnTensorsPass()";
}

def ConvertMHLOToLinalgExt
    : Pass<"iree-mhlo-to-linalg-ext", "func::FuncOp"> {
  let summary =
      "Convert from XLA-HLO ops to LinalgExt ops and distribute to Flow ops";
  let constructor =
      "mlir::iree_compiler::MHLO::createConvertMHLOToLinalgExtPass()";
}

def FlattenTuplesInCFG :
    Pass<"iree-mhlo-flatten-tuples-in-cfg", "ModuleOp"> {
  let summary = "Flattens tuples in a CFG form of MHLO";
  let constructor = "mlir::iree_compiler::MHLO::createFlattenTuplesInCFGPass()";
}

def MHLOToMHLOPreprocessing :
    Pass<"iree-mhlo-to-mhlo-preprocessing", "func::FuncOp"> {
  let summary = "Apply mhlo to mhlo transformations for some mhlo ops";
  let constructor = "mlir::iree_compiler::MHLO::createMHLOToMHLOPreprocessingPass()";
  let options = [
    Option<"orderConvFeatures", "order-conv-features", "bool", /*default=*/"true",
           "Guarantees input/output features ordered from conv kernel">
  ];
}

def VerifyCompilerMHLOInputLegality :
    Pass<"iree-mhlo-verify-compiler-input-legality", "ModuleOp"> {
  let summary = "Verifies that only supported IR constructs are passed to the compiler.";
  let constructor = "mlir::iree_compiler::MHLO::createVerifyCompilerMHLOInputLegality()";
}

def ConvertMHLOToStableHLOPass : Pass<"iree-convert-mhlo-to-stablehlo", "ModuleOp"> {
  let summary = "Convert MHLO to StableHLO, to aid in transition to StableHLO.";
  let constructor = "mlir::iree_compiler::MHLO::createConvertMHLOToStableHLOPass()";
}

//------------------------------------------------------------------------------
// Test passes
//------------------------------------------------------------------------------

def TestMHLOConvertComplexToReal :
    Pass<"iree-test-mhlo-convert-complex-to-real", "func::FuncOp"> {
  let summary = "Test pass that does an MHLO->MHLO conversion of just complex arithmetic ops.";
  let constructor = "mlir::iree_compiler::MHLO::createTestMHLOConvertComplexToRealPass()";
}

#endif // IREE_COMPILER_INPUTCONVERSION_MHLO_PASSES

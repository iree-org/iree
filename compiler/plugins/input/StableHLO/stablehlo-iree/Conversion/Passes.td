// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef STABLEHLO_IREE_CONVERSION_PASSES
#define STABLEHLO_IREE_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// IREE-specific passes
//===----------------------------------------------------------------------===//

def ConvertStableHloToIreeInputDialects :
    Pass<"iree-stablehlo-to-iree-input", "ModuleOp"> {
  let summary = "Convert StableHLO ops to IREE input dialects ops";
  let description = [{
    Converts StableHLO input to IREE input dialects. These include: Linalg,
    Arith, Tensor, Complex, Shape, ML Program, Flow, and Util.
  }];
}

def ConvertStableHloToLinalgExt :
    InterfacePass<"iree-stablehlo-to-linalg-ext", "mlir::FunctionOpInterface"> {
  let summary =
      "Converts from StableHLO ops to LinalgExt ops and distribute to Flow ops";
}

//===----------------------------------------------------------------------===//
// General passes
//===----------------------------------------------------------------------===//

def ConvertStableHloToLinalg :
    Pass<"iree-stablehlo-to-linalg", "ModuleOp"> {
  let summary = "Converts from StableHLO ops to Linalg ops on";
  let options = [Option<"enablePrimitiveOps", "enable-primitive-ops", "bool",
                        /*default=*/"false",
                        "Lower to primitive Linalg ops (map, reduce and "
                        "transpose) when possible, instead of linalg.generic">];
}

def LegalizeControlFlow :
    InterfacePass<"iree-stablehlo-legalize-control-flow", "mlir::FunctionOpInterface"> {
  let summary = "Legalizes from StableHLO control flow to SCF control flow";
}

def LegalizeChlo :
    InterfacePass<"iree-stablehlo-legalize-chlo", "mlir::FunctionOpInterface"> {
  let summary = "Legalizes from CHLO ops flow to StableHLO and Shape ops";
}

def LegalizeStableHLOCustomCalls :
    InterfacePass<"iree-stablehlo-legalize-custom-calls", "mlir::FunctionOpInterface"> {
  let summary = "Legalizes specialized custom calls to decomposed implementations";
}

def LegalizeShapeComputations :
    InterfacePass<"iree-stablehlo-legalize-shape-computations", "mlir::FunctionOpInterface"> {
  let summary = "Legalizes StableHLO shape operations to core-mlir operations";
}

def VerifyCompilerStableHloInputLegality :
    Pass<"iree-stablehlo-verify-compiler-input-legality", "ModuleOp"> {
  let summary =
      "Verifies that only supported IR constructs are passed to the compiler";
}

#endif // STABLEHLO_IREE_CONVERSION_PASSES

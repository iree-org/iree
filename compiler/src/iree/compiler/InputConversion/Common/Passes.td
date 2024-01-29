// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_INPUTCONVERSION_COMMON_PASSES
#define IREE_COMPILER_INPUTCONVERSION_COMMON_PASSES

include "mlir/Pass/PassBase.td"

def IREEImportPublic :
    Pass<"iree-import-public", "ModuleOp"> {
  let summary = "Imports IREE public dialect to internal implementation.";
  let constructor = "mlir::iree_compiler::createIREEImportPublicPass()";
}

def ImportMLProgram :
    Pass<"iree-import-ml-program", "ModuleOp"> {
  let summary = "Imports MLProgram dialect to IREE Equivalents.";
  let constructor = "mlir::iree_compiler::createImportMLProgramPass()";
}

def LinalgQuantizedConvToConvPass
    : InterfacePass<"iree-linalg-quantized-conv-to-conv", "mlir::FunctionOpInterface"> {
  let summary = "lower quantized_conv to conv";
  let constructor = "mlir::iree_compiler::createLinalgQuantizedConvToConvPass()";
}

def LinalgQuantizedMatmulToMatmulPass
    : InterfacePass<"iree-linalg-quantized-matmul-to-matmul", "mlir::FunctionOpInterface"> {
  let summary = "lower quantized_matmul to matmul";
  let constructor = "mlir::iree_compiler::createLinalgQuantizedMatmulToMatmulPass()";
}

def SanitizeModuleNames :
    Pass<"iree-sanitize-module-names", "ModuleOp"> {
  let summary = "Sanitizes module names for uniformity across target implementations.";
  let constructor = "mlir::iree_compiler::createSanitizeModuleNamesPass()";
}

def AutoInputConversionPipeline :
    Pass<"iree-auto-input-conversion", "ModuleOp"> {
  let summary = "Analyzes and runs appropriate input pipeline.";
  let description = [{
    Analyzes the input program to determine which input pipeline, or part of input
    conversion to run, then run that conversion.
  }];
  let constructor = "mlir::iree_compiler::createAutoInputConversionPipelinePass()";
}

#endif // IREE_COMPILER_INPUTCONVERSION_COMMON_PASSES

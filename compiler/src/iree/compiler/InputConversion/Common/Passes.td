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
  let constructor = "mlir::iree_compiler::InputConversion::createIREEImportPublicPass()";
}

def ImportMLProgram :
    Pass<"iree-import-ml-program", "ModuleOp"> {
  let summary = "Imports MLProgram dialect to IREE Equivalents.";
  let constructor = "mlir::iree_compiler::InputConversion::createImportMLProgramPass()";
}

def LinalgQuantizedConvToConvPass
    : InterfacePass<"iree-linalg-quantized-conv-to-conv", "mlir::FunctionOpInterface"> {
  let summary = "lower quantized_conv to conv";
  let constructor = "mlir::iree_compiler::InputConversion::createLinalgQuantizedConvToConvPass()";
}

def LinalgQuantizedMatmulToMatmulPass
    : InterfacePass<"iree-linalg-quantized-matmul-to-matmul", "mlir::FunctionOpInterface"> {
  let summary = "lower quantized_matmul to matmul";
  let constructor = "mlir::iree_compiler::InputConversion::createLinalgQuantizedMatmulToMatmulPass()";
}

def TensorOpsToLinalgPass
    : InterfacePass<"iree-tensor-ops-to-linalg", "mlir::FunctionOpInterface"> {
  let summary = "Convert tensor ops to linalg ops";
  let constructor = "mlir::iree_compiler::InputConversion::createTensorOpsToLinalgPass()";
}

def SanitizeModuleNames :
    Pass<"iree-sanitize-module-names", "ModuleOp"> {
  let summary = "Sanitizes module names for uniformity across target implementations.";
  let constructor = "mlir::iree_compiler::InputConversion::createSanitizeModuleNamesPass()";
}

def AutoInputConversionPipeline :
    Pass<"iree-auto-input-conversion", "ModuleOp"> {
  let summary = "Analyzes and runs appropriate input pipeline.";
  let description = [{
    Analyzes the input program to determine which input pipeline, or part of input
    conversion to run, then run that conversion.
  }];
  let constructor = "mlir::iree_compiler::InputConversion::createAutoInputConversionPipelinePass()";
}

//===----------------------------------------------------------------------===//
// Type Conversion
//===----------------------------------------------------------------------===//

def DemoteI64ToI32 : Pass<"iree-input-conversion-demote-i64-to-i32", "mlir::ModuleOp"> {
  let summary = "Demotes i64 types to i32 types.";
  let constructor = [{
    mlir::iree_compiler::InputConversion::createDemoteI64ToI32Pass()
  }];
}

def DemoteF32ToF16 : Pass<"iree-input-conversion-demote-f32-to-f16", "mlir::ModuleOp"> {
  let summary = "Demotes f32 types to f16 types.";
  let constructor = [{
    mlir::iree_compiler::InputConversion::createDemoteF32ToF16Pass()
  }];
}

def DemoteF64ToF32 : Pass<"iree-input-conversion-demote-f64-to-f32", "mlir::ModuleOp"> {
  let summary = "Demotes f64 types to f32 types.";
  let constructor = [{
    mlir::iree_compiler::InputConversion::createDemoteF64ToF32Pass()
  }];
}

def PromoteF16ToF32 : Pass<"iree-input-conversion-promote-f16-to-f32", "mlir::ModuleOp"> {
  let summary = "Promotes f16 types to f32 types.";
  let constructor = [{
    mlir::iree_compiler::InputConversion::createPromoteF16ToF32Pass()
  }];
}

def PromoteBF16ToF32 : Pass<"iree-input-conversion-promote-bf16-to-f32", "mlir::ModuleOp"> {
  let summary = "Promotes bf16 types to f32 types.";
  let constructor = [{
    mlir::iree_compiler::InputConversion::createPromoteBF16ToF32Pass()
  }];
}

#endif // IREE_COMPILER_INPUTCONVERSION_COMMON_PASSES

// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_INPUTCONVERSION_COMMON_PASSES
#define IREE_COMPILER_INPUTCONVERSION_COMMON_PASSES

include "mlir/Pass/PassBase.td"

def IREEImportPublicPass :
    Pass<"iree-import-public", "ModuleOp"> {
  let summary = "Imports IREE public dialect to internal implementation.";
}

def ImportMLProgramPass :
    Pass<"iree-import-ml-program", "ModuleOp"> {
  let summary = "Imports MLProgram dialect to IREE Equivalents.";
}

def SanitizeModuleNamesPass :
    Pass<"iree-sanitize-module-names", "ModuleOp"> {
  let summary = "Sanitizes module names for uniformity across target implementations.";
}

def AutoInputConversionPipelinePass :
    Pass<"iree-auto-input-conversion", "ModuleOp"> {
  let summary = "Analyzes and runs appropriate input pipeline.";
  let description = [{
    Analyzes the input program to determine which input pipeline, or part of input
    conversion to run, then run that conversion.
  }];
}

//===----------------------------------------------------------------------===//
// Type Conversion
//===----------------------------------------------------------------------===//

def DemoteI64ToI32Pass : Pass<"iree-input-conversion-demote-i64-to-i32", "mlir::ModuleOp"> {
  let summary = "Demotes i64 types to i32 types.";
}

def DemoteF32ToF16Pass : Pass<"iree-input-conversion-demote-f32-to-f16", "mlir::ModuleOp"> {
  let summary = "Demotes f32 types to f16 types.";
}

def DemoteF64ToF32Pass : Pass<"iree-input-conversion-demote-f64-to-f32", "mlir::ModuleOp"> {
  let summary = "Demotes f64 types to f32 types.";
}

def PromoteF16ToF32Pass : Pass<"iree-input-conversion-promote-f16-to-f32", "mlir::ModuleOp"> {
  let summary = "Promotes f16 types to f32 types.";
}

def PromoteBF16ToF32Pass : Pass<"iree-input-conversion-promote-bf16-to-f32", "mlir::ModuleOp"> {
  let summary = "Promotes bf16 types to f32 types.";
}

#endif // IREE_COMPILER_INPUTCONVERSION_COMMON_PASSES

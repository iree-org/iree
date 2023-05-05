// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_INTEGRATIONS_TFL_PASSES
#define IREE_INTEGRATIONS_TFL_PASSES

include "mlir/Pass/PassBase.td"

def ConvertFunctionMetadata :
    Pass<"iree-tflite-convert-function-metadata", "mlir::func::FuncOp"> {
  let summary = "Converts TFLite attributes to IREE attributes on functions.";
  let constructor = "mlir::iree_integrations::TFL::createConvertFunctionMetadataPass()";
}

def ConvertModuleMetadata :
    Pass<"iree-tflite-convert-module-metadata", "mlir::ModuleOp"> {
  let summary = "Converts TFLite attributes to IREE attributes on modules.";
  let constructor = "mlir::iree_integrations::TFL::createConvertModuleMetadataPass()";
}

def LowerGlobalTensors :
    Pass<"iree-tflite-lower-global-tensors", "mlir::ModuleOp"> {
  let summary = "Lowers tflite global tensors to IREE flow dialect variables.";
  let constructor = "mlir::iree_integrations::TFL::createLowerGlobalTensorsPass()";
}

def RetainCallOnceFuncs :
    Pass<"iree-tflite-retain-call-once-funcs", "mlir::ModuleOp"> {
  let summary = "Guarantees that functions used by tfl.call_once are retained.";
  let constructor = "mlir::iree_integrations::TFL::createRetainCallOnceFuncsPass()";
}

def StripFunctionMetadata :
    Pass<"iree-tflite-strip-function-metadata", "mlir::func::FuncOp"> {
  let summary = "Guarantees that functions used by tfl.call_once are retained.";
  let constructor = "mlir::iree_integrations::TFL::createStripFunctionMetadataPass()";
}

def StripModuleMetadata :
    Pass<"iree-tflite-strip-module-metadata", "mlir::ModuleOp"> {
  let summary = "Guarantees that functions used by tfl.call_once are retained.";
  let constructor = "mlir::iree_integrations::TFL::createStripModuleMetadataPass()";
}

def VerifyFullyConverted :
    Pass<"iree-tflite-verify-fully-converted", "mlir::func::FuncOp"> {
  let summary = "Verifies that all TFLite frontend ops were converted and none remain.";
  let constructor = "mlir::iree_integrations::TFL::createVerifyFullyConvertedPass()";
}

#endif  // IREE_INTEGRATIONS_TFL_PASSES

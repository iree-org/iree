// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BINDINGS_TFLITE_TRANSFORMS_PASSES
#define IREE_BINDINGS_TFLITE_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// IREE TFLite C bindings support
//===----------------------------------------------------------------------===//

def WrapEntryPointsPass :
    Pass<"iree-tflite-wrap-entry-points", "mlir::ModuleOp"> {
  let summary = "Wraps entry points in functions compatible with the IREE TFLite C bindings.";
  let description = [{
    Wraps each model entry point in a "_tflite_xx" function that matches the
    expectations of the IREE TFLite C bindings and materializes shape query and
    calculation functions for dynamically shaped I/O.

    For each exported function the pass produces:
    * `_tflite_xx_argN`/`retN` globals carrying shape dimensions
    * `_tflite_xx` entry function wrapping the existing export
    * `_tflite_xx_calculate_shapes` shape calculation function
    * `_tflite_xx_query_input_shape` shape query function
    * `_tflite_xx_query_output_shape` shape query function

    Each I/O of the function gets one global per dynamic dimension storing the
    provided or calculated dimension value at runtime.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::tensor::TensorDialect",
    "IREE::HAL::HALDialect",
    "IREE::Util::UtilDialect",
  ];
}

#endif // IREE_BINDINGS_TFLITE_TRANSFORMS_PASSES

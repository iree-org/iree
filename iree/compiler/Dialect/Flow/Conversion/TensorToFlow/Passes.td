// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_FLOW_CONVERSION_TENSORTOFLOW_PASSES
#define IREE_DIALECT_FLOW_CONVERSION_TENSORTOFLOW_PASSES

include "mlir/Pass/PassBase.td"

def ConvertTensorToFlow :
    Pass<"iree-tensor-to-flow", "FuncOp"> {
  let summary = "Converts ops from tensor to flow dialects";
  let constructor = "mlir::iree_compiler::IREE::Flow::createConvertTensorToFlow()";
}

#endif  // IREE_DIALECT_FLOW_CONVERSION_TENSORTOFLOW_PASSES

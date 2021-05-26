
// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CONVERSION_PASSES
#define IREE_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertHLOToLinalgOnTensors :
    Pass<"iree-codegen-hlo-to-linalg-on-tensors", "FuncOp"> {
  let summary = "Convert from XLA-HLO ops to Linalg ops on tensors";
  let constructor = "mlir::iree_compiler::createHLOToLinalgOnTensorsPass()";
}

def DemoteF32ToF16 :
    Pass<"iree-convert-f32-to-f16", "ModuleOp"> {
  let summary = "Convert f32 operations and values into equivalent f16 ones.";
  let constructor = "mlir::iree_compiler::createDemoteF32ToF16Pass()";
}

def FusionOfTensorOps :
    Pass<"iree-codegen-fusion-of-tensor-ops", ""> {
  let summary = "Fuse operations on tensors";
  let constructor = "mlir::iree_compiler::createFusionOfTensorOpsPass()";
}

def ResolveShapeOps :
    Pass<"iree-codegen-resolve-shape", "FuncOp"> {
  let summary = "resolve shapes";
  let constructor = "mlir::iree_compiler::createResolveShapeOpsPass()";
}

#endif  // IREE_DIALECT_FLOW_PASSES

// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_GLOBALOPTIMIZATION_PASSES
#define IREE_COMPILER_GLOBALOPTIMIZATION_PASSES

include "mlir/Pass/PassBase.td"

def Convert1X1FilterConv2DToMatmul:
    Pass<"iree-global-opt-convert-1x1-filter-conv2d-to-matmul", ""> {
  let summary = "Convert linalg convolution ops with 1x1 kernels into linalg matrix multiplication ops.";
  let constructor = "mlir::iree_compiler::GlobalOptimization::createConvert1X1FilterConv2DToMatmulPass()";
}

def DetachElementwiseFromNamedOps :
    Pass<"iree-global-opt-detach-elementwise-from-named-ops", ""> {
  let summary = "Detaches elementwise ops from named Linalg ops";
  let constructor = "mlir::iree_compiler::GlobalOptimization::createDetachElementwiseFromNamedOpsPass()";
}

def EraseUnusedLinalgOperands :
    Pass<"iree-global-opt-erase-unused-linalg-operands", "mlir::ModuleOp"> {
  let summary = "Erase unused linalg operand and remove dead code.";
  let constructor = "mlir::iree_compiler::GlobalOptimization::createEraseUnusedLinalgOperands()";
}

def ExpandVectors : Pass<"iree-global-opt-expand-vectors", ""> {
  let summary = "Expand vector in vector/matrix operations";
  let constructor = "mlir::iree_compiler::GlobalOptimization::createExpandVectorsPass()";
}

def MaterializeHomogeneousEncodings :
  Pass<"iree-global-opt-materialize-homogeneous-encodings", "mlir::ModuleOp"> {
  let summary = "Materializes logical encodings to physical encodings if there is a single device target.";
  let constructor =
  "mlir::iree_compiler::GlobalOptimization::createMaterializeHomogeneousEncodingsPass()";
}

def RemoveZeroExtentTensors :
    InterfacePass<"iree-global-opt-remove-zero-extent-tensors", "mlir::FunctionOpInterface"> {
  let summary = "Remove tensors that have 0-extents";
  let constructor = "mlir::iree_compiler::GlobalOptimization::createRemoveZeroExtentTensorsPass()";
}

def SetEncoding : Pass<"iree-global-opt-set-encoding", ""> {
  let summary = "Introduce tensor encoding for compute operations";
  let constructor = "mlir::iree_compiler::GlobalOptimization::createSetEncodingPass()";
}

#endif // IREE_COMPILER_GLOBALOPTIMIZATION_PASSES

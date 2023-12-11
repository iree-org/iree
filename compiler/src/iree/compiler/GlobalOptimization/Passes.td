// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_GLOBALOPTIMIZATION_PASSES
#define IREE_COMPILER_GLOBALOPTIMIZATION_PASSES

include "mlir/Pass/PassBase.td"

def CleanupNumericNarrowing :
    Pass<"iree-global-opt-cleanup-numeric-narrowing", ""> {
  let summary = "Cleans up any numeric narrowing ops inserted by iree-global-opt-infer-numeric-narrowing";
  let constructor = "mlir::iree_compiler::GlobalOptimization::createCleanupNumericNarrowingPass()";
}

def Convert1X1FilterConv2DToMatmul:
    Pass<"iree-global-opt-convert-1x1-filter-conv2d-to-matmul", ""> {
  let summary = "Convert linalg convolution ops with 1x1 kernels into linalg matrix multiplication ops.";
  let constructor = "mlir::iree_compiler::GlobalOptimization::createConvert1X1FilterConv2DToMatmulPass()";
}

def DecomposeConcat :
    Pass<"iree-global-opt-decompose-concat", ""> {
  let summary = "Decomposes concatenations into a destination and a sequence of slice inserts";
  let constructor = "mlir::iree_compiler::GlobalOptimization::createDecomposeConcatPass()";
  let options = [
    Option<"enableConcatTransposition", "enable-concat-transposition", "bool",
           /*default=*/"false", "Allows transposing concatenations such that "
                                "they occur on the inner most dims">,
  ];
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

def ExpandTensorShapes :
    Pass<"iree-global-opt-expand-tensor-shapes", "mlir::ModuleOp"> {
  let summary = "Expands tensor shape dimensions into SSA values across the program.";
  let constructor = "mlir::iree_compiler::GlobalOptimization::createExpandTensorShapesPass()";
}

def ExpandVectors : Pass<"iree-global-opt-expand-vectors", ""> {
  let summary = "Expand vector in vector/matrix operations";
  let constructor = "mlir::iree_compiler::GlobalOptimization::createExpandVectorsPass()";
}

def FuseDequantizationMatmul:
    InterfacePass<"iree-global-opt-fuse-dequantization-matmul", "mlir::FunctionOpInterface"> {
  let summary = "Fuse dequantization and matmul linalg.generic ops";
  let constructor = "mlir::iree_compiler::GlobalOptimization::createFuseDequantizationMatmulPass()";
  let options = [
    Option<"enableQuantizedMatmulReassociation", "enable-quantized-matmul-reassociation", "bool",
           /*default=*/"false", "Allow reassociation of quantized matmuls (experimental)">,
  ];
}

def FuseSiluHorizontalMatmul:
    InterfacePass<"iree-global-opt-fuse-silu-horizontal-matmul", "mlir::FunctionOpInterface"> {
  let summary = "Fuse matmul ops and silu linalg.generic op";
  let constructor = "mlir::iree_compiler::GlobalOptimization::createFuseSiluHorizontalMatmulPass()";
}

def GeneralizeLinalgNamedOps :
    InterfacePass<"iree-global-opt-generalize-linalg-named-ops", "mlir::FunctionOpInterface"> {
  let summary = "Convert some Linalg named ops into linalg.generics";
  let constructor = "mlir::iree_compiler::GlobalOptimization::createGeneralizeLinalgNamedOpsPass()";
}

def InferNumericNarrowing :
    Pass<"iree-global-opt-infer-numeric-narrowing", ""> {
  let summary = "Infers and inserts util.numeric.optional_narrow ops at points that may be beneficial";
  let constructor = "mlir::iree_compiler::GlobalOptimization::createInferNumericNarrowingPass()";
}

def LiftGenericToTransposeBatchMatmul:
    Pass<"iree-global-opt-lift-generic-to-tranpose-batch-matmul", ""> {
  let summary = "Convert linalg.generic ops to linalg.batch_matmul, possibly with transposes on operands/result.";
  let constructor = "mlir::iree_compiler::GlobalOptimization::createLiftGenericToTransposeBatchMatmulPass()";
}

def MaterializeHomogeneousEncodings :
  Pass<"iree-global-opt-materialize-homogeneous-encodings", "mlir::ModuleOp"> {
  let summary = "Materializes logical encodings to physical encodings if there is a single device target.";
  let constructor =
  "mlir::iree_compiler::GlobalOptimization::createMaterializeHomogeneousEncodingsPass()";
}

def OptimizeNumerics :
    Pass<"iree-global-opt-optimize-numerics", ""> {
  let summary = "Optimizes numerics given annotations added via iree-global-opt-infer-numeric-narrowing";
  let constructor = "mlir::iree_compiler::GlobalOptimization::createOptimizeNumericsPass()";
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

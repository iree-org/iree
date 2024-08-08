// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_GLOBALOPTIMIZATION_PASSES
#define IREE_COMPILER_GLOBALOPTIMIZATION_PASSES

include "mlir/Pass/PassBase.td"

def CleanupNumericNarrowingPass :
    Pass<"iree-global-opt-cleanup-numeric-narrowing", ""> {
  let summary = "Cleans up any numeric narrowing ops inserted by iree-global-opt-infer-numeric-narrowing.";
}

def Convert1X1FilterConv2DToMatmulPass:
    Pass<"iree-global-opt-convert-1x1-filter-conv2d-to-matmul", ""> {
  let summary = "Convert linalg convolution ops with 1x1 kernels into linalg matrix multiplication ops.";
}

def DecomposeConcatPass :
    Pass<"iree-global-opt-decompose-concat", ""> {
  let summary = "Decomposes concatenations into a destination and a sequence of slice inserts.";
  let options = [
    Option<"enableConcatTransposition", "enable-concat-transposition", "bool",
           /*default=*/"false", "Allows transposing concatenations such that "
                                "they occur on the inner most dims.">,
  ];
}

def DemoteContractionInputsToBF16Pass
    : Pass<"iree-global-opt-demote-contraction-inputs-to-bf16", ""> {
  let summary =
      "Demotes inputs (LHS, RHS) of linalg matmul-like ops from f32 to bf16.";
  let options =
      [Option<"demoteOnly", "demote-only",
              "mlir::iree_compiler::GlobalOptimization::DemotionOption",
              /*default=*/
              "mlir::iree_compiler::GlobalOptimization::DemotionOption::All",
              "Select the type of contraction ops to demote.",
              [{::llvm::cl::values(
            clEnumValN(mlir::iree_compiler::GlobalOptimization::DemotionOption::All,
                       "all",
                      "demote all contraction ops."),
            clEnumValN(mlir::iree_compiler::GlobalOptimization::DemotionOption::Conv,
                       "conv",
                       "Only demote convolution ops."),
            clEnumValN(mlir::iree_compiler::GlobalOptimization::DemotionOption::Matmul,
                       "matmul",
                       "Only demote matmul ops."),
            clEnumValN(mlir::iree_compiler::GlobalOptimization::DemotionOption::None,
                       "none",
                      "demote no contraction ops.")
        )}]>,
  ];
}

def DetachElementwiseFromNamedOpsPass :
    Pass<"iree-global-opt-detach-elementwise-from-named-ops", ""> {
  let summary = "Detaches elementwise ops from named Linalg ops.";
}

def EraseUnusedLinalgOperandsPass :
    Pass<"iree-global-opt-erase-unused-linalg-operands", "mlir::ModuleOp"> {
  let summary = "Erases unused linalg operand and remove dead code.";
}

def ExpandTensorShapesPass :
    Pass<"iree-global-opt-expand-tensor-shapes", "mlir::ModuleOp"> {
  let summary = "Expands tensor shape dimensions into SSA values across the program.";
}

def FuseDequantizationMatmulPass:
    InterfacePass<"iree-global-opt-fuse-dequantization-matmul", "mlir::FunctionOpInterface"> {
  let summary = "Fuses dequantization and matmul linalg.generic ops.";
}

def FuseHorizontalContractionsPass:
    InterfacePass<"iree-global-opt-fuse-horizontal-contractions", "mlir::FunctionOpInterface"> {
  let summary = "Fuses horizontal contraction ops without fusions";
}

def FuseSiluHorizontalMatmulPass:
    InterfacePass<"iree-global-opt-fuse-silu-horizontal-matmul", "mlir::FunctionOpInterface"> {
  let summary = "Fuses matmul ops and silu linalg.generic op.";
}

def GeneralizeLinalgNamedOpsPass :
    InterfacePass<"iree-global-opt-generalize-linalg-named-ops", "mlir::FunctionOpInterface"> {
  let summary = "Convert some Linalg named ops into linalg.generics.";
}

def InferNumericNarrowingPass :
    Pass<"iree-global-opt-infer-numeric-narrowing", ""> {
  let summary = "Infers and inserts util.numeric.optional_narrow ops at points that may be beneficial.";
}

def MaterializeHomogeneousEncodingsPass :
  Pass<"iree-global-opt-materialize-homogeneous-encodings", "mlir::ModuleOp"> {
  let summary = "Materializes logical encodings to physical encodings if there is a single device target.";
}

def OptimizeNumericsPass :
    Pass<"iree-global-opt-optimize-numerics", ""> {
  let summary = "Optimizes numerics given annotations added via iree-global-opt-infer-numeric-narrowing.";
}

def PropagateLinalgTransposePass :
    InterfacePass<"iree-global-opt-propagate-linalg-transpose", "mlir::FunctionOpInterface"> {
  let summary = "Propagates linalg.transpose through a restricted set of ops.";
  let options = [
    Option<"enableAggressivePropagation", "enable-aggressive-propagation", "bool",
           /*default=*/"false", "Enable aggressive propagation to named ops.">,
    Option<"testSinkingOnly", "test-sinking-only", "bool", /*default=*/"false",
           "Flag used for lit-testing sinking patterns only. Not for general usage">,
    Option<"testBubblingOnly", "test-bubbling-only", "bool", /*default=*/"false",
           "Flag used for lit-testing bubbling patterns only. Not for general usage">,
  ];
}

def RaiseSpecialOpsPass :
    Pass<"iree-global-opt-raise-special-ops", ""> {
  let summary = "Raises special ops like softmax to the high level linalg.ext representation.";
}

def RemoveZeroExtentTensorsPass :
    InterfacePass<"iree-global-opt-remove-zero-extent-tensors", "mlir::FunctionOpInterface"> {
  let summary = "Removes tensors that have 0-extents.";
}

def SimplifyPackUnpackPass : Pass<"iree-global-opt-simplify-pack-unpack", ""> {
  let summary = "Simplifies tensor pack and unpack ops.";
}

def GlobalLoopInvariantCodeMotionPass : InterfacePass<"iree-global-opt-loop-invariant-code-motion", "mlir::FunctionOpInterface"> {
  let summary = "Hoist loop invariants out of loops with zero-trip-check.";
}

def DataLayoutPropagationPass : InterfacePass<"iree-global-opt-data-layout-propagation", "mlir::FunctionOpInterface"> {
  let summary = "Propagate pack/unpack ops across other ops to improve fusion";
}

#endif // IREE_COMPILER_GLOBALOPTIMIZATION_PASSES

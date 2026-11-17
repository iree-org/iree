// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_PLUGINS_INPUT_STABLEHLO_CONVERSION_PREPROCESSING_PASSES
#define IREE_COMPILER_PLUGINS_INPUT_STABLEHLO_CONVERSION_PREPROCESSING_PASSES

include "mlir/Pass/PassBase.td"

def StableHLOToStableHLOPreprocessing :
    Pass<"iree-stablehlo-to-stablehlo-preprocessing", "ModuleOp"> {
  let summary = "Applies IREE-specific stablehlo to stablehlo preprocessing transformations";
  let options = [
    Option<"orderConvFeatures", "order-conv-features", "bool", /*default=*/"true",
           "Guarantees input/output features ordered from conv kernel">
  ];
}

def StableHLOCanonicalize :
    InterfacePass<"iree-stablehlo-canonicalize", "mlir::FunctionOpInterface"> {
  let summary = "Canonicalizes StableHLO operations";
}

def DotGeneralToDot :
    InterfacePass<"iree-stablehlo-preprocessing-dot-general-to-dot", "mlir::FunctionOpInterface"> {
  let summary = "Lowers general dot to a non-batched dot when possible";
}

def EinsumToDotGeneral :
    InterfacePass<"iree-stablehlo-preprocessing-einsum-to-dot-general", "mlir::FunctionOpInterface"> {
  let summary = "Legalizes einsum ops to general dot ops";
}

def FlattenTuplesInCFG :
    Pass<"iree-stablehlo-preprocessing-flatten-cfg-tuples", "ModuleOp"> {
  let summary = "Flattens tuples in the CFG form of StableHLO";
  let description = [{
    Tuple arguments and results become their leaves in depth-first order.
    Per-value metadata follows non-tuple slots, including remapped alias and
    tie references. Attributes on tuple slots and ties to tuple slots are
    unsupported. Tuple signatures support direct calls and result-dimension
    helper references; other symbol uses and indirect tuple calls are rejected.
  }];
  let dependentDialects = [
    "::mlir::func::FuncDialect",
    "::mlir::cf::ControlFlowDialect",
    "::mlir::stablehlo::StablehloDialect",
  ];
}

def FlattenPublicTupleSignatures :
    Pass<"iree-stablehlo-preprocessing-flatten-public-tuple-signatures",
         "ModuleOp"> {
  let summary = "Flattens tuples in public function signatures";
  let description = [{
    Public tuple arguments and results become their leaves in depth-first
    order, and empty tuples vanish. The source signature is kept in
    `iree.abi.declaration`, with token leaves shown as the fences they become.
    Per-value metadata follows non-tuple slots, including remapped alias and
    tie references. Attributes on tuple slots and ties to tuple slots are
    unsupported. Symbol users must be direct calls to the function.
  }];
  let dependentDialects = [
    "::mlir::iree_compiler::IREE::HAL::HALDialect",
    "::mlir::stablehlo::StablehloDialect",
  ];
}

def FlattenTuplesInSCF :
    Pass<"iree-stablehlo-preprocessing-flatten-scf-tuples", "ModuleOp"> {
  let summary = "Flattens tuples in the SCF form of StableHLO";
}

def GatherToTorchIndexSelect :
    InterfacePass<"iree-stablehlo-preprocessing-gather-to-torch-index-select", "mlir::FunctionOpInterface"> {
  let summary = "Legalizes gathers to a torch index select";
}

def LowerBounds :
    Pass<"iree-stablehlo-preprocessing-lower-bounds", "func::FuncOp"> {
  let summary = "Lowers the bounds encoding to util.assume.int and tie_shape";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::tensor::TensorDialect",
    "mlir::iree_compiler::IREE::Flow::FlowDialect",
    "mlir::iree_compiler::IREE::Util::UtilDialect",
  ];
}

def LowerComplex :
    InterfacePass<"iree-stablehlo-preprocessing-lower-complex", "mlir::FunctionOpInterface"> {
  let summary = "Lowers complex operations into non-complex operations";
}

def UnfuseBatchNorm :
    InterfacePass<"iree-stablehlo-preprocessing-unfuse-batch-norm", "mlir::FunctionOpInterface"> {
  let summary = "Materializes 'broadcast_dimensions' attributes";
}

#endif // IREE_COMPILER_PLUGINS_INPUT_STABLEHLO_CONVERSION_PREPROCESSING_PASSES

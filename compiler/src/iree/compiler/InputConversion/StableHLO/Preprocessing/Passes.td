// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_INPUTCONVERSION_STABLEHLO_PREPROCESSING_PASSES
#define IREE_COMPILER_INPUTCONVERSION_STABLEHLO_PREPROCESSING_PASSES

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
    Pass<"iree-stablehlo-canonicalize", "func::FuncOp"> {
  let summary = "Canonicalizes StableHLO operations";
}

def DotGeneralToDot :
    Pass<"iree-stablehlo-preprocessing-dot-general-to-dot", "func::FuncOp"> {
  let summary = "Lowers general dot to a non-batched dot when possible";
}

def EinsumToDotGeneral :
    Pass<"iree-stablehlo-preprocessing-einsum-to-dot-general", "func::FuncOp"> {
  let summary = "Legalizes einsum ops to general dot ops";
}

def FlattenTuplesInCFG :
    Pass<"iree-stablehlo-preprocessing-flatten-tuples", "ModuleOp"> {
  let summary = "Flattens tuples in the CFG form of StableHLO";
}

def GatherToTorchIndexSelect :
    Pass<"iree-stablehlo-preprocessing-gather-to-torch-index-select", "func::FuncOp"> {
  let summary = "Legalizes gathers to a torch index select";
}

def LowerComplex :
    Pass<"iree-stablehlo-preprocessing-lower-complex", "func::FuncOp"> {
  let summary = "Lowers complex operations into non-complex operations";
}

def UnfuseBatchNorm :
    Pass<"iree-stablehlo-preprocessing-unfuse-batch-norm", "func::FuncOp"> {
  let summary = "Materializes 'broadcast_dimensions' attributes";
}

#endif // IREE_COMPILER_INPUTCONVERSION_STABLEHLO_PREPROCESSING_PASSES

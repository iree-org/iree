// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef STABLEHLO_IREE_CONVERSION_PREPROCESSING_PASSES
#define STABLEHLO_IREE_CONVERSION_PREPROCESSING_PASSES

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
}

def FlattenTuplesInSCF :
    Pass<"iree-stablehlo-preprocessing-flatten-scf-tuples", "ModuleOp"> {
  let summary = "Flattens tuples in the SCF form of StableHLO";
}

def GatherToTorchIndexSelect :
    InterfacePass<"iree-stablehlo-preprocessing-gather-to-torch-index-select", "mlir::FunctionOpInterface"> {
  let summary = "Legalizes gathers to a torch index select";
}

def LowerComplex :
    InterfacePass<"iree-stablehlo-preprocessing-lower-complex", "mlir::FunctionOpInterface"> {
  let summary = "Lowers complex operations into non-complex operations";
}

def UnfuseBatchNorm :
    InterfacePass<"iree-stablehlo-preprocessing-unfuse-batch-norm", "mlir::FunctionOpInterface"> {
  let summary = "Materializes 'broadcast_dimensions' attributes";
}

#endif // STABLEHLO_IREE_CONVERSION_PREPROCESSING_PASSES

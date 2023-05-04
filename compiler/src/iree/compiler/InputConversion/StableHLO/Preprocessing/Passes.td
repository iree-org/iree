// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_INPUTCONVERSION_STABLEHLO_PREPROCESSING_PASSES
#define IREE_COMPILER_INPUTCONVERSION_STABLEHLO_PREPROCESSING_PASSES

include "mlir/Pass/PassBase.td"

def DotGeneralToDot :
    Pass<"iree-stablehlo-preprocessing-dot-general-to-dot", "func::FuncOp"> {
  let summary = "Lowers general dot to a non-batched dot when possible";
}

def EinsumToDotGeneral :
    Pass<"iree-stablehlo-preprocessing-einsum-to-dot-general", "func::FuncOp"> {
  let summary = "Legalizes einsum ops to general dot ops";
}

def GatherToTorchIndexSelect :
    Pass<"iree-stablehlo-preprocessing-gather-to-torch-index-select", "func::FuncOp"> {
  let summary = "Legalizes gathers to a torch index select";
}

def LowerComplex :
    Pass<"iree-stablehlo-preprocessing-lower-complex", "func::FuncOp"> {
  let summary = "Lower complex operations into non-complex operations";
}

def UnfuseBatchNorm :
    Pass<"iree-stablehlo-preprocessing-unfuse-batch-norm", "func::FuncOp"> {
  let summary = "Materializes 'broadcast_dimensions' attributes";
}

#endif // IREE_COMPILER_INPUTCONVERSION_STABLEHLO_PREPROCESSING_PASSES

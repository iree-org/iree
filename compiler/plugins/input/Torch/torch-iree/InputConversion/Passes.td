// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TORCH_IREE_INPUTCONVERSION_PASSES
#define TORCH_IREE_INPUTCONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertTMTensorToLinalgExt :
    Pass<"torch-iree-tm-tensor-to-linalg-ext", "func::FuncOp"> {
  let summary = "Convert from TMTensor ops to LinalgExt ops on tensors";
  let constructor = "mlir::iree_compiler::TorchInput::createConvertTMTensorToLinalgExtPass()";
}

def SetStrictSymbolicShapesPass :
    Pass<"torch-iree-set-strict-symbolic-shapes", "func::FuncOp"> {
  let summary = "Adds the attribute indicating strict symbolic shapes in Torch IR";
  let constructor = "mlir::iree_compiler::TorchInput::createSetStrictSymbolicShapesPass()";
}

#endif // TORCH_IREE_INPUTCONVERSION_PASSES

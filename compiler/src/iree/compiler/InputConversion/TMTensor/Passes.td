// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_INPUTCONVERSION_TMTENSOR_PASSES
#define IREE_COMPILER_INPUTCONVERSION_TMTENSOR_PASSES

include "mlir/Pass/PassBase.td"

def ConvertTMTensorToLinalgExt :
    Pass<"iree-tm-tensor-to-linalg-ext", "func::FuncOp"> {
  let summary = "Convert from TMTensor ops to LinalgExt ops on tensors";
  let constructor = "mlir::iree_compiler::TMTensor::createConvertTMTensorToLinalgExtPass()";
}

#endif // IREE_COMPILER_INPUTCONVERSION_TMTENSOR_PASSES

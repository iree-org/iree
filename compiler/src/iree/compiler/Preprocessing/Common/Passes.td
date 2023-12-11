// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_PREPROCESSING_COMMON_PASSES
#define IREE_PREPROCESSING_COMMON_PASSES

include "mlir/Pass/PassBase.td"

def ConvertConv2DToImg2Col :
    Pass<"iree-preprocessing-convert-conv2d-to-img2col", ""> {
  let summary = "Convert linalg convolution ops to matmul img2col based implementation";
  let constructor = "mlir::iree_compiler::Preprocessing::createConvertConv2DToImg2ColPass()";
}

def MakeSingleDispatchForFunction :
    Pass<"iree-preprocessing-make-single-dispatch-for-function", "func::FuncOp"> {
  let summary = "Convert entire function into a single dispatch";
  let constructor = "mlir::iree_compiler::Preprocessing::createMakeSingleDispatchForFunctionPass()";
}

def PadLinalgOps :
    Pass<"iree-preprocessing-pad-linalg-ops", ""> {
  let summary = "Pad linalg ops to the next integer multiple of paddingSize.";
  let constructor = "mlir::iree_compiler::Preprocessing::createPadLinalgOpsToIntegerMultiplePass()";
  let options = [
    Option<"paddingSize", "pad-size", "int",
           /*default=*/"4",
           "Specify the padding size">,
  ];
}

#endif  // IREE_PREPROCESSING_COMMON_PASSES

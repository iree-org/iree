// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_DIALECT_VECTOR_EXT_TRANSFORMS_PASSES
#define IREE_CODEGEN_DIALECT_VECTOR_EXT_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def VectorExtFoldUnitExtentDimsPass :
    Pass<"iree-vector-ext-fold-unit-extent-dims", ""> {
  let summary = "Folds unit dims for iree_vector_ext ops";
  let dependentDialects = [
    "::mlir::tensor::TensorDialect",
    "::mlir::iree_compiler::IREE::VectorExt::IREEVectorExtDialect"
  ];
}

def LowerTransferGatherToGatherPass :
    Pass<"iree-vector-ext-lower-transfer-gather-to-gather", "func::FuncOp"> {
  let summary = "Lower iree_vector_ext.transfer_gather to vector.gather";
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::vector::VectorDialect",
    "::mlir::iree_compiler::IREE::VectorExt::IREEVectorExtDialect"
  ];
}

#endif // IREE_CODEGEN_DIALECT_VECTOR_EXT_TRANSFORMS_PASSES

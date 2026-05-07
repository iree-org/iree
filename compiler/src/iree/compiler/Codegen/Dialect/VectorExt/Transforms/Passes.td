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

def LowerTransferGatherScatterToVectorPass :
    Pass<"iree-vector-ext-lower-transfer-gather-scatter-to-vector", "func::FuncOp"> {
  let summary = "Lower transfer_gather/transfer_scatter to vector.gather/vector.scatter";
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::vector::VectorDialect",
    "::mlir::iree_compiler::IREE::VectorExt::IREEVectorExtDialect"
  ];
}

// TODO(Bangtian): Consolidate per-op lowering passes into a single generic
// pass once we have more such lowerings.
def LowerArgCompareToVectorPass :
    Pass<"iree-vector-ext-lower-arg-compare-to-vector", "func::FuncOp"> {
  let summary = "Lower iree_vector_ext.arg_compare to vector/SCF ops";
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::vector::VectorDialect",
    "::mlir::iree_compiler::IREE::VectorExt::IREEVectorExtDialect"
  ];
}

#endif // IREE_CODEGEN_DIALECT_VECTOR_EXT_TRANSFORMS_PASSES

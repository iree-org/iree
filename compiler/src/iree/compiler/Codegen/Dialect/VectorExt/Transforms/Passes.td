// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_DIALECT_VECTOR_EXT_TRANSFORMS_PASSES
#define IREE_CODEGEN_DIALECT_VECTOR_EXT_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def VectorizeIREEVectorExtOpsPass :
    Pass<"iree-vector-ext-vectorize-ops", ""> {
  let summary = "Vectorizes then lowers a few iree_vector_ext ops before vectorization.";
  let dependentDialects = [
    "::mlir::vector::VectorDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::iree_compiler::IREE::VectorExt::IREEVectorExtDialect"
  ];
}

def VectorExtFoldUnitExtentDimsPass :
    Pass<"iree-vector-ext-fold-unit-extent-dims", ""> {
  let summary = "Folds unit dims for iree_vector_ext ops";
  let dependentDialects = [
    "::mlir::tensor::TensorDialect",
    "::mlir::iree_compiler::IREE::VectorExt::IREEVectorExtDialect"
  ];
}

def GreedySLPVectorizerPass : Pass<"iree-greedy-slp-vectorizer"> {
  let summary = "SLP Vectorizer Pass";
  let description = [{
    This pass implements the SLP (Superword Level Parallelism) vectorizer.
    It detects consecutive operations that can be put together into vector
    operations. The pass works bi-directionaly, starting from reads or stores,
    in search of scalars to combine.

    This is greedy vectorizer, it doesn't have any cost model (yet) and it tries
    to create vector ops if we have at least 2 potential ops.
  }];
  let dependentDialects = ["::mlir::vector::VectorDialect"];

  let options = [
    Option<"maxVectorBitwidth", "max-vector-bitwidth", "unsigned",
      /*default=*/"std::numeric_limits<unsigned>::max()",
      "Maximum supported vector bitwidth">,
  ];
}

#endif // IREE_CODEGEN_DIALECT_VECTOR_EXT_TRANSFORMS_PASSES

// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_SANDBOX_PASSES_TD
#define IREE_CODEGEN_SANDBOX_PASSES_TD

include "mlir/Pass/PassBase.td"

def LinalgSplitReduction : Pass<"linalg-split-reduction", "func::FuncOp"> {
  let summary = "Pass to splitReduce linalg operations.";
  let constructor = "mlir::createLinalgSplitReductionPass()";
  let dependentDialects = [
    "::mlir::linalg::LinalgDialect", "::mlir::scf::SCFDialect"
  ];
  let options = [
    // This corresponds to the value in the last level of tiling.
    // `size=0` triggers reading the `lowering_config` and if there's no
    // `lowering_config`, this pass doesn't execute. Executing this pass with
    // tile size of 1 is strictly worse than not executing the pass (and instead
    // relying on tiling from TileAndFuse pass).
    Option<"size", "size", "int64_t", /*default=*/"0",
      "Size that splitReduction should use for parallel reduction results.">,
  ];
}

def LinalgSingleTilingExpert
    : Pass<"linalg-single-tiling-expert-driver", "func::FuncOp"> {
  let summary = "Pass to drive transformations on Linalg on tensors.";
  let constructor = "mlir::createLinalgSingleTilingExpertPass()";
  let options = [
    // Func / op targeting options.
    Option<"anchorFuncOpName", "anchor-func", "std::string", /*default=*/"",
      "Which func op is the anchor to latch on.">,
    Option<"anchorOpName", "anchor-op", "std::string", /*default=*/"",
      "Which linalg op within the func is the anchor to latch on.">,

    // Tiling options.
    ListOption<"tileSizes", "tile-sizes", "int64_t", "Tile sizes",
               "llvm::cl::ZeroOrMore">,
    ListOption<"tileInterchange", "tile-interchange", "int64_t",
                "Tile loop interchange",
                "llvm::cl::ZeroOrMore">,

    // Generalization options.
    Option<"generalize", "generalize", "bool", /*default=*/"false",
      "Convert named operations to their generic form.">,
    ListOption<"iteratorInterchange", "iterator-interchange", "int64_t",
               "Interator interchange.",
               "llvm::cl::ZeroOrMore">,

    // Vectorization options.
    Option<"vectorize", "vectorize", "bool", /*default=*/"false",
      "Rewrite the linalg op as a vector operation.">,
    Option<"enableVectorMasking", "enableVectorMasking", "bool", /*default=*/"false",
      "Enable vector masking during vectorization.">,
    Option<"vectorizePadding", "vectorize-padding", "bool", /*default=*/"false",
      "Rewrite all tensor.pad ops in the function to vector form.">,

    // IREE specific options
    Option<"tilingLevel", "tiling-level", "int64_t", /*default=*/"-1",
      "Use default tiling level used to retrieve the configuration from lowering_config">
  ];
  let dependentDialects = [
    "::mlir::arith::ArithDialect", "::mlir::AffineDialect",
    "::mlir::linalg::LinalgDialect", "::mlir::scf::SCFDialect",
    "::mlir::cf::ControlFlowDialect", "::mlir::tensor::TensorDialect",
    "::mlir::vector::VectorDialect"
  ];
}

def LinalgVectorLowering : Pass<"linalg-vector-lowering", "func::FuncOp"> {
  let summary = "Run transformations that lower high-level vectors.";
  let constructor = "mlir::createLinalgVectorLoweringPass()";
  let options = [
    Option<"vectorLoweringStage", "lower-vector-stage", "int", /*default=*/"0",
      [{Which stage of vector lowering to run:\n"
          "\t0 [default] only lower vector.contract\n"
          "\t1 additionally lower vector.multi_reduction\n"
          "\t2 additionally enable vector.transfer split\n"
          "\t3 additionally lower vector.transfer\n"
          "\t4 additionally lower vector.transfer to scf\n"
          "\t5 additionally lower vector.shape_cast\n"
          "\t6 additionally lower vector.transpose\n}]>,
    Option<"splitVectorTransfersTo", "split-transfers", "std::string",
      /*default=*/"",
      [{Split vector transfers between slow (masked) and fast "
        "(unmasked) variants. Possible options are:\n"
          "\tnone [default]: keep unsplit vector.transfer and pay the price\n"
          "\tlinalg-copy: use linalg.fill + linalg.generic for the slow path\n"
          "\tvector-transfers: use extra small unmasked vector.transfers for"
          " the slow path\n}]>,
    Option<"lowerVectorTransposeTo", "lower-vector-transpose-to",
       "std::string", /*default=*/[{"eltwise"}],
      [{Lower vector.transpose to finer-grained vector ops, options are:\n"
          "\teltwise [default]\n"
          "\tflat_transpose (requires LLVM matrix intrinsics support)\n"
          "\tshuffle (lower 2-D transposes to shape_cast + shuffle)\n}]>,
    Option<"lowerVectorTransposeToAVX2", "lower-vector-transpose-to-avx2", "bool",
      /*default=*/"false",
      "Add specific transpose to avx2 lowering patterns.">,
    Option<"lowerVectorMultiReductionTo", "lower-vector-multi-reduction-to",
       "std::string", /*default=*/[{"innerparallel"}],
      [{Lower vector.multi_reduction to finer-grained vector ops, options are:\n"
          "\tinnerparallel [default]\n"
          "\tinnerreduction\n}]>,
    Option<"lowerVectorContractionTo", "lower-vector-contraction-to", "std::string",
      /*default=*/[{"outerproduct"}],
      [{Lower vector.contract to finer-grained vector ops, options are:\n"
          "\touterproduct [default]\n"
          "\tdot\n"
          "\tmatrixintrinsics\n}]>,
    Option<"unrollVectorTransfers", "unroll-vector-transfers", "bool",
      /*default=*/"true",
      "Run transformations that lower high-level vectors.">,
    Option<"maxTransferRank", "max-transfer-rank", "int64_t", /*default=*/"1",
      "Set the maximum vector load/store rank.">
  ];
  let dependentDialects = [
    "::mlir::arith::ArithDialect", "::mlir::AffineDialect",
    "::mlir::linalg::LinalgDialect", "::mlir::memref::MemRefDialect",
    "::mlir::scf::SCFDialect", "::mlir::cf::ControlFlowDialect",
    "::mlir::vector::VectorDialect"
  ];
}

#endif // IREE_CODEGEN_SANDBOX_PASSES_TD

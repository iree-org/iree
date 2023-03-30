// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_SANDBOX_PASSES_TD
#define IREE_CODEGEN_SANDBOX_PASSES_TD

include "mlir/Pass/PassBase.td"

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

// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_COMMON_CPU_PASSES
#define IREE_CODEGEN_COMMON_CPU_PASSES

include "mlir/Pass/PassBase.td"

//===---------------------------------------------------------------------===//
// Common Passes used for CPU-like backends (keep alphabetical)
//===---------------------------------------------------------------------===//

def CPUMaterializeEncoding :
    InterfacePass<"iree-codegen-cpu-materialize-encoding", "mlir::FunctionOpInterface"> {
  let summary = "Materialize the encoding for tensor as specified by the backend";
  let constructor = "mlir::iree_compiler::createCPUMaterializeEncodingPass()";
}

def CPUMaterializeUpperBoundTileSize :
    InterfacePass<"iree-codegen-cpu-materialize-upper-bound-tile-size", "mlir::FunctionOpInterface"> {
  let summary = "Materialize upper_bound_tile_size to constants.";
  let constructor = "mlir::iree_compiler::createCPUMaterializeUpperBoundTileSizePass()";
}

def CPULowerToUKernels :
    Pass<"iree-codegen-cpu-lower-to-ukernels", ""> {
  let summary =
      "Separate out parts of the IR that lower to a micro-kernel";
  let constructor =
      "mlir::iree_compiler::createCPULowerToUKernelsPass()";
  let options = [
    Option<"optionSkipIntermediateRoundings", "skip-intermediate-roundings",
      "bool", /*default=*/"true",
      "Allow skipping intermediate roundings, e.g. in f16 ukernels internally doing f32 arithmetic.">,
  ];
}

#endif  // IREE_CODEGEN_COMMON_CPU_PASSES

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

def CPUMaterializeHostEncodingPass :
    Pass<"iree-codegen-cpu-materialize-host-encoding", "mlir::ModuleOp"> {
  let summary = "Convert encoding-specific operations based on target attributes.";
  let description = [{
    Examples:
      encoding.set_encoding   -> tensor.pack
      encoding.unset_encoding -> tensor.unpack
      linalg.matmul             -> linalg.mmt4d  "}];
}

def CPUMaterializeDeviceEncodingPass :
    InterfacePass<"iree-codegen-cpu-materialize-device-encoding", "mlir::FunctionOpInterface"> {
  let summary = "Convert encoding-specific operations based on target attributes.";
  let description = [{
    Examples:
      encoding.set_encoding   -> tensor.pack
      encoding.unset_encoding -> tensor.unpack
      linalg.matmul             -> linalg.mmt4d  "}];
}

def CPULowerToUKernelsPass :
    Pass<"iree-codegen-cpu-lower-to-ukernels", ""> {
  let summary =
      "Separate out parts of the IR that lower to a micro-kernel";
  let options = [
    Option<"skipIntermediateRoundings", "skip-intermediate-roundings",
      "bool", /*default=*/"true",
      "Allow skipping intermediate roundings, e.g. in f16 ukernels internally doing f32 arithmetic.">,
  ];
}

def CPUPrepareUkernelsPass :
    InterfacePass<"iree-codegen-cpu-prepare-ukernels", "mlir::FunctionOpInterface"> {
  let summary = "Rank reduce operations to fit existing ukernels requirements."
                "For example, batch_mmt4d ops are decomposed to mmt4d ops";
}

#endif  // IREE_CODEGEN_COMMON_CPU_PASSES

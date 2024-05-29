// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_DIALECT_GPU_TRANSFORMS_PASSES
#define IREE_CODEGEN_DIALECT_GPU_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def FuseAndHoistParallelLoopsPass :
    InterfacePass<"iree-gpu-fuse-and-hoist-parallel-loops", "mlir::FunctionOpInterface"> {
  let summary = "Checks GPU specific resource usage constraints like shared memory limits";
  let dependentDialects = [
    "::mlir::affine::AffineDialect",
    "::mlir::iree_compiler::IREE::GPU::IREEGPUDialect"
  ];
}

#endif // IREE_CODEGEN_DIALECt_GPU_TRANSFORMS_PASSES

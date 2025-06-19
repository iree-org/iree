// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_DIALECT_GPU_TRANSFORMS_PASSES
#define IREE_CODEGEN_DIALECT_GPU_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def CombineBarrierRegionsPass :
    Pass<"iree-gpu-combine-barrier-regions", ""> {
  let summary = "Combines `iree_gpu.barrier_region` ops";
  let dependentDialects = ["::mlir::iree_compiler::IREE::GPU::IREEGPUDialect"];
}

def DistributeInnerTiledToLanesPass :
    InterfacePass<"iree-gpu-distribute-inner-tiled-to-lanes", "mlir::FunctionOpInterface"> {
  let summary = "Distributes iree_codegen.inner_tiled ops to lanes";
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::affine::AffineDialect",
    "::mlir::gpu::GPUDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::tensor::TensorDialect",
  ];
}

def ExpandUndistributedInnerTilesPass :
    InterfacePass<"iree-gpu-expand-undistributed-inner-tiles", "mlir::FunctionOpInterface"> {
  let summary = "Expands the inner dimensions of iree_codegen.inner_tiled ops to match the thread layout";
  let dependentDialects = [
    "::mlir::tensor::TensorDialect",
    "::mlir::iree_compiler::IREE::GPU::IREEGPUDialect",
  ];
  let options = [
    Option<"expandInputs", "expand-inputs",
      "bool", /*default=*/"true",
      "Expand the inner dimensions for the input operands of the inner_tiled ops.">,
    Option<"expandOutputs", "expand-outputs",
      "bool", /*default=*/"true",
      "Expand the inner dimensions for the output operands and results of the inner_tiled ops.">,
  ];
}

def LowerIREEGPUOpsPass :
    InterfacePass<"iree-gpu-lower-ops", "mlir::FunctionOpInterface"> {
  let summary = "Post bufferization lowerings of iree_gpu ops before late lowerings";
  let dependentDialects = [
    "::mlir::gpu::GPUDialect",
  ];
}

def UnrollToIntrinsicsPass :
    InterfacePass<"iree-gpu-unroll-to-intrinsics", "mlir::FunctionOpInterface"> {
  let summary = "Unrolls iree_gpu.multi_mma ops to their inner vector size.";
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::vector::VectorDialect",
  ];
}

def VectorizeIREEGPUOpsPass :
    InterfacePass<"iree-gpu-vectorize-ops", "mlir::FunctionOpInterface"> {
  let summary = "Vectorizes then lowers a few iree_gpu ops before vectorization.";
  let dependentDialects = [
    "::mlir::vector::VectorDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::iree_compiler::IREE::GPU::IREEGPUDialect"
  ];
}

#endif // IREE_CODEGEN_DIALECt_GPU_TRANSFORMS_PASSES

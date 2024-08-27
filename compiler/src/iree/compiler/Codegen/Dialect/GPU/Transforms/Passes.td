// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_DIALECT_GPU_TRANSFORMS_PASSES
#define IREE_CODEGEN_DIALECT_GPU_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def DistributeMmaToLanesPass :
    InterfacePass<"iree-gpu-distribute-mma-to-lanes", "mlir::FunctionOpInterface"> {
  let summary = "Distributes iree_gpu.multi_mma ops to lanes";
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::affine::AffineDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::tensor::TensorDialect",
  ];
}

def ConcretizeMmaShapesPass :
    InterfacePass<"iree-gpu-concretize-mma-shapes", "mlir::FunctionOpInterface"> {
  let summary = "Expands the inner dimensions of iree_gpu.multi_mma ops to match the thread layout";
  let dependentDialects = [
    "::mlir::tensor::TensorDialect",
    "::mlir::iree_compiler::IREE::GPU::IREEGPUDialect",
  ];
  let options = [
    Option<"concretizeInputs", "concretize-inputs",
      "bool", /*default=*/"true",
      "Expand the inner dimensions for the lhs and rhs operands of the multi_mma ops.">,
    Option<"concretizeResult", "concretize-result",
      "bool", /*default=*/"true",
      "Expand the inner dimensions for the acc operand of the multi_mma ops.">,
  ];
}

def FuseAndHoistParallelLoopsPass :
    InterfacePass<"iree-gpu-fuse-and-hoist-parallel-loops", "mlir::FunctionOpInterface"> {
  let summary = "Greedily fuses and hoists parallel loops.";
  let dependentDialects = [
    "::mlir::affine::AffineDialect",
    "::mlir::iree_compiler::IREE::GPU::IREEGPUDialect",
    "::mlir::bufferization::BufferizationDialect"
  ];
}

def LowerIREEGPUOpsPass :
    InterfacePass<"iree-gpu-lower-ops", "mlir::FunctionOpInterface"> {
  let summary = "Post bufferization lowerings of iree_gpu ops before late lowerings";
  let dependentDialects = [
    "::mlir::gpu::GPUDialect",
  ];
}

def PackToIntrinsicsPass :
    InterfacePass<"iree-gpu-pack-to-intrinsics", "mlir::FunctionOpInterface"> {
  let summary = "Packs matmul like operations and converts to iree_gpu.multi_mma";
  let dependentDialects = [
    "::mlir::tensor::TensorDialect",
    "::mlir::iree_compiler::IREE::GPU::IREEGPUDialect"
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

// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_LLVMGPU_PASSES
#define IREE_CODEGEN_LLVMGPU_PASSES

include "mlir/Pass/PassBase.td"

//------------------------------------------------------------------------------
// LLVMGPU Passes (keep alphabetical)
//------------------------------------------------------------------------------

// TODO: Bring the argument in line with the names used elsewhere.
def ConvertToNVVM :
    Pass<"iree-convert-to-nvvm", "ModuleOp"> {
  let summary = "Perform final conversion from builtin/GPU/HAL/standard dialect to LLVM "
    "and NVVM dialects";
  let constructor = "mlir::iree_compiler::createConvertToNVVMPass()";
}

// TODO: Bring the argument in line with the names used elsewhere.
def ConvertToROCDL :
    Pass<"iree-convert-to-rocdl", "ModuleOp"> {
  let summary = "Perform final conversion from builtin/GPU/HAL/standard dialect to LLVM "
    "and ROCDL dialects";
  let constructor = "mlir::iree_compiler::createConvertToROCDLPass()";
}

def ExtractAddressComputationGPU: Pass<"extract-address-computation-gpu"> {
  let summary = "Extract address computations from memory accesses";
  let description = [{
     This pass is similar to `extract-address-computation` except it also
     supports memory accesses that are specific to GPUs.
  }];
  let constructor = "mlir::iree_compiler::createExtractAddressComputationGPUPass()";
  let dependentDialects = [
      "memref::MemRefDialect", "nvgpu::NVGPUDialect", "affine::AffineDialect"
  ];
}

def LLVMGPUCastAddressSpaceFunction :
    Pass<"iree-llvmgpu-cast-address-space-function", "ModuleOp"> {
  let summary = "Pass to cast";
  let constructor = "mlir::iree_compiler::createLLVMGPUCastAddressSpaceFunction()";
}

def LLVMGPULowerExecutableTarget :
    Pass<"iree-llvmgpu-lower-executable-target", "mlir::iree_compiler::IREE::HAL::ExecutableVariantOp"> {
  let summary = "Perform lowering of executable target using one of the IREE::HAL::DispatchLoweringPassPipeline";
  let constructor = "mlir::iree_compiler::createLLVMGPULowerExecutableTargetPass()";
}

def LLVMGPUPackSharedMemoryAlloc :
    Pass<"iree-llvmgpu-pack-shared-memory-alloc", "func::FuncOp"> {
  let summary = "Pass pack shared memory allocation in order to reduce memory usage.";
  let constructor = "mlir::iree_compiler::createLLVMGPUPackSharedMemoryAlloc()";
}

def LLVMGPUSelectLoweringStrategy :
    Pass<"iree-llvmgpu-select-lowering-strategy", "mlir::iree_compiler::IREE::HAL::ExecutableVariantOp"> {
  let summary = "Select a IREE::HAL::DispatchLoweringPassPipeline for lowering the target variant";
  let constructor = "mlir::iree_compiler::createLLVMGPUSelectLoweringStrategyPass()";
}

def LLVMGPUTensorCorePreparation :
    Pass<"iree-llvmgpu-tensorcore-preparation", "func::FuncOp"> {
  let summary = "Pass to transform vector.contract to a form that can be lowered to GPU MMA ops";
  let constructor = "mlir::iree_compiler::createLLVMGPUTensorCorePreparationPass()";
}

def LLVMGPUTensorPad :
    Pass<"iree-llvmgpu-tensor-pad", "func::FuncOp"> {
  let summary = "Pass to pad out tensors up to static dimensions.";
  let constructor = "mlir::iree_compiler::createLLVMGPUTensorPadPass()";
}

def LLVMGPUTileAndDistribute :
    Pass<"iree-llvmgpu-tile-and-distribute", "func::FuncOp"> {
  let summary = "Pass to tile and distribute linalg ops within a workgroup.";
  let constructor = "mlir::iree_compiler::createLLVMGPUTileAndDistribute()";
}

def LLVMGPUVectorLowering :
    Pass<"iree-llvmgpu-vector-lowering", "func::FuncOp"> {
  let summary = "Pass to lower Vector ops before conversion to LLVM.";
  let constructor = "mlir::iree_compiler::createLLVMGPUVectorLoweringPass()";
}

def LLVMGPUVectorToGPU :
    Pass<"iree-llvmgpu-vector-to-gpu", "func::FuncOp"> {
  let summary = "Pass to convert vector to gpu.";
  let constructor = "mlir::iree_compiler::createLLVMGPUVectorToGPU()";
}

//------------------------------------------------------------------------------
// Test Passes
//------------------------------------------------------------------------------

def TestLLVMGPUScalarizeMathOp :
    Pass<"iree-test-llvmgpu-legalize-ops", "ModuleOp"> {
  let summary = "Test pass for several legalization patterns.";
  let constructor = "mlir::iree_compiler::createTestLLVMGPULegalizePass()";
}

#endif // IREE_CODEGEN_LLVMGPU_PASSES

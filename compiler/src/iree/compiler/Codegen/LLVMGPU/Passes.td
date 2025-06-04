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
def ConvertToNVVMPass :
    Pass<"iree-convert-to-nvvm", "ModuleOp"> {
  let summary = "Perform final conversion from builtin/GPU/HAL/standard dialect to LLVM "
    "and NVVM dialects";
}

// TODO: Bring the argument in line with the names used elsewhere.
def ConvertToROCDLPass :
    Pass<"iree-convert-to-rocdl", "ModuleOp"> {
  let summary = "Perform final conversion from builtin/GPU/HAL/standard dialect to LLVM "
    "and ROCDL dialects";
}

def ExtractAddressComputationGPUPass: Pass<"extract-address-computation-gpu"> {
  let summary = "Extract address computations from memory accesses";
  let description = [{
     This pass is similar to `extract-address-computation` except it also
     supports memory accesses that are specific to GPUs.
  }];
  let dependentDialects = [
      "memref::MemRefDialect", "nvgpu::NVGPUDialect", "affine::AffineDialect"
  ];
}

def LLVMGPUAssignConstantOrdinalsPass :
    Pass<"iree-llvmgpu-assign-constant-ordinals", "IREE::HAL::ExecutableVariantOp"> {
  let summary = "Assigns executable constant ordinals across all LLVMGPU variants.";
}

def LLVMGPUCastAddressSpaceFunctionPass :
    Pass<"iree-llvmgpu-cast-address-space-function", "ModuleOp"> {
  let summary = "Cast address space to generic in CallOp and FuncOp";
}

def LLVMGPUCastTypeToFitMMAPass : InterfacePass<"iree-llvmgpu-cast-type-to-fit-mma",
                                                "mlir::FunctionOpInterface"> {
  let summary = "Perform type extension/truncation over vector.contract types "
                "to target GPU MMA intrinsics";
}

def LLVMGPUConfigureTensorLayoutsPass :
    InterfacePass<"iree-llvmgpu-configure-tensor-layouts", "mlir::FunctionOpInterface"> {
  let summary = "Pass to set layouts on tensors for later vector distribution";
}

def LLVMGPULinkExecutablesPass :
    Pass<"iree-llvmgpu-link-executables", "mlir::ModuleOp"> {
  let summary = "Links LLVMGPU HAL executables within the top-level program module.";
  let options = [
    Option<
      "target", "target",
      "std::string", "",
      "Target backend name whose executables will be linked by this pass."
    >,
  ];
}

def LLVMGPULowerExecutableTargetPass :
    InterfacePass<"iree-llvmgpu-lower-executable-target", "mlir::FunctionOpInterface"> {
  let summary = "Perform lowering of executable target using one of the IREE::HAL::DispatchLoweringPassPipeline";
  let options = [
    Option<"forROCDL", "for-rocdl", "bool",
           /*default=*/"false",
           "Enable features only supported on ROCDL such as delaying lowering of subgroup reduce.">,
  ];
}

def LLVMGPUPackSharedMemoryAllocPass :
    InterfacePass<"iree-llvmgpu-pack-shared-memory-alloc", "mlir::FunctionOpInterface"> {
  let summary = "Pass pack shared memory allocation in order to reduce memory usage.";
}

def LLVMGPUPrefetchSharedMemoryPass :
    InterfacePass<"iree-llvmgpu-prefetch-shared-memory", "mlir::FunctionOpInterface"> {
  let summary = "Rotate scf.for loops to prefetch shared memory with distance 1";
}

def LLVMGPUPromoteMatmulToFitMMAPass :
    InterfacePass<"iree-llvmgpu-promote-matmul-to-fit-mma", "mlir::FunctionOpInterface"> {
  let summary = "Pass to promote contraction ops to fit mma shapes";
}

def LLVMGPUSelectLoweringStrategyPass :
    Pass<"iree-llvmgpu-select-lowering-strategy", "ModuleOp"> {
  let summary = "Select a IREE::HAL::DispatchLoweringPassPipeline for lowering the target variant";
}

def LLVMGPUTensorCoreVectorizationPass :
    InterfacePass<"iree-llvmgpu-tensorcore-vectorization", "mlir::FunctionOpInterface"> {
  let summary = "Pass to convert linalg into Vector and transform it to a form that can be lowered to GPU MMA ops";
}

def LLVMGPUTensorPadPass :
    InterfacePass<"iree-llvmgpu-tensor-pad", "mlir::FunctionOpInterface"> {
  let summary = "Pass to pad out tensors up to static dimensions.";
}

def LLVMGPUTileAndDistributePass :
    InterfacePass<"iree-llvmgpu-tile-and-distribute", "mlir::FunctionOpInterface"> {
  let summary = "Pass to tile and distribute linalg ops within a workgroup.";
}

def LLVMGPUVectorDistributePass :
    InterfacePass<"iree-llvmgpu-vector-distribute", "mlir::FunctionOpInterface"> {
  let summary = "Pass to distribute vectorized functions.";
}

def LLVMGPUVectorLoweringPass :
    InterfacePass<"iree-llvmgpu-vector-lowering", "mlir::FunctionOpInterface"> {
  let summary = "Pass to lower Vector ops before conversion to LLVM.";
}

def LLVMGPUVectorToGPUPass :
    InterfacePass<"iree-llvmgpu-vector-to-gpu", "mlir::FunctionOpInterface"> {
  let summary = "Pass to convert vector to gpu.";
}

//------------------------------------------------------------------------------
// Test Passes
//------------------------------------------------------------------------------

def TestLLVMGPUScalarizeMathOpPass :
    Pass<"iree-test-llvmgpu-legalize-ops", "ModuleOp"> {
  let summary = "Test pass for several legalization patterns.";
}

def TestLLVMGPUQueryMMAPass :
    Pass<"iree-test-llvmgpu-query-mma", "ModuleOp"> {
  let summary = "Test pass for querying the supported mma intrinsic instructions.";
}

#endif // IREE_CODEGEN_LLVMGPU_PASSES

// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CONVERSION_PASSES
#define IREE_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

//------------------------------------------------------------------------------
// Common/misc passes
//------------------------------------------------------------------------------

def CleanupBufferAllocView :
    Pass<"iree-codegen-cleanup-buffer-alloc-view", "FuncOp"> {
  let summary = "Performs cleanups over HAL interface/buffer allocation/view operations";
  let constructor = "mlir::iree_compiler::createCleanupBufferAllocViewPass()";
}

def DemoteF32ToF16 :
    Pass<"iree-convert-f32-to-f16", "ModuleOp"> {
  let summary = "Convert f32 operations and values into equivalent f16 ones.";
  let constructor = "mlir::iree_compiler::createDemoteF32ToF16Pass()";
}

def FlattenMemRefSubspan :
  Pass<"iree-codegen-flatten-memref-subspan", "ModuleOp"> {
  let summary = "Flatten n-D MemRef subspan ops to 1-D ones and fold byte offsets on subspan ops to the consumer load/store ops";
  let constructor = "mlir::iree_compiler::createFlattenMemRefSubspanPass()";
}

def FoldTensorExtractOp :
  Pass<"iree-codegen-fold-tensor-extract-op", ""> {
  let summary = "Fold `tensor.extract` operations prior to lowering to LLVM";
  let constructor = "mlir::iree_compiler::createFoldTensorExtractOpPass()";
}

def ForOpCanonicalization :
  Pass<"iree-codegen-canonicalize-scf-for", "FuncOp"> {
  let summary = "An ad-hoc pass to canonicalize selected loop-carried values and dependencies around scf.for";
  let constructor = "mlir::iree_compiler::createForOpCanonicalizationPass()";
}

def LinalgBufferize :
    Pass<"iree-codegen-linalg-bufferize", "FuncOp"> {
  let summary = "Convert from to Linalg ops on tensors to buffers";
  let constructor = "mlir::iree_compiler::createLinalgBufferizePass(nullptr)";
}

def OptimizeVectorTransfer :
    Pass<"iree-codegen-optimize-vector-transfer", "FuncOp"> {
  let summary = "Run optimization transformations on vector transfer operations";
  let constructor = "mlir::iree_compiler::createOptimizeVectorTransferPass()";
}

// TODO: Consider removing or moving to HAL/Transforms in order to avoid
// polluting common pass declarations with HAL specific ops.
def SetNumWorkgroups :
    Pass<"iree-set-num-workgroups", "mlir::iree_compiler::IREE::HAL::ExecutableTargetOp"> {
  let summary = "Set the number of workgroups to use for every entry point function in the dispatch region";
  let constructor = "mlir::iree_compiler::createSetNumWorkgroupsPass()";
}

//------------------------------------------------------------------------------
// LinalgToLinalg
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// LinalgToLLVM
//------------------------------------------------------------------------------

def LinalgToLLVMWorkgroupsVectorization :
    Pass<"iree-codegen-linalg-to-llvm-workgroups-vectorization-pass", "FuncOp"> {
  let summary = "Tile and vectorize llvm workgroups";
  let constructor = "mlir::iree_compiler::createLinalgToLLVMWorkgroupsVectorizationPass()";
}

def ConvertToLLVM :
    Pass<"iree-codegen-convert-to-llvm", "ModuleOp"> {
  let summary = "Perform final conversion from Linalg/HAL/Shape/Vector/Standard to LLVMIR dialect";
  let constructor = "mlir::iree_compiler::createConvertToLLVMPass()";
}

def LinalgVectorization :
    Pass<"iree-codegen-linalg-vectorization-pass", "FuncOp"> {
  let summary = "Vectorize linalg ops";
  let constructor = "mlir::iree_compiler::createLinalgVectorizePass()";
}

def LowerExecutableTarget :
    Pass<"iree-lower-executable-target-pass", "mlir::iree_compiler::IREE::HAL::ExecutableTargetOp"> {
  let summary = "Perform lowering of executable target using one of the IREE::HAL::DispatchLoweringPassPipeline";
  let constructor = "mlir::iree_compiler::createLowerExecutableTargetPass()";
}

def LLVMPadLinalgWorkgroupTiles :
    Pass<"iree-codegen-llvm-pad-linalg-workgroup-tiles", "FuncOp"> {
  let summary = "Padding linalg workgroup tiles into an integer multiple of tiling parameters.";
  let constructor = "mlir::iree_compiler::createLLVMPadLinalgWorkgroupTilesPass()";
}

def LinalgToLLVMPlanConvLoopOrder :
    Pass<"iree-codegen-linalg-to-llvm-plan-conv-loop-order", "FuncOp"> {
  let summary = "Convert linalg.conv to linalg.generic with a CPU-friendly iterator order";
  let constructor = "mlir::iree_compiler::createLinalgToLLVMPlanConvLoopOrderPass()";
}

def LinalgToLLVMTilePadAndVectorizeWorkgroups :
    Pass<"iree-codegen-linalg-to-llvm-tile-pad-and-vectorize-workgroups", "FuncOp"> {
  let summary = "Tile and pad workgroups tiles";
  let constructor = "mlir::iree_compiler::createLinalgToLLVMTilePadAndVectorizeWorkgroupsPass()";
}

def LinalgToLLVMUnfuseFMAOps :
    Pass<"iree-codegen-linalg-to-llvm-unfuse-fma-pass", "FuncOp"> {
  let summary = "Convert llvm.fma into unfused mulf and addf ops";
  let constructor = "mlir::iree_compiler::createLinalgToLLVMUnfuseFMAOpsPass()";
}

//------------------------------------------------------------------------------
// LinalgToLLVMGPU
//------------------------------------------------------------------------------

def LinalgToLLVMGPUVectorization :
    Pass<"iree-codegen-llvmgpu-vectorization", "FuncOp"> {
  let summary = "Pass to convert linalg into Vector.";
  let constructor = "mlir::iree_compiler::createLinalgToLLVMGPUVectorizationPass()";
}

def LinalgToLLVMGPUTileAndDistribute :
    Pass<"iree-codegen-llvmgpu-tile-and-distribute", "mlir::iree_compiler::IREE::HAL::ExecutableTargetOp"> {
  let summary = "Pass to tile and distribute linalg ops within a workgroup.";
  let constructor = "mlir::iree_compiler::createLinalgToLLVMGPUTileAndDistributeToThreads()";
}

def LinalgToLLVMGPURemoveSingleIterationLoop :
    Pass<"iree-llvmgpu-remove-single-iteration-loop", "FuncOp"> {
  let summary = "Remove distributed loop with single iteration.";
  let constructor = "mlir::iree_compiler::createLinalgToLLVMGPURemoveSingleIterationLoopPass()";
}

// TODO: Bring the argument in line with the names used elsewhere.
def LinalgToLLVMGPULowerExecutableTarget :
    Pass<"iree-lower-executable-target-gpu-pass", "mlir::iree_compiler::IREE::HAL::ExecutableTargetOp"> {
  let summary = "Perform lowering of executable target using one of the IREE::HAL::DispatchLoweringPassPipeline";
  let constructor = "mlir::iree_compiler::createLinalgToLLVMGPULowerExecutableTargetPass()";
}

// TODO: Bring the argument in line with the names used elsewhere.
def LinalgToLLVMGPUConvertToROCDL :
    Pass<"iree-codegen-convert-to-rocdl", "ModuleOp"> {
  let summary = "Perform final conversion from builtin/GPU/HAL/standard dialect to LLVM "
    "and ROCDL dialects";
  let constructor = "mlir::iree_compiler::createLinalgToLLVMGPUConvertToROCDLPass()";
}

// TODO: Bring the argument in line with the names used elsewhere.
def LinalgToLLVMGPUConvertToNVVM :
    Pass<"iree-codegen-convert-to-nvvm", "ModuleOp"> {
  let summary = "Perform final conversion from builtin/GPU/HAL/standard dialect to LLVM "
    "and NVVM dialects";
  let constructor = "mlir::iree_compiler::createLinalgToLLVMGPUConvertToNVVMPass()";
}

//------------------------------------------------------------------------------
// LinalgToSPIRV
//------------------------------------------------------------------------------

def LinalgToSPIRVConcretizeTileAmongWorkgroups :
    Pass<"iree-spirv-concretize-tile-among-workgroups", "mlir::iree_compiler::IREE::HAL::ExecutableTargetOp"> {
  let summary = "Replace hal.interface.workgroup.* ops with constant values from chosen tiling and distribution scheme";
  let constructor = "mlir::iree_compiler::createLinalgToSPIRVConcretizeTileAmongWorkgroupsPass(mlir::iree_compiler::SPIRVCodegenOptions::getFromCLOptions())";
}

// TODO: Rename argument to be fully qualified.
def LinalgToSPIRVConvertToGPU :
    Pass<"iree-codegen-convert-to-gpu", "mlir::iree_compiler::IREE::HAL::ExecutableTargetOp"> {
  let summary = "Map tiled linalg and loop ops to GPU";
  let constructor = "mlir::iree_compiler::createLinalgToSPIRVConvertToGPUPass()";
}

// TODO: Rename argument to be fully qualified.
def LinalgToSPIRVConvertToSPIRV :
    Pass<"iree-codegen-convert-to-spirv", "ModuleOp"> {
  let summary = "Perform final conversion from builtin/GPU/HAL/standard dialect to SPIR-V dialect";
  let constructor = "mlir::iree_compiler::createLinalgToSPIRVConvertToSPIRVPass()";
}

// TODO: Rename argument to be fully qualified.
// TODO: Does not appear used?
def LinalgToSPIRVFoldProcessorIDUses :
    Pass<"iree-codegen-fold-gpu-procid-uses", "mlir::iree_compiler::IREE::HAL::ExecutableTargetOp"> {
  let summary = "Fold GPU processor ID uses where possible";
  let constructor = "mlir::iree_compiler::createLinalgToSPIRVFoldProcessorIDUsesPass()";
}

// TODO: Rename argument to be fully qualified.
def LinalgToSPIRVTileAndVectorizeOneWorkgroup :
    Pass<"iree-spirv-tile-and-vectorize-in-one-workgroup", "mlir::iree_compiler::IREE::HAL::ExecutableTargetOp"> {
  let summary = "Tile and vectorize Linalg operations on buffers in one workgroup";
  let constructor = "mlir::iree_compiler::createLinalgToSPIRVTileAndVectorizeOneWorkgroupPass(mlir::iree_compiler::SPIRVCodegenOptions::getFromCLOptions())";
}

// TODO: Rename argument to be fully qualified.
def LinalgToSPIRVVectorizeMemRefLoadStore :
    Pass<"iree-spirv-vectorize-memref-load-store", "ModuleOp"> {
  let summary = "Vectorize interface memrefs and their load/store for better memory access";
  let constructor = "mlir::iree_compiler::createLinalgToSPIRVVectorizeMemRefLoadStore()";
}

// TODO: Rename argument to be fully qualified.
def LinalgToSPIRVVectorToCooperativeMatrix :
    Pass<"iree-spirv-vector-to-cooperative-matrix", "FuncOp"> {
  let summary = "Convert vector load/store/arithmetic ops to cooperative matrix ops when possible";
  let constructor = "mlir::iree_compiler::createLinalgToSPIRVVectorToCooperativeMatrixPass()";
}

// TODO: Rename argument to be fully qualified.
def LinalgToSPIRVConvertVectorToGPU :
    Pass<"iree-codegen-vector-to-gpu", "FuncOp"> {
  let summary = "Convert vector dialect to gpu subgroup level GPU instructions";
  let constructor = "mlir::iree_compiler::createLinalgToSPIRVConvertVectorToGPU()";
}

//------------------------------------------------------------------------------
// LinalgToVector Passes
//------------------------------------------------------------------------------

// TODO: Rename argument to be fully qualified.
def LinalgToVectorVectorizeConv :
    Pass<"iree-codegen-vectorize-linalg-conv", "FuncOp"> {
  let summary = "Vectorize a very specific form of linalg.conv";
  let constructor = "mlir::iree_compiler::createLinalgToVectorVectorizeConvPass();";
}

//------------------------------------------------------------------------------
// Target specific VectorToLLVM Passes
//------------------------------------------------------------------------------

def VectorToAArch64InlineAsm :
    Pass<"iree-codegen-vector-to-aarch64-inline-asm", "FuncOp"> {
  let summary = "Convert vector operations to aarch64 inline asm LLVMIR dialect";
  let constructor = "mlir::iree_compiler::createVectorToAArch64InlineAssemblyPass()";
}

//------------------------------------------------------------------------------
// Test passes
//------------------------------------------------------------------------------

def TestLinalgToLLVMGPUScalarizeMathOp :
    Pass<"iree-test-llvmgpu-scalarize-math-op", "FuncOp"> {
  let summary = "Test pass for scalarization patterns.";
  let constructor = "mlir::iree_compiler::createTestLinalgToLLVMGPUScalarizeMathOpPass()";
}

#endif  // IREE_DIALECT_FLOW_PASSES

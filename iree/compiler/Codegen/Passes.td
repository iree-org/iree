// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_PASSES
#define IREE_CODEGEN_PASSES

include "mlir/Pass/PassBase.td"

//------------------------------------------------------------------------------
// Common/misc passes
//------------------------------------------------------------------------------

def CleanupBufferAllocView :
    Pass<"iree-codegen-cleanup-buffer-alloc-view", "FuncOp"> {
  let summary =
      "Performs cleanups over HAL interface/buffer allocation/view operations";
  let constructor = "mlir::iree_compiler::createCleanupBufferAllocViewPass()";
}

def ConvertToDestinationPassingStyle :
    Pass<"iree-codegen-convert-to-destination-passing-style", "FuncOp"> {
  let summary =
      "Transforms the code to make the dispatch use destination-passing style";
  let constructor = "mlir::iree_compiler::createConvertToDestinationPassingStylePass()";
}

def DemoteF32ToF16 :
    Pass<"iree-convert-f32-to-f16", "ModuleOp"> {
  let summary = "Convert f32 operations and values into equivalent f16 ones.";
  let constructor = "mlir::iree_compiler::createDemoteF32ToF16Pass()";
}

def FlattenMemRefSubspan :
  Pass<"iree-codegen-flatten-memref-subspan", "ModuleOp"> {
  let summary =
      "Flatten n-D MemRef subspan ops to 1-D ones and fold byte offsets";
  let constructor = "mlir::iree_compiler::createFlattenMemRefSubspanPass()";
}

def FoldAffineMinInDistributedLoops :
  Pass<"iree-codegen-fold-affinemin-in-distributed-loops", "FuncOp"> {
  let summary = "Fold `affine.min` ops in distributed loops";
  let constructor = "mlir::iree_compiler::createFoldAffineMinInDistributedLoopsPass()";
}

def FoldTensorExtractOp :
  Pass<"iree-codegen-fold-tensor-extract-op", ""> {
  let summary = "Fold `tensor.extract` operations prior to lowering to LLVM";
  let constructor = "mlir::iree_compiler::createFoldTensorExtractOpPass()";
}

def ForOpCanonicalization :
  Pass<"iree-codegen-canonicalize-scf-for", "FuncOp"> {
  let summary =
      "Adhoc canonicalization of selected loop-carried values/dependencies for scf.for ops";
  let constructor = "mlir::iree_compiler::createForOpCanonicalizationPass()";
}

def LinalgBufferize :
    Pass<"iree-codegen-linalg-bufferize", "FuncOp"> {
  let summary = "Convert from to Linalg ops on tensors to buffers";
  let constructor = "mlir::iree_compiler::createLinalgBufferizePass(nullptr)";
}

def IREEComprehensiveBufferize :
    Pass<"iree-codegen-iree-comprehensive-bufferize", "ModuleOp"> {
  let summary = "Convert from to Linalg ops on tensors to buffers";
  let constructor = "mlir::iree_compiler::createIREEComprehensiveBufferizePass()";
}

def OptimizeVectorTransfer :
    Pass<"iree-codegen-optimize-vector-transfer", "FuncOp"> {
  let summary =
      "Run optimization transformations on vector transfer operations";
  let constructor = "mlir::iree_compiler::createOptimizeVectorTransferPass()";
}

def RemoveSingleIterationLoop :
    Pass<"iree-codegen-remove-single-iteration-loop", "FuncOp"> {
  let summary = "Remove distributed loop with single iteration.";
  let constructor = "mlir::iree_compiler::createRemoveSingleIterationLoopPass()";
}

// TODO: Consider removing or moving to HAL/Transforms in order to avoid
// polluting common pass declarations with HAL specific ops.
def SetNumWorkgroups :
    Pass<"iree-set-num-workgroups",
         "mlir::iree_compiler::IREE::HAL::ExecutableVariantOp"> {
  let summary =
      "Set the number of workgroups for entry point functions";
  let constructor = "mlir::iree_compiler::createSetNumWorkgroupsPass()";
}

// TODO: Rename argument to be fully qualified.
def LinalgToVectorVectorizeConv :
    Pass<"iree-codegen-vectorize-linalg-conv", "FuncOp"> {
  let summary = "Vectorize a very specific form of linalg.conv";
  let constructor =
      "mlir::iree_compiler::createLinalgToVectorVectorizeConvPass()";
}

def LinalgToVectorVectorizeMMT4d :
    Pass<"iree-codegen-vectorize-linalg-mmt4d", "FuncOp"> {
  let summary = "Vectorize a very specific form of linalg.mmt4d";
  let constructor =
      "mlir::iree_compiler::createLinalgToVectorVectorizeMMT4dPass()";
}
//------------------------------------------------------------------------------
// LLVMCPU
//------------------------------------------------------------------------------

def ConvertToLLVM :
    Pass<"iree-convert-to-llvm", "ModuleOp"> {
  let summary =
      "Perform final conversion from Linalg/HAL/Shape/Vector/Standard to LLVMIR dialect";
  let constructor = "mlir::iree_compiler::createConvertToLLVMPass()";
}

def LLVMCPULowerExecutableTarget :
    Pass<"iree-llvmcpu-lower-executable-target",
         "mlir::iree_compiler::IREE::HAL::ExecutableVariantOp"> {
  let summary =
      "Lower executable target using an IREE::HAL::DispatchLoweringPassPipeline";
  let constructor =
      "mlir::iree_compiler::createLLVMCPULowerExecutableTargetPass()";
}

def LLVMCPUSynchronizeSymbolVisibility :
    Pass<"iree-llvmcpu-synchronize-symbol-visibility", "ModuleOp"> {
  let summary = "Synchronizes LLVM linkage with MLIR symbol visibility";
  let constructor = "mlir::iree_compiler::createLLVMCPUSynchronizeSymbolVisibilityPass()";
}

def LLVMCPUTileAndVectorize
    : Pass<"iree-llvmcpu-tile-and-vectorize", "FuncOp"> {
  let summary = "Tile and vectorize workgroups tiles";
  let constructor = "mlir::iree_compiler::createLLVMCPUTileAndVectorizePass()";
}

def LLVMCPUTileFuseAndVectorize
    : Pass<"iree-llvmcpu-tile-fuse-and-vectorize", "FuncOp"> {
  let summary = "Tile, fuse and vectorize Linalg ops";
  let constructor =
      "mlir::iree_compiler::createLLVMCPUTileFuseAndVectorizePass()";
}

def LLVMCPUUnfuseFMAOps :
    Pass<"iree-llvmcpu-unfuse-fma-pass", "FuncOp"> {
  let summary = "Convert llvm.fma into unfused mulf and addf ops";
  let constructor = "mlir::iree_compiler::createLLVMCPUUnfuseFMAOpsPass()";
}

def LLVMCPUVectorization :
    Pass<"iree-llvmcpu-vectorization", "FuncOp"> {
  let summary = "Tile and vectorize for CPU backends";
  let constructor = "mlir::iree_compiler::createLLVMCPUVectorizationPass()";
}

def VectorToAArch64InlineAsm :
    Pass<"iree-llvmcpu-vector-to-aarch64-inline-asm", "FuncOp"> {
  let summary = "Convert vector operations to aarch64 inline asm LLVMIR dialect";
  let constructor = "mlir::iree_compiler::createVectorToAArch64InlineAssemblyPass()";
}

//------------------------------------------------------------------------------
// LLVMGPU
//------------------------------------------------------------------------------

// TODO: Bring the argument in line with the names used elsewhere.
def ConvertToROCDL :
    Pass<"iree-convert-to-rocdl", "ModuleOp"> {
  let summary = "Perform final conversion from builtin/GPU/HAL/standard dialect to LLVM "
    "and ROCDL dialects";
  let constructor = "mlir::iree_compiler::createConvertToROCDLPass()";
}

// TODO: Bring the argument in line with the names used elsewhere.
def ConvertToNVVM :
    Pass<"iree-convert-to-nvvm", "ModuleOp"> {
  let summary = "Perform final conversion from builtin/GPU/HAL/standard dialect to LLVM "
    "and NVVM dialects";
  let constructor = "mlir::iree_compiler::createConvertToNVVMPass()";
}

// TODO: Bring the argument in line with the names used elsewhere.
def LLVMGPULowerExecutableTarget :
    Pass<"iree-llvmgpu-lower-executable-target-pass", "mlir::iree_compiler::IREE::HAL::ExecutableVariantOp"> {
  let summary = "Perform lowering of executable target using one of the IREE::HAL::DispatchLoweringPassPipeline";
  let constructor = "mlir::iree_compiler::createLLVMGPULowerExecutableTargetPass()";
}

def LLVMGPUTileAndDistribute :
    Pass<"iree-llvmgpu-tile-and-distribute", "FuncOp"> {
  let summary = "Pass to tile and distribute linalg ops within a workgroup.";
  let constructor = "mlir::iree_compiler::createLLVMGPUTileAndDistributeToThreads()";
}

def LLVMGPUVectorization :
    Pass<"iree-llvmgpu-vectorization", "FuncOp"> {
  let summary = "Pass to convert linalg into Vector.";
  let constructor = "mlir::iree_compiler::createLLVMGPUVectorizationPass()";
}

def LLVMGPUVectorLowering :
    Pass<"iree-llvmgpu-vector-lowering", "FuncOp"> {
  let summary = "Pass to lower Vector ops before conversion to LLVM.";
  let constructor = "mlir::iree_compiler::createLLVMGPUVectorLoweringPass()";
}

def LLVMGPUDistributeSharedMemoryCopy :
    Pass<"iree-llvmgpu-distribute-shared-memory-copy", "FuncOp"> {
  let summary = "Pass to distribute shared memory copies to threads.";
  let constructor = "mlir::iree_compiler::createLLVMGPUDistributeSharedMemoryCopy()";
}

def LLVMGPUPipelining :
    Pass<"iree-llvmgpu-pipelining", "FuncOp"> {
  let summary = "Pass to do software pipelining.";
  let constructor = "mlir::iree_compiler::createLLVMGPUPipeliningPass()";
}

//------------------------------------------------------------------------------
// SPIR-V
//------------------------------------------------------------------------------

// TODO: Rename argument to be fully qualified.
def ConvertToSPIRV : Pass<"iree-convert-to-spirv", "ModuleOp"> {
  let summary = "Perform the final conversion to SPIR-V dialect";
  let constructor = "mlir::iree_compiler::createConvertToSPIRVPass()";
}

def SPIRVLowerExecutableTarget :
    Pass<"iree-spirv-lower-executable-target-pass",
         "mlir::iree_compiler::IREE::HAL::ExecutableVariantOp"> {
  let summary = "Lower the executable target to SPIR-V using one of the "
                "IREE::HAL::DispatchLoweringPassPipeline";
  let constructor =
      "mlir::iree_compiler::createSPIRVLowerExecutableTargetPass()";
}

// TODO: Rename argument to be fully qualified.
def SPIRVTileAndDistribute : Pass<"iree-spirv-tile-and-distribute", "FuncOp"> {
  let summary = "Tile and distribute Linalg ops with buffer semantics to "
                "invocations";
  let constructor = "mlir::iree_compiler::createSPIRVTileAndDistributePass()";
}

def SPIRVTileAndVectorizeToCooperativeOps : Pass<
    "iree-spirv-tile-and-vectorize-to-cooperative-ops", "FuncOp"> {
  let summary = "Tile Linalg ops with buffer semantics to subgroups and "
                "vectorize to vector ops suitable for lowering to SPIR-V "
                "cooperative ops";
  let constructor =
    "mlir::iree_compiler::createSPIRVTileAndVectorizeToCooperativeOpsPass()";
}

def SPIRVVectorize : Pass<"iree-spirv-vectorize", "FuncOp"> {
  let summary = "Vectorize Linalg ops with buffer semantics";
  let constructor = "mlir::iree_compiler::createSPIRVVectorizePass()";
}

def SPIRVVectorToCooperativeOps :
    Pass<"iree-spirv-vector-to-cooperative-ops", "FuncOp"> {
  let summary = "Convert vector ops to SPIR-V cooperative ops";
  let constructor =
      "mlir::iree_compiler::createSPIRVVectorToCooperativeOpsPass()";
}

// TODO: Rename argument to be fully qualified.
def SPIRVVectorizeLoadStore :
    Pass<"iree-spirv-vectorize-load-store", "ModuleOp"> {
  let summary = "Vectorize load/store of memrefs for better memory access";
  let constructor = "mlir::iree_compiler::createSPIRVVectorizeLoadStore()";
}

// TODO: Rename argument to be fully qualified.
def SPIRVCopyToWorkgroupMemory :
    Pass<"iree-spirv-copy-to-workgroup-memory", "FuncOp"> {
  let summary = "Lower linalg.copy for copying data to workgroup memory";
  let constructor =
      "mlir::iree_compiler::createSPIRVCopyToWorkgroupMemoryPass()";
}

//------------------------------------------------------------------------------
// Test passes
//------------------------------------------------------------------------------

def TestLLVMGPUScalarizeMathOp :
    Pass<"iree-test-llvmgpu-legalize-ops", "ModuleOp"> {
  let summary = "Test pass for several legalization patterns.";
  let constructor = "mlir::iree_compiler::createTestLLVMGPULegalizePass()";
}

#endif  // IREE_DIALECT_FLOW_PASSES

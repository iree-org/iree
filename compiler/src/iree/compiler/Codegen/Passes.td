// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_PASSES
#define IREE_CODEGEN_PASSES

include "mlir/Pass/PassBase.td"

//------------------------------------------------------------------------------
// LLVMCPU
//------------------------------------------------------------------------------

def VerifyLinalgTransformLegality :
    Pass<"iree-llvmcpu-verify-linalg-transform-legality", "ModuleOp"> {
  let summary = "Verify that only supported IR constructs are passed to the compiler.";
  let constructor = "mlir::iree_compiler::createVerifyLinalgTransformLegalityPass()";
}

def LLVMCPUTile :
    Pass<"iree-llvmcpu-tile", "func::FuncOp"> {
  let summary = "Pass to tile TilingInterface operations.";
  let constructor =
      "mlir::iree_compiler::createLLVMCPUTilePass()";
  let options = [
    Option<"tilingLevel", "tiling-level", "int64_t", /*default=*/"-1",
      "Use default tiling level used to retrieve the configuration from lowering_config">
  ];
}

def LLVMCPUTileAndFuse :
    Pass<"iree-llvmcpu-tile-and-fuse", "func::FuncOp"> {
  let summary = "Pass to tile and fuse TilingInterface operations.";
  let constructor =
      "mlir::iree_compiler::createLLVMCPUTileAndFusePass()";
  let options = [
    Option<"tilingLevel", "tiling-level", "int64_t", /*default=*/"-1",
      "Use default tiling level used to retrieve the configuration from lowering_config">
  ];
}

def LLVMCPUTensorPad :
    Pass<"iree-llvmcpu-tensor-pad", "func::FuncOp"> {
  let summary = "Pass to pad operations on tensors in top-down order.";
  let constructor =
      "mlir::iree_compiler::createLLVMCPUTensorPadPass()";
}

def LLVMCPUPeel :
    Pass<"iree-llvmcpu-peel", "func::FuncOp"> {
  let summary = "Pass to perform peeling on non-distributed loops.";
  let constructor =
      "mlir::iree_compiler::createLLVMCPUPeelPass()";
}

def LLVMCPUSplitReduction : Pass<"iree-llvmcpu-split-reduction", "func::FuncOp"> {
  let summary = "Pass to splitReduce linalg operations.";
  let constructor = "mlir::iree_compiler::createLLVMCPUSplitReductionPass()";
  let options = [
    Option<"enableFpReductionReordering", "enable-fp-reduction-reordering",
      "bool", /*default=*/"false",
      "Flag to enable reduction reordering on floating points.">,
  ];
}

def LLVMCPUVectorization :
    Pass<"iree-llvmcpu-vectorization", "func::FuncOp"> {
  let summary = "Pass to perform vectorization on tensor/linalg ops.";
  let options = [
    Option<"enableVectorMasking", "enable-vector-masking", "bool",/*default=*/"false",
      "Enable vector masking during vectorization.">,
    Option<"vectorizePadding", "vectorize-padding", "bool", /*default=*/"false",
      "Rewrite all tensor.pad ops in the function to vector form.">,
    Option<"vectorizeGatherAccesses", "vectorize-gather-accesses", "bool", /*default=*/"false",
      "Enable vectorizaiton of operations that may generate vector.gather operations.">,
  ];
  let constructor =
      "mlir::iree_compiler::createLLVMCPUVectorizationPass()";
}

def LLVMCPUVectorLowering :
    Pass<"iree-llvmcpu-vector-lowering", "func::FuncOp"> {
  let summary = "Pass to lower Vector ops before conversion to LLVM.";
  let options = [
    Option<"splitVectorTransfersTo", "split-transfers", "std::string",
      /*default=*/"",
      [{Split vector transfers between slow (masked) and fast "
        "(unmasked) variants. Possible options are:\n"
          "\tnone [default]: keep unsplit vector.transfer and pay the price\n"
          "\tlinalg-copy: use linalg.fill + linalg.generic for the slow path\n"
          "\tvector-transfers: use extra small unmasked vector.transfers for"
          " the slow path\n}]>,
    Option<"lowerVectorTransposeToAVX2", "lower-vector-transpose-to-avx2", "bool",
      /*default=*/"false",
      "Add specific transpose to avx2 lowering patterns.">,
  ];
  let constructor =
      "mlir::iree_compiler::createLLVMCPUVectorLoweringPass()";
}

def ConvertToLLVM :
    Pass<"iree-convert-to-llvm", "ModuleOp"> {
  let summary =
      "Perform final conversion from Linalg/HAL/Shape/Vector/Standard to LLVMIR dialect";
  let constructor = "mlir::iree_compiler::createConvertToLLVMPass()";
  let options = [
    Option<"reassociateFpReductions", "reassociateFpReductions", "bool",
            /*default=*/"false",
           "Specifies if FP add and mult reductions can be reordered">,
  ];
}

def LLVMCPUEmitVectorizationRemarks :
    Pass<"iree-llvmcpu-emit-vectorization-remarks", "func::FuncOp"> {
  let summary = "Emit vectorization remarks on Linalg ops";
  let constructor =
      "mlir::iree_compiler::createLLVMCPUEmitVectorizationRemarksPass()";
}

def LLVMCPUCheckIRBeforeLLVMConversion :
    Pass<"iree-llvmcpu-check-ir-before-llvm-conversion", "ModuleOp"> {
  let summary = "Checks CPU backend specific IR constraints (like no allocas)";
  let constructor = "mlir::iree_compiler::createLLVMCPUCheckIRBeforeLLVMConversionPass()";
}

def LLVMCPULowerExecutableTarget :
    Pass<"iree-llvmcpu-lower-executable-target",
         "mlir::iree_compiler::IREE::HAL::ExecutableVariantOp"> {
  let summary =
      "Lower executable target using an IREE::HAL::DispatchLoweringPassPipeline";
  let constructor =
      "mlir::iree_compiler::createLLVMCPULowerExecutableTargetPass()";
}

def ExpandArithF16ToF32 :
    Pass<"iree-llvmcpu-expand-f16-op-to-f32", ""> {
  let summary =
      "Preform f16 opertaions by expanding them to f32.";
  let constructor =
      "mlir::iree_compiler::createExpandF16OpToF32Pass()";
}

def LLVMCPULowerToUKernels :
    Pass<"iree-llvmcpu-lower-to-ukernels", ""> {
  let summary =
      "Separate out parts of the IR that lower to a micro-kernel";
  let constructor =
      "mlir::iree_compiler::createLLVMCPULowerToUKernelsPass()";
}

def LLVMCPUMaterializeEncoding :
    Pass<"iree-llvmcpu-materialize-encoding", "func::FuncOp"> {
  let summary = "Materialize the encoding for tensor as specified by the backend";
  let constructor = "mlir::iree_compiler::createLLVMCPUMaterializeEncodingPass()";
}

def LLVMCPUSynchronizeSymbolVisibility :
    Pass<"iree-llvmcpu-synchronize-symbol-visibility", "ModuleOp"> {
  let summary = "Synchronizes LLVM linkage with MLIR symbol visibility";
  let constructor = "mlir::iree_compiler::createLLVMCPUSynchronizeSymbolVisibilityPass()";
}

def LLVMCPUMmt4dVectorLowering
    : Pass<"iree-llvmcpu-mmt4d-vector-lowering", "func::FuncOp"> {
  let summary = "Apply vector lowering logic to vector ops";
  let constructor =
      "mlir::iree_compiler::createLLVMCPUMmt4dVectorLoweringPass()";
}

def LLVMCPUUnfuseFMAOps :
    Pass<"iree-llvmcpu-unfuse-fma-pass", "func::FuncOp"> {
  let summary = "Convert llvm.fma into unfused mulf and addf ops";
  let constructor = "mlir::iree_compiler::createLLVMCPUUnfuseFMAOpsPass()";
}

def VectorContractCustomKernels :
    Pass<"iree-llvmcpu-vector-contract-custom-kernels", "func::FuncOp"> {
  let summary = "Enable custom kernels (inline assembly or intrinsics) for some vector.contract ops";
  let constructor = "mlir::iree_compiler::createVectorContractCustomKernelsPass()";
}

def LLVMCPULinkExecutables :
    Pass<"iree-llvmcpu-link-executables", "mlir::ModuleOp"> {
  let summary = "Links LLVMCPU HAL executables within the top-level program module.";
  let constructor = "mlir::iree_compiler::createLLVMCPULinkExecutablesPass()";
}

def LLVMCPUAssignConstantOrdinals :
    Pass<"iree-llvmcpu-assign-constant-ordinals", "IREE::HAL::ExecutableVariantOp"> {
  let summary = "Assigns executable constant ordinals across all LLVMCPU variants.";
  let constructor = "mlir::iree_compiler::createLLVMCPUAssignConstantOrdinalsPass()";
}

def LLVMCPUAssignImportOrdinals :
    Pass<"iree-llvmcpu-assign-import-ordinals", "IREE::HAL::ExecutableVariantOp"> {
  let summary = "Assigns executable import ordinals across all LLVMCPU variants.";
  let constructor = "mlir::iree_compiler::createLLVMCPUAssignImportOrdinalsPass()";
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

// TODO: Bring the argument in line with the names used elsewhere.
def LLVMGPULowerExecutableTarget :
    Pass<"iree-llvmgpu-lower-executable-target", "mlir::iree_compiler::IREE::HAL::ExecutableVariantOp"> {
  let summary = "Perform lowering of executable target using one of the IREE::HAL::DispatchLoweringPassPipeline";
  let constructor = "mlir::iree_compiler::createLLVMGPULowerExecutableTargetPass()";
}

def LLVMGPUCastAddressSpaceFunction :
    Pass<"iree-llvmgpu-cast-address-space-function", "ModuleOp"> {
  let summary = "Pass to cast";
  let constructor = "mlir::iree_compiler::createLLVMGPUCastAddressSpaceFunction()";
}

def LLVMGPUTileAndDistribute :
    Pass<"iree-llvmgpu-tile-and-distribute", "func::FuncOp"> {
  let summary = "Pass to tile and distribute linalg ops within a workgroup.";
  let constructor = "mlir::iree_compiler::createLLVMGPUTileAndDistribute()";
}

def LLVMGPUPackSharedMemoryAlloc :
    Pass<"iree-llvmgpu-pack-shared-memory-alloc", "func::FuncOp"> {
  let summary = "Pass pack shared memory allocation in order to reduce memory usage.";
  let constructor = "mlir::iree_compiler::createLLVMGPUPackSharedMemoryAlloc()";
}

def LLVMGPUTensorCoreVectorization :
    Pass<"iree-llvmgpu-tensorcore-vectorization", "func::FuncOp"> {
  let summary = "Pass to convert linalg into Vector and transform it to a form that can be lowered to GPU MMA ops";
  let constructor = "mlir::iree_compiler::createLLVMGPUTensorCoreVectorizationPass()";
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

def LLVMGPUTensorPad :
    Pass<"iree-llvmgpu-tensor-pad", "func::FuncOp"> {
  let summary = "Pass to pad out tensors up to static dimensions.";
  let constructor = "mlir::iree_compiler::createLLVMGPUTensorPadPass()";
}

//------------------------------------------------------------------------------
// SPIR-V
//------------------------------------------------------------------------------

def ConvertToSPIRV : Pass<"iree-convert-to-spirv", "ModuleOp"> {
  let summary = "Perform the final conversion to SPIR-V dialect";
  let constructor = "mlir::iree_compiler::createConvertToSPIRVPass()";
  let options = [
    Option<"enableFastMathOption", "enable-fast-math", "bool", /*default=*/"false",
          "Enable fast math mode during type conversion (i.e. assume no NaN/infinity)">,
    Option<"indexBitsOption", "index-bits", "unsigned", /*default=*/"32",
          "Specify the bit widths for SPIR-V indices">,
  ];
}

def SPIRVVectorToGPUSubgroupMMA :
    Pass<"iree-spirv-vector-to-gpu-subgroup-mma-ops", "func::FuncOp"> {
  let summary = "Pass to convert vector ops to GPU subgroup MMA ops.";
  let constructor = "mlir::iree_compiler::createSPIRVVectorToGPUSubgroupMMAOpsPass()";
}

def SPIRVLowerExecutableTarget :
    Pass<"iree-spirv-lower-executable-target-pass",
         "mlir::iree_compiler::IREE::HAL::ExecutableVariantOp"> {
  let summary = "Lower the executable target to SPIR-V using one of the "
                "IREE::HAL::DispatchLoweringPassPipeline";
  let constructor =
      "mlir::iree_compiler::createSPIRVLowerExecutableTargetPass()";
}

def SPIRVTile : Pass<"iree-spirv-tile", "func::FuncOp"> {
  let summary = "Tile Linalg ops with tensor semantics to invocations";
  let constructor = "mlir::iree_compiler::createSPIRVTilePass()";
}

def SPIRVDistribute : Pass<"iree-spirv-distribute", "func::FuncOp"> {
  let summary = "Distribute tiled loop nests to invocations";
  let constructor = "mlir::iree_compiler::createSPIRVDistributePass()";
}

def SPIRVTileAndDistribute : Pass<"iree-spirv-tile-and-distribute", "func::FuncOp"> {
  let summary = "Tile and distribute Linalg ops with buffer semantics to "
                "invocations";
  let constructor = "mlir::iree_compiler::createSPIRVTileAndDistributePass()";
}

def SPIRVTileToCooperativeOps : Pass<
    "iree-spirv-tile-to-cooperative-ops", "func::FuncOp"> {
  let summary = "Tile Linalg ops with buffer semantics to subgroups and "
                "vectorize to vector ops suitable for lowering to SPIR-V "
                "cooperative ops";
  let constructor =
    "mlir::iree_compiler::createSPIRVTileToCooperativeOpsPass()";
}

def SPIRVVectorizeToCooperativeOps : Pass<
    "iree-spirv-vectorize-to-cooperative-ops", "func::FuncOp"> {
  let summary = "Tile Linalg ops with buffer semantics to subgroups and "
                "vectorize to vector ops suitable for lowering to SPIR-V "
                "cooperative ops";
  let constructor =
    "mlir::iree_compiler::createSPIRVVectorizeToCooperativeOpsPass()";
}

def SPIRVTileAndPromote : Pass<"iree-spirv-tile-and-promote", "func::FuncOp"> {
  let summary = "Promote tiled Linalg ops with buffer semantics to use "
                "workgroup memory and then tile to invocations";
  let constructor =
    "mlir::iree_compiler::createSPIRVTileAndPromotePass()";
  let options = [
    Option<"promoteC", "promote-c", "bool", /*default=*/"false",
          "Promote C matrix to use shared memory">,
    Option<"skipThread", "skip-thread", "bool", /*default=*/"false",
          "Skip tiling and distributing to GPU threads">,
  ];
}

def SPIRVVectorize : Pass<"iree-spirv-vectorize", "func::FuncOp"> {
  let summary = "Vectorize Linalg ops with buffer semantics";
  let constructor = "mlir::iree_compiler::createSPIRVVectorizePass()";
}

def SPIRVVectorizeLoadStore :
    Pass<"iree-spirv-vectorize-load-store", "ModuleOp"> {
  let summary = "Vectorize load/store of memrefs for better memory access";
  let constructor = "mlir::iree_compiler::createSPIRVVectorizeLoadStore()";
}

def SPIRVBreakDownLargeVector : Pass<"iree-spirv-breakdown-large-vector",
                                     "func::FuncOp"> {
  let summary = "Break down vectors not natively supported by SPIR-V";
  let constructor = "mlir::iree_compiler::createSPIRVBreakDownLargeVectorPass()";
}

def SPIRVCreateFastSlowPath :
    Pass<"iree-spirv-create-fast-slow-path", "func::FuncOp"> {
  let summary = "Create separate fast and slow paths to handle padding";
  let constructor = "mlir::iree_compiler::createSPIRVCreateFastSlowPathPass()";
}

def SPIRVEmulateI64 :
    Pass<"iree-spirv-emulate-i64", "ModuleOp"> {
  let summary = "Emulate 64-bit integer ops with 32-bit integer ops";
  let constructor = "mlir::iree_compiler::createSPIRVEmulateI64Pass()";
}

def SPIRVEraseStorageBufferStaticShape :
    Pass<"iree-spirv-erase-storage-buffer-static-shape", "func::FuncOp"> {
  let summary = "Turn static shaped storage buffer subspan ops into dynamic shaped ones";
  let constructor = "mlir::iree_compiler::createSPIRVEraseStorageBufferStaticShapePass()";
}

def SPIRVMapMemRefStorageClass :
    Pass<"iree-spirv-map-memref-storage-class", "func::FuncOp"> {
  let summary = "Map MemRef memory spaces to SPIR-V storage classes";
  let constructor = "mlir::iree_compiler::createSPIRVMapMemRefStorageClassPass()";
}

def SPIRVAnnotateWinogradLoops : Pass<"iree-spirv-annotate-winograd-loops", "func::FuncOp"> {
  let summary = "Annotate innermost Winograd loops with spirv distribute attribute";
  let constructor = "mlir::iree_compiler::createSPIRVAnnotateWinogradLoopsPass()";
}

//------------------------------------------------------------------------------
// VMVX Passes
//------------------------------------------------------------------------------

def VMVXMaterializeEncoding :
    Pass<"iree-vmvx-materialize-encoding", "func::FuncOp"> {
  let summary = "Materialize the encoding for tensor as specified by the backend";
  let constructor = "mlir::iree_compiler::createVMVXMaterializeEncodingPass()";
}


def VMVXLowerLinalgMicrokernels :
    Pass<"iree-vmvx-lower-linalg-microkernels", ""> {
  let summary =
      "Lowers linalg ops to the VMVX microkernel library";
  let constructor = "mlir::iree_compiler::createVMVXLowerLinalgMicrokernelsPass()";
  let options = [
    Option<"warnOnUnconverted", "warn-on-unconverted", "bool",
           /*default=*/"false",
           "Warns on any unconverted linalg ops which remain live">
  ];
}

def VMVXLinkExecutables :
    Pass<"iree-vmvx-link-executables", "mlir::ModuleOp"> {
  let summary = "Links VMVX HAL executables within the top-level program module.";
  let constructor = "mlir::iree_compiler::createVMVXLinkExecutablesPass()";
}

def VMVXAssignConstantOrdinals :
    Pass<"iree-vmvx-assign-constant-ordinals", "IREE::HAL::ExecutableVariantOp"> {
  let summary = "Assigns executable constant ordinals across all VMVX variants.";
  let constructor = "mlir::iree_compiler::createVMVXAssignConstantOrdinalsPass()";
}

//------------------------------------------------------------------------------
// WGSL Passes
//------------------------------------------------------------------------------

def WGSLReplacePushConstants :
    Pass<"iree-wgsl-replace-push-constants", "func::FuncOp"> {
  let summary =
      "Replaces push constant loads with binding loads for when using "
      "WGSL without push constant support";
  let constructor = "mlir::iree_compiler::createWGSLReplacePushConstantsPass()";
}

//------------------------------------------------------------------------------
// Test Passes
//------------------------------------------------------------------------------

def TestLLVMGPUScalarizeMathOp :
    Pass<"iree-test-llvmgpu-legalize-ops", "ModuleOp"> {
  let summary = "Test pass for several legalization patterns.";
  let constructor = "mlir::iree_compiler::createTestLLVMGPULegalizePass()";
}

#endif  // IREE_DIALECT_FLOW_PASSES

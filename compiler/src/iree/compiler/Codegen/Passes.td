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
    Pass<"iree-codegen-cleanup-buffer-alloc-view", "func::FuncOp"> {
  let summary =
      "Performs cleanups over HAL interface/buffer allocation/view operations";
  let constructor = "mlir::iree_compiler::createCleanupBufferAllocViewPass()";
}

def ConvertToDestinationPassingStyle :
    Pass<"iree-codegen-convert-to-destination-passing-style", "func::FuncOp"> {
  let summary =
      "Transforms the code to make the dispatch use destination-passing style";
  let constructor = "mlir::iree_compiler::createConvertToDestinationPassingStylePass()";
  let options = [
    Option<"useWARForCooperativeMatrixCodegen", "use-war-for-cooperative-matrix-codegen",
           "bool", /*default=*/"false",
           "WAR for failure in Cooperative matrix codegen pipelines. See #10648.">
  ];
}

def DecomposeLinalgGeneric :
    Pass<"iree-codegen-decompose-linalg-generic", ""> {
  let summary = "Decomposes linalg generic ops into individual ops";
  let description = [{
    It is sometimes advantageous to operate on generic ops which contain
    at most one non-yield body operation. This is most often the case when
    needing to materialize individual ops (which some backends require).
    Note that this is often an extreme pessimization unless if part of a
    lowering flow which was designed for it.

    Operates on tensor based linalg ops.
  }];
  let constructor = "mlir::iree_compiler::createDecomposeLinalgGenericPass()";
}

def DecomposeAffineOps: Pass<"decompose-affine-ops"> {
  let summary = "Decompose `affine.apply` operations into sub `affine.apply`";
  let description = [{
    Decompose `affine.apply` operations into sub `affine.apply` where each
    sub expression references values that are defined in the same loop scope.
    The sub expression are then stitched back together following the loop
    nest order.

    The goal of this pass is to break down `affine.apply` expressions such
    that the resulting sub expressions can be hoisted out in their respective
    loop.

    E.g., Let's say we have
    ```mlir
    %res = affine.apply
             affine_map<()[s0, s1, s2] -> (s0 * 1024 + s1 * 32 + s2)>()
               [%loopVariant, %inv1, %inv2]
    ```
    Where `%inv1` and `%inv2` are loop invariant and `%loopVariant` is not.
    This will produce the following subexpressions:
    ```mlir
    // Loop invariant computations first.
    %inv1x32 =
      affine.apply affine_map<()[s0] -> (s0 * 32)>()[%inv1]
    %inv1x32_plus_inv2 =
      affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%inv1x32, %inv2]
    // Loop variant computation next.
    %loopVariantx1024 =
      affine.apply affine_map<()[s0] -> (s0 * 1024)>()[%loopVariant]
    // Compose things back together.
    %res =
      affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()
        [%loopVariant, %inv1x32_plus_inv2]
    ```
    Now the sequence of instructions leading to and including
    `%inv1x32_plus_inv2` can be hoisted out of the loop.

    This pass requires `scf.for` structures to still be around otherwise
    the break down will be meaningless.

    Note: The decomposition performed by this pass will be undone by
    canonicalization. Make sure to lower the resulting ops before that.
  }];
  let constructor = "mlir::iree_compiler::createDecomposeAffineOpsPass()";
  let dependentDialects = [
      "AffineDialect"
  ];
}

def ExtractAddressComputation: Pass<"extract-address-computation"> {
  let summary = "Extract address computations from memory accesses";
  let description = [{
    Extract the address computation from the instructions with memory
    accesses such that these memory accesses use only a base pointer.

    For instance,
    ```mlir
    memref.load %base[%off0, ...]
    ```

    Will be rewritten in:
    ```mlir
    %new_base = memref.subview %base[%off0,...][1,...][1,...]
    memref.load %new_base[%c0,...]
    ```
  }];
  let constructor = "mlir::iree_compiler::createExtractAddressComputationPass()";
  let dependentDialects = [
      "memref::MemRefDialect"
  ];
}

def ExtractAddressComputationGPU: Pass<"extract-address-computation-gpu"> {
  let summary = "Extract address computations from memory accesses";
  let description = [{
     This pass is similar to `extract-address-computation` except it also
     supports memory accesses that are specific to GPUs.
  }];
  let constructor = "mlir::iree_compiler::createExtractAddressComputationGPUPass()";
  let dependentDialects = [
      "memref::MemRefDialect", "nvgpu::NVGPUDialect", "AffineDialect"
  ];
}

def FixupSubspanWithOffsets :
    Pass<"iree-codegen-fixup-subspan-with-offsets", "func::FuncOp"> {
  let summary = "Fixes up `hal.interface.binding.subspan` ops with offsets";
  let constructor =
      "mlir::iree_compiler::createFixupSubspanWithOffsetsPass()";
  let options = [
    Option<"keepDeadSubspanOps", "keep-dead-subspan-ops",
            "bool", /*default=*/"false",
            "Flag to not remove dead subspan ops (used for testing only)">
  ];
}

def FlattenMemRefSubspan :
  Pass<"iree-codegen-flatten-memref-subspan", "ModuleOp"> {
  let summary =
      "Flatten n-D MemRef subspan ops to 1-D ones and fold byte offsets";
  let constructor = "mlir::iree_compiler::createFlattenMemRefSubspanPass()";
}

def FoldAffineMinInDistributedLoops :
  Pass<"iree-codegen-fold-affinemin-in-distributed-loops", "func::FuncOp"> {
  let summary = "Fold `affine.min` ops in distributed loops";
  let constructor = "mlir::iree_compiler::createFoldAffineMinInDistributedLoopsPass()";
}

def FoldTensorExtractOp :
  Pass<"iree-codegen-fold-tensor-extract-op", ""> {
  let summary = "Fold `tensor.extract` operations prior to lowering to LLVM";
  let constructor = "mlir::iree_compiler::createFoldTensorExtractOpPass()";
}

def ForOpCanonicalization :
  Pass<"iree-codegen-canonicalize-scf-for", "func::FuncOp"> {
  let summary =
      "Adhoc canonicalization of selected loop-carried values/dependencies for scf.for ops";
  let constructor = "mlir::iree_compiler::createForOpCanonicalizationPass()";
}

def BufferizeCopyOnlyDispatches :
  Pass<"iree-codegen-bufferize-copy-only-dispatches", "ModuleOp"> {
  let summary =
      "Bufferize dispatches that copy to/from interfaces to convert to a linalg.copy op";
  let constructor = "mlir::iree_compiler::createBufferizeCopyOnlyDispatchesPass()";
}

def EliminateEmptyTensors :
    Pass<"iree-eliminate-empty-tensors", "ModuleOp"> {
  let summary = "Eliminate tensor.empty ops to avoid buffer allocations";
  let constructor = "mlir::iree_compiler::createEliminateEmptyTensorsPass()";
}

def HoistStaticallyBoundAllocations :
    Pass<"iree-hoist-statically-bound-allocations", "func::FuncOp"> {
  let summary = "Hoist statically bound alloca ops to the entry block of functions";
  let constructor = "mlir::iree_compiler::createHoistStaticallyBoundAllocationsPass()";
}

def IREEComprehensiveBufferize :
    Pass<"iree-codegen-iree-comprehensive-bufferize", "ModuleOp"> {
  let summary = "Convert from to Linalg ops on tensors to buffers";
  let constructor = "mlir::iree_compiler::createIREEComprehensiveBufferizePass()";
  let options = [
    Option<"testAnalysisOnly", "test-analysis-only", "bool",
            /*default=*/"false",
           "Only runs inplaceability analysis (for testing purposes only)">,
    Option<"printConflicts", "print-conflicts", "bool",
            /*default=*/"false",
           "Annotates IR with RaW conflicts. Requires test-analysis-only.">,
  ];
}

def LowerUKernelOpsToCalls :
    Pass<"iree-codegen-lower-ukernel-ops-to-calls", "ModuleOp"> {
  let summary = "Lower micro-kernel wrapper ops into function calls";
  let constructor = "mlir::iree_compiler::createLowerUKernelOpsToCallsPass()";
}

def OptimizeVectorTransfer :
    Pass<"iree-codegen-optimize-vector-transfer", "func::FuncOp"> {
  let summary =
      "Run optimization transformations on vector transfer operations";
  let constructor = "mlir::iree_compiler::createOptimizeVectorTransferPass()";
  let options = [
    Option<"optionFlatten", "flatten", "bool", "false",
           "Flatten the vector type of vector transfers where possible (contiguous row-major data).">
  ];
}

def SplitFullPartialTransfer :
    Pass<"iree-codegen-split-full-partial-transfer", "func::FuncOp"> {
  let summary =
      "Split a vector.transfer operation into an in-bounds (i.e., no "
      "out-of-bounds masking) fastpath and a slowpath.";
  let constructor = "mlir::iree_compiler::createSplitFullPartialTransferPass()";
  let options = [
    Option<"splitVectorTransfersTo", "split-transfers", "std::string",
      /*default=*/"",
      [{Split vector transfers between slow (masked) and fast "
        "(unmasked) variants. Possible options are:\n"
          "\tnone [default]: keep unsplit vector.transfer and pay the price\n"
          "\tlinalg-copy: use linalg.fill + linalg.generic for the slow path\n"
          "\tvector-transfers: use extra small unmasked vector.transfers for"
          " the slow path\n}]>,
  ];
}

def TestExecutablePreprocessing :
    Pass<"iree-codegen-test-executable-preprocessing", ""> {
  let summary = "Tests iree-hal-preprocess-executables-with behavior.";
  let constructor = "mlir::iree_compiler::createTestExecutablePreprocessingPass()";
}

def TestPartitionableLoopsInterface :
    Pass<"iree-codegen-test-partitionable-loops-interface", ""> {
  let summary = "Test the PartitionableLoopsInterface";
  let constructor = "mlir::iree_compiler::createTestPartitionableLoopsInterfacePass()";
}

def TileAndDistributeToWorkgroups :
    Pass<"iree-codegen-tile-and-distribute-to-workgroups", "IREE::HAL::ExecutableVariantOp"> {
  let summary = "Tile and distribute operations to workgroups";
  let constructor = "mlir::iree_compiler::createTileAndDistributeToWorkgroupsPass()";
  let options = [
    Option<"maxWorkgroupParallelDims", "max-workgroup-parallel-dims", "int32_t",
      /*default=*/ "",
      "Maximum number of dims to distribute workgroups across.">,
    Option<"distributionMethod", "distribution-method", "int32_t",
      /*default=*/ "0",
      "Pick the distribution method">      
  ];
}

def WorkgroupSpecialization :
    Pass<"iree-codegen-workgroup-specialization", "func::FuncOp"> {
  let summary = "Specialize workgroup distribution loops";
  let constructor = "mlir::iree_compiler::createWorkgroupSpecializationPass()";
}

def TypePropagation :
    Pass<"iree-codegen-type-propagation", "func::FuncOp"> {
  let summary = "Propogate the type of tensor to avoid load/stores of illegal bit widths";
  let constructor = "mlir::iree_compiler::createTypePropagationPass()";
}

def RemoveSingleIterationLoop :
    Pass<"iree-codegen-remove-single-iteration-loop", "func::FuncOp"> {
  let summary = "Remove distributed loop with single iteration.";
  let constructor = "mlir::iree_compiler::createRemoveSingleIterationLoopPass()";
}

def TensorToVectorVectorizePad :
    Pass<"iree-codegen-vectorize-tensor-pad", "func::FuncOp"> {
  let summary = "Vectorize a very specific form of tensor.pad with "
                "control flows";
  let constructor =
      "mlir::iree_compiler::createVectorizePadPass()";
}

def VectorizePackUnPackOps :
    Pass<"iree-codegen-vectorize-pack-unpack-ops", "func::FuncOp"> {
  let summary = "Vectorize a very specific form of tensor.pad with "
                "control flows";
  let constructor =
      "mlir::iree_compiler::createVectorizePackUnPackOpsPass()";
}

def PolynomialApproximationPass :
    Pass<"iree-codegen-polynomial-approximation", ""> {
  let summary = "Convert math operations to their polynomial approximation";
  let constructor =
      "mlir::iree_compiler::createPolynomialApproximationPass()";
}

def MemrefCopyToLinalgPass :
    Pass<"iree-codegen-memrefcopy-to-linalg", "func::FuncOp"> {
  let summary = "Convert memref.copy to linalg op";
  let constructor =
      "mlir::iree_compiler::createMemrefCopyToLinalgPass()";
}

def GPUDistributeSharedMemoryCopy :
    Pass<"iree-gpu-distribute-shared-memory-copy", "func::FuncOp"> {
  let summary = "Pass to distribute shared memory copies to threads.";
  let constructor = "mlir::iree_compiler::createGPUDistributeSharedMemoryCopy()";
}

def GPUReduceBankConflicts :
    Pass<"iree-gpu-reduce-bank-conflicts", "func::FuncOp"> {
  let summary = "Pass to try to reduce the number of bank conflicts.";
  let constructor = "mlir::iree_compiler::createGPUReduceSharedMemoryBankConflicts()";
}

def GPUMultiBuffering :
    Pass<"iree-gpu-multi-buffering", "func::FuncOp"> {
  let summary = "Pass to do multi buffering.";
  let constructor = "mlir::iree_compiler::createGPUMultiBuffering()";
}

def GPUPipelining : Pass<"iree-gpu-pipelining", "func::FuncOp"> {
  let summary = "Pass to do software pipelining.";
  let constructor = "mlir::iree_compiler::createGPUPipeliningPass()";
  let options = [
    Option<"epiloguePeeling", "epilogue-peeling", "bool",
            /*default=*/"true",
           "Try to use un-peeling epilogue when false, peeled epilouge o.w.">,
    Option<"depth", "pipeline-depth", "int64_t",
            /*default=*/"2",
           "Number of stages ">,
    Option<"scheduleIndex", "schedule-index", "int64_t",
            /*default=*/"0",
           "Allows picking different schedule for the pipelining transformation.">,
    Option<"transformFileName", "transform-file-name", "std::string",
            /*default=*/"\"\"",
            "Optional filename containing a transform dialect specification to "
            "apply. If left empty, the IR is assumed to contain one top-level "
            "transform dialect operation somewhere in the module.">,
  ];
}

def WorkGroupSwizzle :
    Pass<"iree-workgroup-swizzle", "func::FuncOp"> {
  let summary = "swizzle the workgroup ids for better cache reuse";
  let constructor = "mlir::iree_compiler::createWorkGroupSwizzle()";
  let options = [
    Option<"logTile", "logTile", "unsigned",
            /*default=*/"0",
           "pass the tile value for unit testing">,
  ];
}

def PadDynamicAlloc :
    Pass<"iree-codegen-pad-dynamic-alloc", "func::FuncOp"> {
  let summary = "Pass to pad dynamic alloc into static one.";
  let constructor = "mlir::iree_compiler::createPadDynamicAlloc()";
}

def TransformDialectInterpreter :
    Pass<"iree-transform-dialect-interpreter"> {
  let summary = "Pass to apply transform dialect operations.";
  let constructor =
    "mlir::iree_compiler::createTransformDialectInterpreterPass()";
  let options = [
    Option<"transformFileName", "transform-file-name", "std::string",
            /*default=*/"\"\"",
            "Optional filename containing a transform dialect specification to "
            "apply. If left empty, the IR is assumed to contain one top-level "
            "transform dialect operation somewhere in the module.">,
    Option<"debugPayloadRootTag", "debug-payload-root-tag", "std::string",
            /*default=*/"\"\"",
            "Select the operation with 'transform.iree_tag' attribute having "
            "the given value as payload IR root. This allows user control on "
            "what operation to transform in debug mode, without requiring "
            "intimate knowledge of the IREE nested pass pipeline.\\n"
            "If empty (normal operation mode), select the pass anchor "
            "operation in the IREE pipeline, as the payload IR root.">,
    Option<"debugTransformRootTag", "debug-transform-root-tag", "std::string",
            /*default=*/"\"\"",
            "Select the operation with 'transform.iree_tag' attribute having "
            "the given value as container IR for top-level transform ops. This "
            "allows user control on what transformation to apply in debug "
            "mode, without requiring intimate knowledge of the IREE nested "
            "pass pipeline.\\n"
            "If empty (normal operation mode), select the container of the "
            "top-level transform op.">
  ];
}

def GPUVectorization :
    Pass<"iree-codegen-gpu-vectorization", "func::FuncOp"> {
  let summary = "Pass to convert linalg into Vector.";
  let constructor = "mlir::iree_compiler::createGPUVectorizationPass()";
  let options = [
    Option<"generateContract", "generate-contract", "bool",
            /*default=*/"true",
           "Try to convert reduction to vector.contract.">,
    Option<"maxVectorSize", "max-vector-size", "int64_t",
            /*default=*/"4096",
           "Max vector size allowed to avoid creating large vectors.">
  ];
}

def GPUTileReduction :
    Pass<"iree-codegen-gpu-tile-reduction", "func::FuncOp"> {
  let summary = "Pass to tile linalg reduction dimensions.";
  let constructor = "mlir::iree_compiler::createGPUTileReductionPass()";
}

def VectorReduceToGPU :
    Pass<"iree-codegen-reduction-to-gpu", "func::FuncOp"> {
  let summary = "Convert vector reduction to gpu ops.";
  let constructor = "mlir::iree_compiler::createConvertVectorReductionToGPUPass()";
}

def FuseTensorPadWithConsumer :
    Pass<"iree-codegen-fuse-tensor-pad-with-consumer", "func::FuncOp"> {
  let summary = "Fuse tensor.pad op into its consumer op's tiled loop nest";
  let constructor = "mlir::iree_compiler::createFuseTensorPadWithConsumerPass()";
}

def ConcretizePadResultShape :
    Pass<"iree-codegen-concretize-pad-result-shape", "func::FuncOp"> {
  let summary =
      "Concretizes tensor.pad op's result shape if its source op"
      "implements OffsetSizeAndStrideOpInterface.";
  let constructor = "mlir::iree_compiler::createConcretizePadResultShapePass()";
}

def EraseHALDescriptorTypeFromMemRef :
    Pass<"iree-codegen-erase-hal-descriptor-type-from-memref", "func::FuncOp"> {
  let summary = "Erase #hal.descriptor_type from MemRef memory space";
  let constructor =
      "mlir::iree_compiler::createEraseHALDescriptorTypeFromMemRefPass()";
}

def RematerializeParallelOps :
    Pass<"iree-codegen-rematerialize-parallel-ops", "func::FuncOp"> {
  let summary = "Pass to rematerialize and merge parallel ops to avoid creating temporary allocs.";
  let constructor = "mlir::iree_compiler::createRematerializeParallelOpsPass()";
}

def InstrumentMemoryAccesses :
    Pass<"iree-codegen-instrument-memory-accesses", "func::FuncOp"> {
  let summary = "Instruments memory reads and writes for address tracking when dispatch instrumentation is enabled.";
  let constructor = "mlir::iree_compiler::createInstrumentMemoryAccessesPass()";
}

//------------------------------------------------------------------------------
// LLVMCPU
//------------------------------------------------------------------------------

def VerifyLinalgTransformLegality :
    Pass<"iree-llvmcpu-verify-linalg-transform-legality", "ModuleOp"> {
  let summary = "Verify that only supported IR constructs are passed to the compiler.";
  let constructor = "mlir::iree_compiler::createVerifyLinalgTransformLegalityPass()";
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

// TODO: Bring the argument in line with the names used elsewhere.
def LLVMGPULowerExecutableTarget :
    Pass<"iree-llvmgpu-lower-executable-target", "mlir::iree_compiler::IREE::HAL::ExecutableVariantOp"> {
  let summary = "Perform lowering of executable target using one of the IREE::HAL::DispatchLoweringPassPipeline";
  let constructor = "mlir::iree_compiler::createLLVMGPULowerExecutableTargetPass()";
}

def LLVMGPUTileAndDistribute :
    Pass<"iree-llvmgpu-tile-and-distribute", "func::FuncOp"> {
  let summary = "Pass to tile and distribute linalg ops within a workgroup.";
  let constructor = "mlir::iree_compiler::createLLVMGPUTileAndDistribute()";
}

def LLVMGPUTileTensor :
    Pass<"iree-llvmgpu-tile-tensor", "func::FuncOp"> {
  let summary = "Pass to tile linalg on tensor ops within a workgroup.";
  let constructor = "mlir::iree_compiler::createLLVMGPUTileTensor()";
}

def LLVMGPUDistribute :
    Pass<"iree-llvmgpu-distribute", "func::FuncOp"> {
  let summary = "Pass to distribute scf.forall ops.";
  let constructor = "mlir::iree_compiler::createLLVMGPUDistribute()";
}

def LLVMGPUTensorAlloc :
    Pass<"iree-llvmgpu-alloc", "func::FuncOp"> {
  let summary = "Pass to create allocation for some values.";
  let constructor = "mlir::iree_compiler::createLLVMGPUTensorAlloc()";
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
  let summary = "Emulate 64-bit integer opts with 32-bit integer ops";
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

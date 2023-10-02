// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_COMMON_PASSES
#define IREE_CODEGEN_COMMON_PASSES

include "mlir/Pass/PassBase.td"

//===---------------------------------------------------------------------===//
// Common passes for all backends (keep alphabetical)
//===---------------------------------------------------------------------===//

def AddFastMathFlags
    : Pass<"iree-codegen-add-fast-math-flags", "LLVM::LLVMFuncOp"> {
  let summary = "Add fast math flags to all the operations supporting them, "
                "given a floating-point mode.";
  let constructor = "mlir::iree_compiler::createAddFastMathFlagsPass()";
}

def BubbleUpOrdinalOps : Pass<"iree-codegen-bubble-up-ordinal-ops", ""> {
  let summary = "Bubbles op ordinal ops to allow for workgroup count computation";
  let constructor = "mlir::iree_compiler::createBubbleUpOrdinalOpsPass()";
}

def BufferizeCopyOnlyDispatches :
  Pass<"iree-codegen-bufferize-copy-only-dispatches", "ModuleOp"> {
  let summary =
      "Bufferize dispatches that copy to/from interfaces to convert to a linalg.copy op";
  let constructor = "mlir::iree_compiler::createBufferizeCopyOnlyDispatchesPass()";
}

def CleanupBufferAllocView :
    Pass<"iree-codegen-cleanup-buffer-alloc-view", "func::FuncOp"> {
  let summary =
      "Performs cleanups over HAL interface/buffer allocation/view operations";
  let constructor = "mlir::iree_compiler::createCleanupBufferAllocViewPass()";
}

def ConcretizePadResultShape :
    Pass<"iree-codegen-concretize-pad-result-shape", "func::FuncOp"> {
  let summary =
      "Concretizes tensor.pad op's result shape if its source op"
      "implements OffsetSizeAndStrideOpInterface.";
  let constructor = "mlir::iree_compiler::createConcretizePadResultShapePass()";
}

def ConvertBf16ArithToF32 : Pass<"iree-convert-bf16-arith-to-f32", "ModuleOp"> {
  let summary = "Convert bf16 arithmetic operations to f32";
  let constructor = "mlir::iree_compiler::createConvertBf16ArithToF32Pass()";
}

def ConvertBf16ToUInt16Buffers :
    Pass<"iree-convert-bf16-to-uint16-buffers", "ModuleOp"> {
  let summary = "Convert bf16 buffers to uint16 equivalents";
  let constructor = "mlir::iree_compiler::createConvertBf16ToUInt16BuffersPass()";
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
      "affine::AffineDialect"
  ];
}

def DecomposeBatchMmt4DOps
    : Pass<"iree-codegen-decompose-batch-mmt4d-ops", "func::FuncOp"> {
  let summary = "Decompose batch_mmt4d ops into mmt4d ops";
  let constructor =
      "mlir::iree_compiler::createDecomposeBatchMmt4DOpsPass()";
}

def DecomposeConvolutionToLowerDimOps :
    Pass<"iree-codegen-decompose-convolution-to-lower-dim-ops", ""> {
  let summary = "Decomposes linalg convolution ops to lower dim ops";
  let constructor =
  "mlir::iree_compiler::createDecomposeConvolutionToLowerDimOpsPass()";
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

def DecomposePackUnPackOps :
    Pass<"iree-codegen-decompose-pack-unpack-ops", "func::FuncOp"> {
  let summary = "Decompose pack/unpack ops into vectorizable ops";
  let constructor = "mlir::iree_compiler::createDecomposePackUnPackOpsPass()";
  let options = [
    Option<"tileOuterToOne", "tile-outer-to-one", "bool", "false",
           "Always apply tiling to make outer dimension be ones">
  ];
}

def ReplaceSlowMinMaxOps
    : Pass<"iree-codegen-replace-slow-min-max-ops", "func::FuncOp"> {
  let summary =
      "Replace slow min/max operations that propagate NaNs and distinguish "
      "between +/-0.0 with faster min/max operations that ignore them.";
  let constructor = "mlir::iree_compiler::createReplaceSlowMinMaxOpsPass()";
}

def EliminateEmptyTensors :
    Pass<"iree-eliminate-empty-tensors", "ModuleOp"> {
  let summary = "Eliminate tensor.empty ops to avoid buffer allocations";
  let constructor = "mlir::iree_compiler::createEliminateEmptyTensorsPass()";
}

def EmulateNarrowType :
    Pass<"iree-codegen-emulate-narrow-type", "ModuleOp"> {
  let summary = "Emulate narrow integer operations using wide integer operations";
  let constructor = "mlir::iree_compiler::createEmulateNarrowTypePass()";
}

def EraseDeadAllocAndStores :
    Pass<"iree-codegen-erase-dead-alloc-and-stores", "func::FuncOp"> {
  let summary = "Erase alloc ops if all the uses are just stores";
  let constructor =
      "mlir::iree_compiler::createEraseDeadAllocAndStoresPass()";
}

def EraseHALDescriptorTypeFromMemRef
    : Pass<"iree-codegen-erase-hal-descriptor-type-from-memref"> {
  let summary = "Erase #hal.descriptor_type from MemRef memory space";
  let constructor =
      "mlir::iree_compiler::createEraseHALDescriptorTypeFromMemRefPass()";
}

def ConvertHALDescriptorTypeToGPUAddressSpace
    : Pass<"iree-codegen-convert-hal-descriptor-type-to-gpu-address-space"> {
  let summary = "Convert #hal.descriptor_type to #gpu.address_space<global>";
  let constructor =
      "mlir::iree_compiler::createConvertHALDescriptorTypeToGPUAddressSpacePass()";
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

def FuseTensorPadWithConsumer :
    Pass<"iree-codegen-fuse-tensor-pad-with-consumer", "func::FuncOp"> {
  let summary = "Fuse tensor.pad op into its consumer op's tiled loop nest";
  let constructor = "mlir::iree_compiler::createFuseTensorPadWithConsumerPass()";
}

def GenericVectorization :
    Pass<"iree-codegen-generic-vectorization", "func::FuncOp"> {
  let summary = "Pass to perform vectorization on tensor/linalg ops.";
  let options = [
    Option<"enableVectorMasking", "enable-vector-masking", "bool",/*default=*/"false",
      "Enable vector masking during vectorization.">,
    Option<"vectorizePadding", "vectorize-padding", "bool", /*default=*/"false",
      "Rewrite all tensor.pad ops in the function to vector form.">,
    Option<"vectorizeGatherAccesses", "vectorize-gather-accesses", "bool", /*default=*/"false",
      "Enable vectorizaiton of operations that may generate vector.gather operations.">,
    Option<"enableCleanup", "enable-cleanup", "bool",/*default=*/"true",
      "Enable cleanups after vectorization. The patterns touch the structure"
      "generated from tiling so it affects later steps like bufferization and vector hoisting.">,
    Option<"generateContract", "generate-contract", "bool",/*default=*/"true",
      "Enable conversion for reduction ops to contraction ops.">,
    Option<"maxVectorSize", "max-vector-size", "int64_t",
            /*default=*/"2147483647",
           "Max vector size allowed to avoid creating large vectors.">
  ];
  let constructor =
      "mlir::iree_compiler::createGenericVectorizationPass()";
}

def HoistRedundantVectorTransfers :
    Pass<"iree-codegen-hoist-redundant-vector-transfers", "func::FuncOp"> {
  let summary = "Hoist redundant vector transfers";
  let constructor = "mlir::iree_compiler::createHoistRedundantVectorTransfersPass()";
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

def IREEExpandStridedMetadata :
    Pass<"iree-codegen-expand-strided-metadata", ""> {
  let summary = "Resolve memref.extract_strided_metadata operations";
  let constructor = "mlir::iree_compiler::createIREEExpandStridedMetadataPass()";
  let options = [
    Option<"allowUnresolved", "allow-unresolved", "bool", /*default=*/"false",
           "Allow unresolved strided metadata op (for testing)">,
  ];
}

def InstrumentMemoryAccesses :
    Pass<"iree-codegen-instrument-memory-accesses", "func::FuncOp"> {
  let summary = "Instruments memory reads and writes for address tracking when dispatch instrumentation is enabled.";
  let constructor = "mlir::iree_compiler::createInstrumentMemoryAccessesPass()";
}

def LowerUKernelOpsToCalls :
    Pass<"iree-codegen-lower-ukernel-ops-to-calls", "ModuleOp"> {
  let summary = "Lower micro-kernel wrapper ops into function calls";
  let constructor = "mlir::iree_compiler::createLowerUKernelOpsToCallsPass()";
}

def MemrefCopyToLinalgPass :
    Pass<"iree-codegen-memrefcopy-to-linalg", "func::FuncOp"> {
  let summary = "Convert memref.copy to linalg op";
  let constructor =
      "mlir::iree_compiler::createMemrefCopyToLinalgPass()";
}

def OptimizeVectorTransfer :
    Pass<"iree-codegen-optimize-vector-transfer", "func::FuncOp"> {
  let summary =
      "Run optimization transformations on vector transfer operations";
  let constructor = "mlir::iree_compiler::createOptimizeVectorTransferPass()";
  let options = [
    Option<"optionFlatten", "flatten", "bool", "false",
           "Flatten the vector type of vector transfers where possible (contiguous row-major data).">,
    Option<"optionDropUnitDims", "drop-unit-dims", "bool", /*default=*/"true",
           "Drop unit dims in vector transfers where possible (might generate vector.shape_cast).">,
  ];
  let dependentDialects = [
      "memref::MemRefDialect"
  ];
}

def PadDynamicAlloc :
    Pass<"iree-codegen-pad-dynamic-alloc", "func::FuncOp"> {
  let summary = "Pass to pad dynamic alloc into static one.";
  let constructor = "mlir::iree_compiler::createPadDynamicAlloc()";
}

def PolynomialApproximationPass :
    Pass<"iree-codegen-polynomial-approximation", ""> {
  let summary = "Convert math operations to their polynomial approximation";
  let constructor =
      "mlir::iree_compiler::createPolynomialApproximationPass()";
}

def RematerializeParallelOps :
    Pass<"iree-codegen-rematerialize-parallel-ops", "func::FuncOp"> {
  let summary = "Pass to rematerialize and merge parallel ops into consumers.";
  let constructor = "mlir::iree_compiler::createRematerializeParallelOpsPass()";
}

def RemoveSingleIterationLoop :
    Pass<"iree-codegen-remove-single-iteration-loop", "func::FuncOp"> {
  let summary = "Remove distributed loop with single iteration.";
  let constructor = "mlir::iree_compiler::createRemoveSingleIterationLoopPass()";
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

def TensorToVectorVectorizePad :
    Pass<"iree-codegen-vectorize-tensor-pad", "func::FuncOp"> {
  let summary = "Vectorize a very specific form of tensor.pad with "
                "control flows";
  let constructor =
      "mlir::iree_compiler::createVectorizePadPass()";
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

def TransformDialectInterpreter :
    Pass<"iree-transform-dialect-interpreter"> {
  let summary = "Pass to apply transform dialect operations.";
  let constructor =
    "mlir::iree_compiler::createTransformDialectInterpreterPass()";
  let description = [{
    This pass runs the transform dialect interpreter and applies the named
    sequence transformation specified by the provided name (defaults to
    `__transform_main`).
  }];
  let options = [
    Option<"entryPoint", "entry-point", "std::string",
           /*default=*/[{"__transform_main"}],
           "Entry point of the pass pipeline.">,
  ];
}

def TypePropagation :
    Pass<"iree-codegen-type-propagation", "func::FuncOp"> {
  let summary = "Propogate the type of tensor to avoid load/stores of illegal bit widths";
  let constructor = "mlir::iree_compiler::createTypePropagationPass()";
}

#endif // IREE_CODEGEN_COMMON_PASSES

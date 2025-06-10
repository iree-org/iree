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

def AddFastMathFlagsPass
    : Pass<"iree-codegen-add-fast-math-flags", "LLVM::LLVMFuncOp"> {
  let summary = "Add fast math flags to all the operations supporting them, "
                "given a floating-point mode.";
}

def BlockDynamicDimensionsPass
    : Pass<"iree-codegen-block-dynamic-dimensions"> {
  let summary = "Expand dynamic dimensions that are known to be multiples of "
                "statically known values.";
  let dependentDialects = ["iree_compiler::IREE::Util::UtilDialect"];
}

def BubbleUpOrdinalOpsPass : Pass<"iree-codegen-bubble-up-ordinal-ops", ""> {
  let summary = "Bubbles op ordinal ops to allow for workgroup count computation";
  let description = [{
    Pass to bubble up ordinal operations to allow workgroup count computation
    based on slices to correlate back to workload computation.
  }];
}

def BufferizeCopyOnlyDispatchesPass :
  InterfacePass<"iree-codegen-bufferize-copy-only-dispatches", "mlir::FunctionOpInterface"> {
  let summary =
      "Bufferize dispatches that copy to/from interfaces to convert to a linalg.copy op";
  let description = [{
    Pass to bufferize dispatches that are copying from one interface to
    another. This will create a `linalg.generic` op which is a copy that can
    then be used by backends to handle appropriately.
  }];
}

def BufferizeDispatchTensorLoadStorePass :
  InterfacePass<"iree-codegen-bufferize-dispatch-tensor-load-store", "mlir::FunctionOpInterface"> {
  let summary =
      "Bufferize the iree_tensor_ext.dispatch.tensor.load/store ops at dispatch boundaries";
  let description = [{
    Pass to bufferize the edges of dispatch regions, converting
    iree_tensor_ext.dispatch.tensor.load ops to iree_codegen.load_from_buffer, and
    iree_tensor_ext.dispatch.tensor.store ops to iree_codegen.store_to_buffer.
  }];
  let dependentDialects = [
    "IREE::Codegen::IREECodegenDialect",
    "memref::MemRefDialect"
  ];
}

def ConfigTrackingCanonicalizerPass :
    Pass<"iree-codegen-config-tracking-canonicalize", ""> {
  let summary = "Codegen specific canonicalization pass that tracks lowering configs";
  let options = [
    Option<"testConvergence", "test-convergence", "bool",
           /*default=*/"false", "Fails if the patterns fail to converge">
  ];
}

def CleanupBufferAllocViewPass :
    InterfacePass<"iree-codegen-cleanup-buffer-alloc-view", "mlir::FunctionOpInterface"> {
  let summary =
      "Performs cleanups over HAL interface/buffer allocation/view operations";
}

def ConcretizePadResultShapePass :
    InterfacePass<"iree-codegen-concretize-pad-result-shape", "mlir::FunctionOpInterface"> {
  let summary =
      "Concretizes tensor.pad op's result shape if its source op"
      "implements OffsetSizeAndStrideOpInterface.";
}

def ConvertAccGEMMToGEMMPass :
    Pass<"iree-convert-accgemm-to-gemm", ""> {
  let summary = "Convert accumulating GEMMs to GEMMs post dispatch creation.";
}

def ConvertBf16ArithToF32Pass : Pass<"iree-convert-bf16-arith-to-f32", ""> {
  let summary = "Convert bf16 arithmetic operations to f32";
}

def ConvertBf16ToUInt16BuffersPass :
    Pass<"iree-codegen-convert-bf16-to-uint16-buffers", ""> {
  let summary = "Convert BF16 buffer ops and conversions to simulated behavior with uint16.";
}

def ConvertUnsupportedFloatArithPass
    : InterfacePass<"iree-convert-unsupported-float-arith",
                    "mlir::FunctionOpInterface"> {
  let summary = "Convert arith operations on unsupported(source types) float "
                "types to the target type. Populates the source and target "
                "based on the target architecture.";
}

def ConvertToDestinationPassingStylePass :
    InterfacePass<"iree-codegen-convert-to-destination-passing-style", "mlir::FunctionOpInterface"> {
  let summary =
      "Transforms the code to make the dispatch use destination-passing style";
  let description = [{
    Converts entry point function within dispatch regions to use
    destination-passing style, which is better suited for the upstream
    comprehensive bufferization pass.
  }];
  let options = [
    Option<"convertInputsToDestinations", "convert-inputs-to-destinations",
           "bool", /*default=*/"true",
           "Controls whether to adjust consumers to convert one of its inputs to a destination">,
    Option<"useWARForCooperativeMatrixCodegen", "use-war-for-cooperative-matrix-codegen",
           "bool", /*default=*/"false",
           "WAR for failure in Cooperative matrix codegen pipelines. See #10648.">
  ];
}

def CombineLayoutTransformationPass :
    InterfacePass<"iree-codegen-combine-layout-transformation", "mlir::FunctionOpInterface"> {
  let summary =
    "Combines layout transformation operations into a single map_scatter operation.";
  let description = [{
    Starting from iree_codegen.store_to_buffer ops, iteratively combine producer
    layout/indexing transformation ops (linalg.transpose, tensor.collapse_shape,
    etc.) into a single iree_linalg_ext.map_scatter operation. For tensor.pad
    ops, the writing of pad values is distributed to workgroups, and then the
    padding values are written directly to the output buffer of the
    store_to_buffer op.
  }];
  let dependentDialects = [
    "iree_compiler::IREE::LinalgExt::IREELinalgExtDialect",
    "scf::SCFDialect",
    "tensor::TensorDialect"
  ];
}

def ConvolutionToIGEMMPass :
    InterfacePass<"iree-codegen-convolution-to-igemm", "mlir::FunctionOpInterface"> {
  let summary =
      "Transforms convolution operations into an implicit GEMM format.";
  let dependentDialects = [
    "tensor::TensorDialect",
    "iree_compiler::IREE::LinalgExt::IREELinalgExtDialect"
  ];
}

def DecomposeAffineOpsPass: Pass<"iree-codegen-decompose-affine-ops"> {
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
  let dependentDialects = [
      "mlir::affine::AffineDialect"
  ];
}

def DecomposeConvolutionToLowerDimOpsPass :
    Pass<"iree-codegen-decompose-convolution-to-lower-dim-ops", ""> {
  let summary = "Decomposes linalg convolution ops to lower dim ops";
}

def DecomposeMemrefsPass :
    Pass<"iree-codegen-decompose-memrefs", ""> {
  let summary = "Decomposes memrefs";
}

def DecomposeLinalgGenericPass :
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
}

def DecomposePackUnPackOpsPass :
    InterfacePass<"iree-codegen-decompose-pack-unpack-ops", "mlir::FunctionOpInterface"> {
  let summary = "Decompose pack/unpack ops into vectorizable ops";
  let options = [
    Option<"tileOuterToOne", "tile-outer-to-one", "bool", "false",
           "Always apply tiling to make outer dimension be ones">,
    Option<"useOnlyReshapes", "use-only-reshapes", "bool", "false",
           "Use decomposition into reshape ops, even when packing unit dimensions.">
  ];
  let dependentDialects = [
    "arith::ArithDialect",
    "linalg::LinalgDialect",
    "scf::SCFDialect",
    "tensor::TensorDialect"
  ];
}

def DecomposeBoundaryPackUnPackOpsPass :
    InterfacePass<"iree-codegen-decompose-boundary-pack-unpack-ops", "mlir::FunctionOpInterface"> {
  let summary = "Wrapper for DecomposePackUnPackOpsPass to decompose ops at function boundaries";
  let options = [
    Option<"tileOuterToOne", "tile-outer-to-one", "bool", "false",
           "Always apply tiling to make outer dimension be ones">
  ];
  let dependentDialects = [
    "arith::ArithDialect",
    "linalg::LinalgDialect",
    "scf::SCFDialect",
    "tensor::TensorDialect"
  ];
}

def DecomposeSoftmaxPass :
    InterfacePass<"iree-codegen-decompose-softmax", "mlir::FunctionOpInterface"> {
  let summary =
      "Decomposes softmax op into a sequence of linalg ops";
  let options = [
    Option<"useFusion", "use-fusion",
           "bool", /*default=*/"true",
           "Whether to use the internal pass fusion logic for the exp function. See #15862.">
  ];
}

def DropVectorUnitDimsPass :
    InterfacePass<"iree-codegen-drop-vector-unit-dims", "mlir::FunctionOpInterface"> {
  let summary = "Pass to drop vector unit dims.";
}

def ReconcileTranslationInfoPass
    : Pass<"iree-codegen-reconcile-translation-info", "IREE::HAL::ExecutableVariantOp"> {
  let summary =
      "Reconcile information (like workgroup_size, subgroup_size) across "
      "`TranslationInfo` set on each function in the dispatch and merge them"
      "and set them at the appropriate places in the surrounding HAL ops";
  let options = [
    Option<"distributeAlong", "distribute-along",
           "::mlir::iree_compiler::IREE::Codegen::WorkgroupId",
           /*default=*/"IREE::Codegen::WorkgroupId::IdZ",
           "Constrain the workgroup distribution along grid dimensions.",
           [{
           ::llvm::cl::values(
            clEnumValN(IREE::Codegen::WorkgroupId::IdX, "x",
              "Constrain the workgroup distribution to use only workgroups along x."),
            clEnumValN(IREE::Codegen::WorkgroupId::IdY, "y",
              "Constrain the workgroup distribution to use only workgroups along x and y."),
            clEnumValN(IREE::Codegen::WorkgroupId::IdZ, "z",
              "Constrain the workgroup distribution to use only workgroups along x, y and z."))}]>
  ];
}

def ReplaceSlowMinMaxOpsPass
    : InterfacePass<"iree-codegen-replace-slow-min-max-ops", "mlir::FunctionOpInterface"> {
  let summary =
      "Replace slow min/max operations that propagate NaNs and distinguish "
      "between +/-0.0 with faster min/max operations that ignore them.";
}

def EliminateEmptyTensorsPass :
    InterfacePass<"iree-eliminate-empty-tensors", "mlir::FunctionOpInterface"> {
  let summary = "Eliminate tensor.empty ops to avoid buffer allocations";
}

def EmulateNarrowTypePass :
    Pass<"iree-codegen-emulate-narrow-type", ""> {
  let summary = "Emulate narrow integer operations using wide integer operations";
  let description = [{
    A pass to emulate memref load operations that use narrow integer types
    with equivalent operations on supported wide integer types.
  }];
}

def EraseDeadAllocAndStoresPass :
    InterfacePass<"iree-codegen-erase-dead-alloc-and-stores", "mlir::FunctionOpInterface"> {
  let summary = "Erase alloc ops if all the uses are just stores";
}

def EraseHALDescriptorTypeFromMemRefPass
    : Pass<"iree-codegen-erase-hal-descriptor-type-from-memref"> {
  let summary = "Erase #hal.descriptor_type from MemRef memory space";
}

def ConvertHALDescriptorTypeToGPUAddressSpacePass
    : Pass<"iree-codegen-convert-hal-descriptor-type-to-gpu-address-space"> {
  let summary = "Convert #hal.descriptor_type to #gpu.address_space<global>";
  let dependentDialects = ["::mlir::gpu::GPUDialect"];
}

def ExtractAddressComputationPass : Pass<"iree-codegen-extract-address-computation"> {
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
  let dependentDialects = [
      "memref::MemRefDialect"
  ];
}

def FissionTransferOpsInControlFlowPass : InterfacePass<"iree-codegen-fission-transfer-ops-in-control-flow", "mlir::FunctionOpInterface"> {
  let summary =
      "Fission transfer read and write ops in control flow to allow prefetching.";
  let dependentDialects = [
      "memref::MemRefDialect"
  ];
  let options = [
    Option<"FissionMultiTrip", "fission-multi-trip",
           "bool", /*default=*/"false",
           "Allow fission in presence of loops with greater than one trip count.">
  ];
}

def FlattenMemRefSubspanPass : Pass<"iree-codegen-flatten-memref-subspan", "ModuleOp"> {
  let summary =
      "Flatten n-D MemRef subspan ops to 1-D ones and fold byte offsets";
  let description = [{
    Flattens n-D MemRef subspan ops to 1-D MemRef and folds the byte offsets
    on subspan ops to the consumer load/store ops, in preparation for lowering
    to backends that require linearized access.
  }];
}

def FoldAffineMinInDistributedLoopsPass :
  InterfacePass<"iree-codegen-fold-affinemin-in-distributed-loops", "mlir::FunctionOpInterface"> {
  let summary = "Fold `affine.min` ops in distributed loops";
}

def FoldTensorExtractOpPass :
  Pass<"iree-codegen-fold-tensor-extract-op", ""> {
  let summary = "Fold `tensor.extract` operations prior to lowering to LLVM";
  let description = [{
    After running the upstream TensorConstantBufferize pass, remove
    tensor_loads introduced for use only in tensor_extract. These can be
    folded to use a load of the created memref object that holds the constant
    values.
  }];
}

def ForallToForPass :
  InterfacePass<"iree-codegen-forall-to-for", "mlir::FunctionOpInterface"> {
  let summary = "Convert scf.forall to nested scf.for loops";
  let description = [{
    Converts scf.forall operations (without workgroup mapping) to nested
    scf.for loops.
  }];
}

def ForOpCanonicalizationPass :
  InterfacePass<"iree-codegen-canonicalize-scf-for", "mlir::FunctionOpInterface"> {
  let summary =
      "Adhoc canonicalization of selected loop-carried values/dependencies for scf.for ops";
}

def FuseTensorPadWithConsumerPass :
    InterfacePass<"iree-codegen-fuse-tensor-pad-with-consumer", "mlir::FunctionOpInterface"> {
  let summary = "Fuse tensor.pad op into its consumer op's tiled loop nest";
}

def GenericVectorizationPass :
    InterfacePass<"iree-codegen-generic-vectorization", "mlir::FunctionOpInterface"> {
  let summary = "Pass to perform vectorization on tensor/linalg ops.";
  let options = [
    Option<"enableVectorMasking", "enable-vector-masking", "bool",/*default=*/"false",
      "Enable vector masking during vectorization.">,
    Option<"useConfiguredVectorSizes", "use-configured-vector-sizes", "bool",/*default=*/"true",
      "Control whether the op lowering config represents a set of masked vector sizes">,
    Option<"vectorizeCopies", "vectorize-copies", "bool", /*default=*/"true",
      "Enable vectorization of linalg.copy operations.">,
    Option<"vectorizePadding", "vectorize-padding", "bool", /*default=*/"false",
      "Rewrite all tensor.pad ops in the function to vector form.">,
    Option<"vectorizeGatherAccesses", "vectorize-gather-accesses", "bool", /*default=*/"false",
      "Enable vectorizaiton of operations that may generate vector.gather operations.">,
    Option<"vectorizeToTransferGather", "vectorize-to-transfer-gather", "bool", /*default=*/"false",
      "Enables vectorization of gather-like operations that may generate iree_vector_ext.transfer_gather">,
    Option<"enableCleanup", "enable-cleanup", "bool",/*default=*/"true",
      "Enable cleanups after vectorization. The patterns touch the structure"
      "generated from tiling so it affects later steps like bufferization and vector hoisting.">,
    Option<"generateContract", "generate-contract", "bool",/*default=*/"true",
      "Enable conversion for reduction ops to contraction ops.">,
    Option<"foldCastIntoContract", "fold-cast-into-contract", "bool",/*default=*/"false",
      "Enable folding casting ops into vector.contract.">,
    Option<"maxVectorSize", "max-vector-size", "int64_t",
            /*default=*/"2147483647",
           "Max vector size allowed to avoid creating large vectors.">
  ];
  let dependentDialects = [
    "::mlir::arith::ArithDialect"
  ];
}

def OptimizeTensorInsertExtractSlicesPass
    : InterfacePass<"iree-codegen-optimize-tensor-insert-extract-slices",
                    "mlir::FunctionOpInterface"> {
  let summary = "Optimize tensor.insert_slice/tensor.extract_slice operations "
                "(e.g. hoist and fold)";
  let options = [
    Option<"foldIdentitySlices", "fold-identity-slices", "bool", "false",
           "Enable folding of identity tensor.*_slice ops.">
  ];
}

def HoistUnrolledVectorExtractInsertSlicePass :
    InterfacePass<"iree-codegen-hoist-vector-extract-insert-slice", "mlir::FunctionOpInterface"> {
  let summary = "Hoist unrolled vector (extract, insert) pairs out of scf.for op";
}

def HoistStaticallyBoundAllocationsPass :
    InterfacePass<"iree-codegen-hoist-statically-bound-allocations", "mlir::FunctionOpInterface"> {
  let summary = "Hoist statically bound alloca ops to the entry block of functions";
  // Note: These options only exist to help with testing, real world uses should look at the target.
  // There also should be no observable change if the input IR is not scalable.
  let options = [
    Option<"vscaleMin", "vscale-min", "unsigned",
            /*default=*/"0",
           "Minimum possible value of vscale.">,
    Option<"vscaleMax", "vscale-max", "unsigned",
            /*default=*/"0",
           "Maximum possible value of vscale (a value of zero means unbounded).">
  ];
  let dependentDialects = [
      "affine::AffineDialect"
  ];
}

def IREEBufferizeConstantsPass :
    Pass<"iree-codegen-iree-bufferize-constants", ""> {
  let summary = "Convert from arith.constant on tensors to buffers";
}

def IREEComprehensiveBufferizePass :
    InterfacePass<"iree-codegen-iree-comprehensive-bufferize", "mlir::FunctionOpInterface"> {
  let summary = "Convert from to Linalg ops on tensors to buffers";
  let options = [
    Option<"testAnalysisOnly", "test-analysis-only", "bool",
            /*default=*/"false",
           "Only runs inplaceability analysis (for testing purposes only)">,
    Option<"printConflicts", "print-conflicts", "bool",
            /*default=*/"false",
           "Annotates IR with RaW conflicts. Requires test-analysis-only.">,
  ];
}

def IREEExpandStridedMetadataPass :
    Pass<"iree-codegen-expand-strided-metadata", ""> {
  let summary = "Resolve memref.extract_strided_metadata operations";
  let options = [
    Option<"allowSubviewExpansion", "allow-subview-expansion", "bool", /*default=*/"false",
           "Enables expansion of memref.subview ops">,
    Option<"allowUnresolved", "allow-unresolved", "bool", /*default=*/"false",
           "Allow unresolved strided metadata op (for testing)">,
  ];
}
def IREEInjectAssumeAlignmentPass :
    InterfacePass<"iree-codegen-inject-assume-alignment", "mlir::FunctionOpInterface"> {
  let summary = "Insert memref.assume_alignment ops right after hal.interface.binding.subspan ops, if alignment is present in bindings.";
  let dependentDialects = [
      "memref::MemRefDialect"
  ];
}

def InstrumentMemoryAccessesPass :
    InterfacePass<"iree-codegen-instrument-memory-accesses", "mlir::FunctionOpInterface"> {
  let summary = "Instruments memory reads and writes for address tracking when dispatch instrumentation is enabled.";
}

def LinkTuningSpecsPass : Pass<"iree-codegen-link-tuning-specs", "ModuleOp"> {
  let summary =
      "Link nested transform dialect tuning specs named sequences into a single entry point";
  let description = [{
    Given a module with multiple nested tuning specs, introduce a new named sequence
    that includes all the other tuning spec entry points. The order of inclusion is the same
    as the order in which these nested tuning specs appear in the IR.

    A tuning spec entry point is a `transform.named_sequence` op annotated with the
    `iree_codegen.tuning_spec` unit attribute. We require it to perform in-place op
    modification and not consume the handle.
  }];
  let dependentDialects = ["transform::TransformDialect"];
}

def LowerExecutableUsingTransformDialectPass :
    Pass<"iree-codegen-lower-executable-using-transform-dialect", "ModuleOp"> {
  let summary = "Lower executables using the transform dialect recipe provided in the module.";
}

def LoweringConfigInterpreterPass :
    Pass<"iree-codegen-lowering-config-interpreter"> {
  let summary = "Pass to apply lowering config annotated strategies.";
  let description = [{
    This pass runs the transform dialect interpreter and applies the named
    sequence transformation specified by lowering configs annotated on
    operations.
  }];
}

def LowerUKernelOpsToCallsPass :
    Pass<"iree-codegen-lower-ukernel-ops-to-calls", "ModuleOp"> {
  let summary = "Lower micro-kernel wrapper ops into function calls";
}

def MaterializeHostEncodingPass :
    Pass<"iree-codegen-materialize-host-encoding", "mlir::ModuleOp"> {
  let summary = "Materialize the encoding for tensor as specified by the backend.";
}

def MaterializeDeviceEncodingPass :
    InterfacePass<"iree-codegen-materialize-device-encoding", "mlir::FunctionOpInterface"> {
  let summary = "Materialize the encoding for tensor as specified by the backend.";
  let options = [
    Option<"testCLGPUTarget", "test-cl-gpu-target", "bool", /*default=*/"false",
           "Flag used for lit-testing GPU target only. Not for general usage">,
  ];
}

// TODO(hanchung): Remove the pass after we deprecate MaterializeHomogeneousEncodingsPass.
def MaterializeEncodingIntoNopPass :
    InterfacePass<"iree-codegen-materialize-encoding-into-nop", "mlir::FunctionOpInterface"> {
  let summary = "Drop the encodings from tensor types with encodings.";
}

def MaterializeEncodingIntoPaddingPass :
    InterfacePass<"iree-codegen-materialize-encoding-into-padding", "mlir::FunctionOpInterface"> {
  let summary = "Materialize `#iree_encoding.pad_encoding_layout` attributes.";
  let description = [{
    Handles padding introduced by `pad_encoding_layout` encoding layouts, which
    requires `iree_tensor_ext.dispatch.tensor.load`/`.store` to be adjusted to account for
    padding regions.
    Materializes any other encoding layouts into nop.
  }];
}

def MaterializeTuningSpecsPass : Pass<"iree-codegen-materialize-tuning-specs", "ModuleOp"> {
  let summary =
      "Load tuning spec transform dialect libraries and encode them in the module";
  let description = [{
    Links all available tuning spec transform dialect modules into a single
    tuning spec. Next, serializes this tuning spec to bytecode and attaches it
    as a module attribute. We do this so that the full tuning spec is always
    encoded in the program IR and can be checked with `--mlir-print-ir-after-all`
    (or equivalent). The alternative would be to add the tuning spec as a
    submodule in the compiled program, but this may result in the tuning spec
    being inadvertently visited by other passes that attempt to `walk` the outer
    module. Serialization makes the tuning specs opaque and prevents it from
    happening.

    This attribute is expected to be short-lived and removed by
    `iree-codegen-materialize-user-configs`.
  }];
}

def MaterializeUserConfigsPass : Pass<"iree-codegen-materialize-user-configs", "ModuleOp"> {
  let summary = "Sets the lowering configs and translation info from user configs";
  let dependentDialects = [
      "transform::TransformDialect"
  ];
}

def MemrefCopyToLinalgPass :
    InterfacePass<"iree-codegen-memrefcopy-to-linalg", "mlir::FunctionOpInterface"> {
  let summary = "Convert memref.copy to linalg op";
}

def NormalizeLoopBoundsPass :
    Pass<"iree-codegen-normalize-loop-bounds", ""> {
  let summary = "Normalize the loop bounds of `scf.for` and `scf.forall`";
  let description = [{
    Normalizes the iteration range of `scf.for` and `scf.forall` loops to
    [0, ub) += 1.
  }];
  let options = [
    Option<"normalizeFor", "normalize-for", "bool", "true",
           "Enable normalization for `scf.for` loops">,
    Option<"normalizeForall", "normalize-forall", "bool", "true",
           "Enable normalization for `scf.forall` loops">,
  ];
  let dependentDialects = [
      "affine::AffineDialect",
      "arith::ArithDialect"
  ];
}

def OptimizeVectorTransferPass :
    InterfacePass<"iree-codegen-optimize-vector-transfer", "mlir::FunctionOpInterface"> {
  let summary =
      "Run optimization transformations on vector transfer operations";
  let options = [
    Option<"flatten", "flatten", "bool", "false",
           "Flatten the vector type of vector transfers where possible (contiguous row-major data).">,
    Option<"redundantHoisting", "redundant-hoisting", "bool", "true",
           "Enables use of redundant vector transfer hoisting.">,
  ];
  let dependentDialects = [
      "memref::MemRefDialect"
  ];
}

def PadDynamicAllocPass :
    InterfacePass<"iree-codegen-pad-dynamic-alloc", "mlir::FunctionOpInterface"> {
  let summary = "Pass to pad dynamic alloc into static one.";
}

def MathTransformPass :
    Pass<"iree-codegen-math-transform", ""> {
  let summary = "Apply math ops transformations: approximations, rewrites to other math ops, operand casts.";
}

def PropagateDispatchSizeBoundsPass :
    InterfacePass<"iree-codegen-propagate-dispatch-size-bounds", "mlir::FunctionOpInterface"> {
  let summary = "Pass to annotate workitem and workgroup IDs with known bounds";
  let dependentDialects = ["::mlir::arith::ArithDialect"];
}

def PropagateReshapesByExpansionPass :
    Pass<"iree-codegen-propagate-reshapes-by-expansion", ""> {
  let summary = "Propagates reshaping operations by expansion.";
  let description = [{
    Pass to propagate reshapes by expansion through all ops without explicit
    lowering configurations.
  }];
}

def RematerializeParallelOpsPass :
    InterfacePass<"iree-codegen-rematerialize-parallel-ops", "mlir::FunctionOpInterface"> {
  let summary = "Pass to rematerialize and merge parallel ops into consumers.";
}

def RemoveSingleIterationLoopPass :
    InterfacePass<"iree-codegen-remove-single-iteration-loop", "mlir::FunctionOpInterface"> {
  let summary = "Remove distributed loop with single iteration.";
}

def ResolveSwizzleHintsPass :
    InterfacePass<"iree-codegen-resolve-swizzle-hints", "mlir::FunctionOpInterface"> {
  let summary = "Resolves iree_codegen.swizzle_hint ops";
  let dependentDialects = [
    "affine::AffineDialect",
    "arith::ArithDialect",
  ];
}

def SpecializeExportsPass :
    Pass<"iree-codegen-specialize-exports", "IREE::HAL::ExecutableVariantOp">{
   let summary = "Specializes exported functions based on annotated ranges";
}

def StripCompilationInfoPass :
    Pass<"iree-codegen-strip-compilation-info", "">{
   let summary = "Remove all the the lowering configuration and translation info attributes.";
}

// TODO: Replace with upstream: https://github.com/iree-org/iree/issues/18759
def IREELoopInvariantCodeMotionPass :
  Pass<"iree-loop-invariant-code-motion", ""> {
  let summary = "Performs LICM on loops guaranteed to have >= 1 trip";
  let description = [{
    This is a mirror of the upstream LICM pass that restricts to loops that are
    guaranteed to have at least one trip. This currently only supports loops
    that expose a lower and upper bound as the generic loop-like interface does
    not expose a way to query for trip count.

    Additionally code motion of `scf.forall` ops with mappings is always unsafe
    and is explicitly disabled.
  }];
}

def SplitFullPartialTransferPass :
    InterfacePass<"iree-codegen-split-full-partial-transfer", "mlir::FunctionOpInterface"> {
  let summary =
      "Split a vector.transfer operation into an in-bounds (i.e., no "
      "out-of-bounds masking) fastpath and a slowpath.";
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

def TensorToVectorVectorizePadPass :
    InterfacePass<"iree-codegen-vectorize-tensor-pad", "mlir::FunctionOpInterface"> {
  let summary = "Vectorize a very specific form of tensor.pad with "
                "control flows";
}

def TestExecutablePreprocessingPass :
    Pass<"iree-codegen-test-executable-preprocessing", ""> {
  let summary = "Tests iree-hal-preprocess-executables-with behavior.";
}

def TestPartitionableLoopsInterfacePass :
    Pass<"iree-codegen-test-partitionable-loops-interface", ""> {
  let summary = "Test the PartitionableLoopsInterface";
}

def TileAndDistributeToWorkgroupsPass :
    InterfacePass<"iree-codegen-tile-and-distribute-to-workgroups", "mlir::FunctionOpInterface"> {
  let summary = "Tile and distribute operations to workgroups";
  let options = [
    Option<"maxWorkgroupParallelDims", "max-workgroup-parallel-dims", "int32_t",
      /*default=*/ "kNumMaxParallelDims",
      "Maximum number of dims to distribute workgroups across.">,
    Option<"distributionMethod", "distribution-method", "linalg::DistributionMethod",
      /*default=*/ "linalg::DistributionMethod::Cyclic",
      "Pick the distribution method. See linalg::DistributionMethod for details",
      [{::llvm::cl::values(
            clEnumValN(linalg::DistributionMethod::Cyclic,
                       "0", "Use Cyclic strategy"),
            clEnumValN(linalg::DistributionMethod::CyclicNumProcsGeNumIters,
                       "1", "Use CyclicNumProcGeNumIter strategy"),
            clEnumValN(linalg::DistributionMethod::CyclicNumProcsEqNumIters,
                       "2", "Use CyclicNumProcEqNumIter strategy"),
            clEnumValN(linalg::DistributionMethod::None,
                       "3", "Use None strategy")
        )}]>,
  ];
}

def TileAndDistributeToWorkgroupsUsingForallOpPass :
    InterfacePass<"iree-codegen-tile-and-distribute-to-workgroups-using-forall-op",
                  "mlir::FunctionOpInterface"> {
  let summary = "Tile and distribute operation to workgroups (using scf.forall op)";
  let dependentDialects = [
    "affine::AffineDialect",
    "IREE::Codegen::IREECodegenDialect",
    "IREE::LinalgExt::IREELinalgExtDialect",
    "scf::SCFDialect",
    "tensor::TensorDialect",
  ];
  let options = [
    Option<"transposeWorkgroup", "transpose-workgroup", "bool", /*default=*/"false",
           "Swaps the workgroup mapping attribute x and y."
            "Only swaps when the loop bounds are static.">,
  ];
}

def TileLargeTensorsPass :
    InterfacePass<"iree-codegen-tile-large-tensors", "mlir::FunctionOpInterface"> {
  let summary = "Greedily tiles all linalg ops that are beyond a certain size";
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::affine::AffineDialect",
    "::mlir::scf::SCFDialect",
  ];
  let options = [
    Option<"maxVectorSize", "max-vector-size", "int64_t",
           /*default=*/"64",
           "Maximum static size to tile to (i.e. all remaining ops will be smaller)">,
  ];
}

def TransformDialectInterpreterPass :
    Pass<"iree-transform-dialect-interpreter"> {
  let summary = "Pass to apply transform dialect operations.";
  let description = [{
    This pass runs the transform dialect interpreter and applies the named
    sequence transformation specified by the provided name (defaults to
    `TransformDialect::kTransformEntryPointSymbolName` (i.e. `__transform_main`)).
  }];
  let options = [
    Option<"entryPoint", "entry-point", "std::string",
           /*default=*/"::mlir::transform::TransformDialect::kTransformEntryPointSymbolName.str()",
           "Entry point of the pass pipeline.">,
    Option<"libraryFileName", "library-file-name", "std::string",
           /*default=*/[{""}],
           "File path to load a library of transform dialect strategies from.">,
  ];
}

def TypePropagationPass :
    InterfacePass<"iree-codegen-type-propagation", "mlir::FunctionOpInterface"> {
  let summary = "Propogate the type of tensor to avoid load/stores of illegal bit widths";
}

def UnrollAnnotatedLoopsPass :
    InterfacePass<"iree-codegen-unroll-annotated-loops", "mlir::FunctionOpInterface"> {
  let summary = "Unrolls all scf.for loops marked with `unroll_loop`";
}

def VectorizeMemrefCopyPass :
    Pass<"iree-codegen-vectorize-memref-copy", ""> {
  let summary = "Vectorizes memref copy operations.";
}

def VectorTransferLoweringPass :
    InterfacePass<"iree-codegen-vector-transfer-lowering", "mlir::FunctionOpInterface"> {
  let summary = "Pass to lower transfer ops to simpler ops like `vector.load`, `vector.store`, `vector.broadcast`, and a set of scf ops.";
  let options = [
    Option<"enableScalableLowerings", "enable-scalable-lowerings", "bool",
      /*default=*/"false",
      "Enables scalable vector specific transfer lowerings">,
  ];
}

def VerifyWorkgroupDistributionPass :
    InterfacePass<"iree-codegen-verify-workgroup-distribution", "mlir::FunctionOpInterface"> {
  let summary = "Pass to verify proper distribution to workgroups.";
  let description = [{
    Pass to verify that all writes to global memory are explicitly mapped to
    workgroups. This means that in cases where we use loops (scf.forall) to
    manage distribution to workgroups, we require that all ops with write
    side effects are contained within a workgroup distributed loop.
  }];
}

#endif // IREE_CODEGEN_COMMON_PASSES

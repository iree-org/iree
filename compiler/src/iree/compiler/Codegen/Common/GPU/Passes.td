// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_COMMON_GPU_PASSES
#define IREE_CODEGEN_COMMON_GPU_PASSES

include "mlir/Pass/PassBase.td"

//===---------------------------------------------------------------------===//
// Common Passes used for GPU-like backends (keep alphabetical)
//===---------------------------------------------------------------------===//

def DecomposeHorizontallyFusedGemmsPass :
    InterfacePass<"iree-codegen-gpu-decompose-horizontally-fused-gemms",
                  "mlir::FunctionOpInterface"> {
  let summary =
      "Decomposes a horizontally fused GEMM back into its constituent GEMMs";
  let dependentDialects = [
    "::mlir::iree_compiler::IREE::GPU::IREEGPUDialect",
    "::mlir::linalg::LinalgDialect",
  ];
}

def GPUCheckResourceUsagePass :
    InterfacePass<"iree-codegen-gpu-check-resource-usage", "mlir::FunctionOpInterface"> {
  let summary = "Checks GPU specific resource usage constraints like shared memory limits";
  let constructor = "mlir::iree_compiler::createGPUCheckResourceUsagePass()";
}

def GPUCombineLayoutTransformationPass :
    InterfacePass<"iree-codegen-gpu-combine-layout-transformation", "mlir::FunctionOpInterface"> {
  let summary =
    "Combines layout transformation operations into a single map_scatter operation.";
  let description = [{
    Starting from iree_codegen.store_to_buffer ops, iteratively combine producer
    layout/indexing transformation ops (linalg.transpose, tensor.collapse_shape,
    etc.) into a single iree_linalg_ext.map_scatter operation. For tensor.pad
    ops, the writing of pad values is distributed to workgroups and threads, and
    then the padding values are written directly to the output buffer of the
    store_to_buffer op.
  }];
  let dependentDialects = [
    "iree_compiler::IREE::LinalgExt::IREELinalgExtDialect",
    "gpu::GPUDialect",
    "scf::SCFDialect",
    "tensor::TensorDialect"
  ];
}

def GPUCombineValueBarriersPass :
    Pass<"iree-codegen-gpu-combine-value-barriers", ""> {
  let summary = "Combines `iree_gpu.value_barrier` ops";
  let dependentDialects = ["::mlir::iree_compiler::IREE::GPU::IREEGPUDialect"];
}

def GPUCreateFastSlowPathPass :
    InterfacePass<"iree-codegen-gpu-create-fast-slow-path", "mlir::FunctionOpInterface"> {
  let summary = "Create separate fast and slow paths to handle padding";
  let dependentDialects = ["::mlir::scf::SCFDialect"];
}

def GPUDistributeCopyUsingForallPass :
    InterfacePass<"iree-codegen-gpu-distribute-copy-using-forall", "mlir::FunctionOpInterface"> {
  let summary = "Pass to distribute copies to threads.";
  let dependentDialects = [
    "::mlir::affine::AffineDialect", "::mlir::gpu::GPUDialect", "::mlir::scf::SCFDialect"
  ];
}

def GPULowerToGlobalLoadsPass :
    InterfacePass<"iree-codegen-gpu-lower-to-global-loads", "mlir::FunctionOpInterface"> {
  let summary = "Emit direct global loads instructions.";
  let dependentDialects = [
    "::mlir::affine::AffineDialect", "::mlir::gpu::GPUDialect", "::mlir::scf::SCFDialect",
    "::mlir::iree_compiler::IREE::GPU::IREEGPUDialect"
  ];
}

def GPUDistributeForallPass :
    InterfacePass<"iree-codegen-gpu-distribute-forall", "mlir::FunctionOpInterface"> {
  let summary = "Pass to distribute scf.forall ops.";
  let dependentDialects = [
    "::mlir::affine::AffineDialect",
    "::mlir::gpu::GPUDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::iree_compiler::IREE::GPU::IREEGPUDialect",
  ];
}

def GPUDistributePass :
    InterfacePass<"iree-codegen-gpu-distribute", "mlir::FunctionOpInterface"> {
  let summary = "Pass to distribute scf.forall ops using upstream patterns.";
  let dependentDialects = [
    "::mlir::affine::AffineDialect",
    "::mlir::gpu::GPUDialect",
    "::mlir::iree_compiler::IREE::GPU::IREEGPUDialect",
  ];
}

def GPUDistributeSharedMemoryCopyPass :
    InterfacePass<"iree-codegen-gpu-distribute-shared-memory-copy", "mlir::FunctionOpInterface"> {
  let summary = "Pass to distribute shared memory copies to threads.";
  let dependentDialects = [
    "::mlir::gpu::GPUDialect", "::mlir::scf::SCFDialect", "::mlir::vector::VectorDialect"
  ];
}

def GPUDistributeScfForPass :
    InterfacePass<"iree-codegen-gpu-distribute-scf-for", "mlir::FunctionOpInterface"> {
  let summary = "Distribute tiled loop nests to invocations";
  let dependentDialects = ["::mlir::gpu::GPUDialect"];
  let options = [
    Option<"useBlockDims", "use-block-dims", "bool",
           /*default=*/"true",
           "Use gpu.block_dim ops to query distribution sizes.">,
  ];
}

def GPUBubbleResourceCastsPass :
    Pass<"iree-codegen-gpu-bubble-resource-casts", ""> {
  let summary = "Bubbles iree_gpu.buffer_resource_cast ops upwards.";
  let dependentDialects = [
    "::mlir::iree_compiler::IREE::GPU::IREEGPUDialect"
  ];
}

def GPUFuseAndHoistParallelLoopsPass :
    InterfacePass<"iree-codegen-gpu-fuse-and-hoist-parallel-loops", "mlir::FunctionOpInterface"> {
  let summary = "Greedily fuses and hoists parallel loops.";
  let dependentDialects = [
    "::mlir::affine::AffineDialect",
    "::mlir::iree_compiler::IREE::GPU::IREEGPUDialect",
    "::mlir::bufferization::BufferizationDialect"
  ];
}

def GPUGeneralizeNamedOpsPass :
    InterfacePass<"iree-codegen-gpu-generalize-named-ops", "mlir::FunctionOpInterface"> {
  let summary = "Convert named Linalg ops to linalg.generic ops";
}

def GPUGreedilyDistributeToThreadsPass :
    InterfacePass<"iree-codegen-gpu-greedily-distribute-to-threads", "mlir::FunctionOpInterface"> {
  let summary = "Greedily distributes all remaining tilable ops to threads";
  let dependentDialects = [
    "::mlir::affine::AffineDialect",
    "::mlir::gpu::GPUDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::iree_compiler::IREE::GPU::IREEGPUDialect",
  ];
}

def GPUInferMemorySpacePass :
    InterfacePass<"iree-codegen-gpu-infer-memory-space", "mlir::FunctionOpInterface"> {
  let summary = "Pass to infer and set the memory space for all alloc_tensor ops.";
  let dependentDialects = [
    "::mlir::gpu::GPUDialect"
  ];
}

def GPULowerToUKernelsPass :
    Pass<"iree-codegen-gpu-lower-to-ukernels", ""> {
  let summary = "Lower suitable ops to previously-selected microkernels";
  let dependentDialects = [
    "::mlir::iree_compiler::IREE::Codegen::IREECodegenDialect",
    "::mlir::iree_compiler::IREE::GPU::IREEGPUDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::bufferization::BufferizationDialect",
    "::mlir::gpu::GPUDialect",
    "::mlir::tensor::TensorDialect",
  ];
}

def GPUMultiBufferingPass :
    InterfacePass<"iree-codegen-gpu-multi-buffering", "mlir::FunctionOpInterface"> {
  let summary = "Pass to do multi buffering.";
  let dependentDialects = ["::mlir::affine::AffineDialect"];
  let options = [
    Option<"numBuffers", "num-buffers", "unsigned",
            /*default=*/"5",
            "Number of buffers to use.">,
  ];
}

def GPUPackToIntrinsicsPass :
    InterfacePass<"iree-codegen-gpu-pack-to-intrinsics", "mlir::FunctionOpInterface"> {
  let summary = "Packs matmul like operations and converts to iree_gpu.multi_mma";
  let dependentDialects = [
    "::mlir::tensor::TensorDialect",
    "::mlir::iree_compiler::IREE::GPU::IREEGPUDialect"
  ];
}

def GPUPadOperandsPass :
    InterfacePass<"iree-codegen-gpu-pad-operands",
                  "mlir::FunctionOpInterface"> {
  let summary = "Pass to pad operands of ops with padding configuration provided. ";
  let dependentDialects = [
    "::mlir::linalg::LinalgDialect",
    "::mlir::iree_compiler::IREE::GPU::IREEGPUDialect"
  ];
}

def GPUPipeliningPass :
    InterfacePass<"iree-codegen-gpu-pipelining", "mlir::FunctionOpInterface"> {
  let summary = "Pass to do software pipelining.";
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

def GPUPromoteMatmulOperandsPass :
    InterfacePass<"iree-codegen-gpu-promote-matmul-operands",
                  "mlir::FunctionOpInterface"> {
  let summary = "Pass to insert copies with a different thread configuration "
                "on matmul operands";
  let dependentDialects = [
    "::mlir::bufferization::BufferizationDialect",
    "::mlir::gpu::GPUDialect",
    "::mlir::linalg::LinalgDialect",
    "::mlir::iree_compiler::IREE::GPU::IREEGPUDialect"
  ];
}

def GPUAllocPrivateMemoryForDPSOpsPass :
    InterfacePass<"iree-codegen-gpu-alloc-private-memory-for-dps-ops", "mlir::FunctionOpInterface"> {
  let summary = "Pass to add private memory allocations prior to DPS interface ops.";
  let description = [{
    Creates a `bufferization.alloc_tensor` in private space for all DPS ops
    with unused results that can't be removed. These unused results, if
    originating from loads from global memory, trigger allocations in global
    memory space during bufferization, which will fail. So, the allocations
    must be made earlier to avoid failed bufferization.
  }];
  let dependentDialects = [
    "::mlir::bufferization::BufferizationDialect",
    "::mlir::gpu::GPUDialect"
  ];
}

def GPUReduceBankConflictsPass :
    InterfacePass<"iree-codegen-gpu-reduce-bank-conflicts", "mlir::FunctionOpInterface"> {
  let summary = "Pass to try to reduce the number of bank conflicts by padding memref.alloc ops.";
  let options = [
    Option<"paddingBits", "padding-bits", "unsigned",
            /*default=*/"128",
            "Padding size (in bits) to introduce between rows.">
  ];
}

def GPUReuseSharedMemoryAllocsPass :
    InterfacePass<"iree-codegen-gpu-reuse-shared-memory-allocs", "mlir::FunctionOpInterface"> {
  let summary = "Pass to reuse shared memory allocations with no overlapping liveness.";
  let dependentDialects = [
    "::mlir::nvgpu::NVGPUDialect"
  ];
}

def GPUTensorAllocPass :
    InterfacePass<"iree-codegen-gpu-tensor-alloc", "mlir::FunctionOpInterface"> {
  let summary = "Pass to create allocations for some tensor values to use"
                "GPU shared memory";
  let constructor = "mlir::iree_compiler::createGPUTensorAlloc()";
  let dependentDialects = ["::mlir::bufferization::BufferizationDialect"];
}

def GPUTensorTilePass :
    InterfacePass<"iree-codegen-gpu-tensor-tile", "mlir::FunctionOpInterface"> {
  let summary = "Pass to tile tensor (linalg) ops within a GPU workgroup";
  let dependentDialects = [
    "::mlir::affine::AffineDialect", "::mlir::gpu::GPUDialect", "::mlir::scf::SCFDialect"
  ];
  let options = [
    Option<"distributeToSubgroup", "distribute-to-subgroup", "bool",
           /*default=*/"false",
           "Distribute the workloads to subgroup if true, otherwise distribute to threads.">,
  ];
}

def GPUApplyTilingLevelPass :
    InterfacePass<"iree-codegen-gpu-apply-tiling-level", "mlir::FunctionOpInterface"> {
  let summary = "Pass to tile tensor ops based on tiling configs";
  let dependentDialects = [
    "::mlir::affine::AffineDialect", "::mlir::gpu::GPUDialect",
    "::mlir::scf::SCFDialect", "::mlir::tensor::TensorDialect"
  ];
  let options = [
    Option<"tilingLevel", "tiling-level", "IREE::GPU::TilingLevel",
           /*default=*/"IREE::GPU::TilingLevel::Reduction",
           "Tiling level to tile. Supported levels are 'reduction' and 'thread'",
           [{llvm::cl::values(
              clEnumValN(IREE::GPU::TilingLevel::Reduction, "reduction",
                         "Tile and fuse all annotated ops to serial loops"),
              clEnumValN(IREE::GPU::TilingLevel::PartialReduction, "partial_reduction",
                         "Tile and fuse all annotated ops to partial reduuction loops"),
              clEnumValN(IREE::GPU::TilingLevel::Thread, "thread",
                         "Tile and fuse all annotated ops to threads"),
              clEnumValN(IREE::GPU::TilingLevel::Subgroup, "subgroup",
                         "Tile and fuse all annotated ops to threads")
           )}]>,
    Option<"allowZeroSlices", "allow-zero-slices", "bool",
           /*default=*/"true",
           "Allow pad fusion to generate zero size slices">
  ];
}

def GPUApplyPaddingLevelPass :
    InterfacePass<"iree-codegen-gpu-apply-padding-level", "mlir::FunctionOpInterface"> {
  let summary = "Pass to pad based on tiling configs";
  let dependentDialects = ["::mlir::tensor::TensorDialect"];
  let options = [
    Option<"tilingLevel", "tiling-level", "IREE::GPU::TilingLevel",
           /*default=*/"IREE::GPU::TilingLevel::Reduction",
           "Tiling level to tile. Supported levels are 'reduction' and 'thread'",
           [{llvm::cl::values(
              clEnumValN(IREE::GPU::TilingLevel::Reduction, "reduction",
                         "Tile and fuse all annotated ops to serial loops"),
              clEnumValN(IREE::GPU::TilingLevel::PartialReduction, "partial_reduction",
                         "Tile and fuse all annotated ops to partial reduuction loops")
           )}]>
  ];
}

def GPUTensorTileToSerialLoopsPass :
    InterfacePass<"iree-codegen-gpu-tensor-tile-to-serial-loops", "mlir::FunctionOpInterface"> {
  let summary = "Pass to tile reduction dimensions for certain GPU ops";
  let dependentDialects = ["::mlir::scf::SCFDialect"];
  let options = [
    Option<"coalesceLoops", "coalesce-loops", "bool", /*default=*/"false",
           "Collapse the loops that are generated to a single loops">,
  ];
}

def GPUTilePass : InterfacePass<"iree-codegen-gpu-tile", "mlir::FunctionOpInterface"> {
  let summary = "Tile Linalg ops with tensor semantics to invocations";
}

def GPUTileReductionPass :
    InterfacePass<"iree-codegen-gpu-tile-reduction", "mlir::FunctionOpInterface"> {
  let summary = "Pass to tile linalg reduction dimensions.";
  let dependentDialects = ["::mlir::scf::SCFDialect"];
}

def GPUVerifyDistributionPass :
    InterfacePass<"iree-codegen-gpu-verify-distribution", "mlir::FunctionOpInterface"> {
  let summary = "Pass to verify writes before resolving distributed contexts.";
  let dependentDialects = [
    "::mlir::gpu::GPUDialect",
  ];
}

def GPUVectorAllocPass :
    InterfacePass<"iree-codegen-gpu-vector-alloc", "mlir::FunctionOpInterface"> {
  let summary = "Pass to create allocations for contraction inputs to copy "
                "to GPU shared memory";
  let dependentDialects = [
    "::mlir::gpu::GPUDialect",
    "::mlir::vector::VectorDialect",
    "::mlir::bufferization::BufferizationDialect",
    "::mlir::iree_compiler::IREE::GPU::IREEGPUDialect",
  ];
}

def ReorderWorkgroupsPass :
    InterfacePass<"iree-codegen-reorder-workgroups", "mlir::FunctionOpInterface"> {
  let summary = "Reorder workgroup ids for better cache reuse";
  let constructor = "mlir::iree_compiler::createReorderWorkgroups()";
  let dependentDialects = ["::mlir::affine::AffineDialect"];
  let options = [
    Option<"strategy", "strategy", "std::string", /*default=*/"",
           "Workgroup reordering strategy, one of: '' (none),  'transpose'">,
  ];
}

def VectorReductionToGPUPass :
    InterfacePass<"iree-codegen-vector-reduction-to-gpu", "mlir::FunctionOpInterface"> {
  let summary = "Convert vector reduction to GPU ops.";
  let dependentDialects = [
    "::mlir::affine::AffineDialect", "::mlir::gpu::GPUDialect",
    "::mlir::memref::MemRefDialect", "::mlir::scf::SCFDialect",
  ];
  let options = [
    Option<"expandSubgroupReduction", "expand-subgroup-reduction", "bool",
           /*default=*/"true",
           "Lower subgroup reductions to gpu ops immediately where possible.">,
  ];
}

def ExpandGPUOpsPass :
    InterfacePass<"iree-codegen-expand-gpu-ops", "mlir::FunctionOpInterface"> {
  let summary = "Expands high-level GPU ops, such as clustered gpu.subgroup_reduce.";
  let dependentDialects = [
    "::mlir::gpu::GPUDialect"
  ];
}

#endif // IREE_CODEGEN_COMMON_GPU_PASSES

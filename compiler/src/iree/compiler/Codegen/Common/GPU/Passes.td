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

def GPUCheckResourceUsage :
    Pass<"iree-codegen-gpu-check-resource-usage", "ModuleOp"> {
  let summary = "Checks GPU specific resource usage constraints like shared memory limits";
  let constructor = "mlir::iree_compiler::createGPUCheckResourceUsagePass()";
}

def GPUDistribute :
    Pass<"iree-codegen-gpu-distribute", "func::FuncOp"> {
  let summary = "Pass to distribute scf.forall ops.";
  let constructor = "mlir::iree_compiler::createGPUDistribute()";
}

def GPUDistributeSharedMemoryCopy :
    Pass<"iree-codegen-gpu-distribute-shared-memory-copy", "func::FuncOp"> {
  let summary = "Pass to distribute shared memory copies to threads.";
  let constructor = "mlir::iree_compiler::createGPUDistributeSharedMemoryCopy()";
}

def GPUGeneralizeNamedOps :
    Pass<"iree-codegen-gpu-generalize-named-ops", "func::FuncOp"> {
  let summary = "Convert named Linalg ops to linalg.generic ops";
  let constructor = "mlir::iree_compiler::createGPUGeneralizeNamedOpsPass()";
}

def GPULowerToUKernels :
    Pass<"iree-codegen-gpu-lower-to-ukernels", ""> {
  let summary =
      "Separate out parts of the IR that lower to a micro-kernel";
  let constructor =
      "mlir::iree_compiler::createGPULowerToUKernelsPass()";
}

def GPUMultiBuffering :
    Pass<"iree-codegen-gpu-multi-buffering", "func::FuncOp"> {
  let summary = "Pass to do multi buffering.";
  let constructor = "mlir::iree_compiler::createGPUMultiBuffering()";
}

def GPUPipelining : Pass<"iree-codegen-gpu-pipelining", "func::FuncOp"> {
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

def GPUReduceBankConflicts :
    Pass<"iree-codegen-gpu-reduce-bank-conflicts", "func::FuncOp"> {
  let summary = "Pass to try to reduce the number of bank conflicts.";
  let constructor = "mlir::iree_compiler::createGPUReduceSharedMemoryBankConflicts()";
}

def GPUSynchronizeBuffers :
    Pass<"iree-codegen-gpu-synchronize-buffers", "func::FuncOp"> {
  let summary = "Pass to insert barriers for buffers marked with a #requires_sync memory space attribute.";
  let description = [{
    Insert gpu.barrier operations based on a Read-After-Write and
    Write-After-Read analysis. The barrier operations are only inserted
    for memrefs with a #requires_sync memory space.
    
    The user is expected to have set the memory space on the memref with
    #requires_sync, which takes the address space and a barrier token as
    parameters.
     
    The barrier token is used to determine which buffers need to be synchronized
    for RaW and WaR conflicts i.e. only buffers with same barrier token can
    have conflicts with each other. If no barrier token is specified, the
    buffer is assumed to conflict with everything.

    The pass will replace #requires_sync memory spaces with the memory space
    given in #requires_sync attribute.

    Example:

    ```
    %mem = memref.alloc() 
      : memref<10x10xf32, #iree_linalg_ext.requires_sync<#gpu.address_space<workgroup>, 1>>
    %a = transfer_read %mem[0, 0] : vector<10xf32>
    transfer_write %a, %mem
    ```

    The pass will convert his to

    ```
    %mem = memref.alloc() 
      : memref<10x10xf32, #gpu.address_space<workgroup>>
    %a = transfer_read %mem[0, 0] : vector<10xf32>
    gpu.barrier
    transfer_write %a, %mem
    ```
  }];
  let constructor = "mlir::iree_compiler::createGPUSynchronizeBuffersPass()";
}

def GPUTensorAlloc :
    Pass<"iree-codegen-gpu-tensor-alloc", "func::FuncOp"> {
  let summary = "Pass to tile reduction dimensions and create allocations for "
                "some tensor values to use GPU shared memory";
  let constructor = "mlir::iree_compiler::createGPUTensorAlloc()";
}

def GPUTensorTile :
    Pass<"iree-codegen-gpu-tensor-tile", "func::FuncOp"> {
  let summary = "Pass to tile tensor (linalg) ops within a GPU workgroup";
  let constructor = "mlir::iree_compiler::createGPUTensorTile()";
}

def GPUTileReduction :
    Pass<"iree-codegen-gpu-tile-reduction", "func::FuncOp"> {
  let summary = "Pass to tile linalg reduction dimensions.";
  let constructor = "mlir::iree_compiler::createGPUTileReductionPass()";
}

def VectorReductionToGPU :
    Pass<"iree-codegen-vector-reduction-to-gpu", "func::FuncOp"> {
  let summary = "Convert vector reduction to GPU ops.";
  let constructor = "mlir::iree_compiler::createConvertVectorReductionToGPUPass()";
}

def WorkgroupSpecialization :
    Pass<"iree-codegen-workgroup-specialization", "func::FuncOp"> {
  let summary = "Specialize workgroup distribution loops";
  let constructor = "mlir::iree_compiler::createWorkgroupSpecializationPass()";
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

#endif // IREE_CODEGEN_COMMON_GPU_PASSES

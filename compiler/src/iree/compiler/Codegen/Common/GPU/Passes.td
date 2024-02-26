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
    InterfacePass<"iree-codegen-gpu-distribute", "mlir::FunctionOpInterface"> {
  let summary = "Pass to distribute scf.forall ops.";
  let constructor = "mlir::iree_compiler::createGPUDistribute()";
}

def GPUDistributeSharedMemoryCopy :
    InterfacePass<"iree-codegen-gpu-distribute-shared-memory-copy", "mlir::FunctionOpInterface"> {
  let summary = "Pass to distribute shared memory copies to threads.";
  let constructor = "mlir::iree_compiler::createGPUDistributeSharedMemoryCopy()";
}

def GPUGeneralizeNamedConvolutionOps :
    InterfacePass<"iree-codegen-gpu-generalize-named-convolution-ops", "mlir::FunctionOpInterface"> {
  let summary = "Convert named Linalg convolution ops to linalg.generic ops";
  let constructor = "mlir::iree_compiler::createGPUGeneralizeNamedConvolutionOpsPass()";
}

def GPUGeneralizeNamedOps :
    InterfacePass<"iree-codegen-gpu-generalize-named-ops", "mlir::FunctionOpInterface"> {
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
    InterfacePass<"iree-codegen-gpu-multi-buffering", "mlir::FunctionOpInterface"> {
  let summary = "Pass to do multi buffering.";
  let constructor = "mlir::iree_compiler::createGPUMultiBuffering()";
}

def GPUPipelining : InterfacePass<"iree-codegen-gpu-pipelining", "mlir::FunctionOpInterface"> {
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
    InterfacePass<"iree-codegen-gpu-reduce-bank-conflicts", "mlir::FunctionOpInterface"> {
  let summary = "Pass to try to reduce the number of bank conflicts.";
  let constructor = "mlir::iree_compiler::createGPUReduceSharedMemoryBankConflicts()";
}

def GPUTensorAlloc :
    InterfacePass<"iree-codegen-gpu-tensor-alloc", "mlir::FunctionOpInterface"> {
  let summary = "Pass to create allocations for some tensor values to use"
                "GPU shared memory";
  let constructor = "mlir::iree_compiler::createGPUTensorAlloc()";
}

def GPUTensorTile :
    InterfacePass<"iree-codegen-gpu-tensor-tile", "mlir::FunctionOpInterface"> {
  let summary = "Pass to tile tensor (linalg) ops within a GPU workgroup";
  let constructor = "mlir::iree_compiler::createGPUTensorTile()";
}

def GPUTensorTileToSerialLoops :
    InterfacePass<"iree-codegen-gpu-tensor-tile-to-serial-loops", "mlir::FunctionOpInterface"> {
  let summary = "Pass to tile reduction dimensions for certain GPU ops";
  let constructor = "mlir::iree_compiler::createGPUTensorTileToSerialLoops()";
}

def GPUTileReduction :
    InterfacePass<"iree-codegen-gpu-tile-reduction", "mlir::FunctionOpInterface"> {
  let summary = "Pass to tile linalg reduction dimensions.";
  let constructor = "mlir::iree_compiler::createGPUTileReductionPass()";
}

def GPUVectorAlloc :
    InterfacePass<"iree-codegen-gpu-vector-alloc", "mlir::FunctionOpInterface"> {
  let summary = "Pass to create allocations for contraction inputs to copy "
                "to GPU shared memory";
  let constructor = "mlir::iree_compiler::createGPUVectorAlloc()";
}

def VectorReductionToGPU :
    InterfacePass<"iree-codegen-vector-reduction-to-gpu", "mlir::FunctionOpInterface"> {
  let summary = "Convert vector reduction to GPU ops.";
  let constructor = "mlir::iree_compiler::createConvertVectorReductionToGPUPass()";
}

def WorkgroupSpecialization :
    InterfacePass<"iree-codegen-workgroup-specialization", "mlir::FunctionOpInterface"> {
  let summary = "Specialize workgroup distribution loops";
  let constructor = "mlir::iree_compiler::createWorkgroupSpecializationPass()";
}

def WorkGroupSwizzle :
    InterfacePass<"iree-workgroup-swizzle", "mlir::FunctionOpInterface"> {
  let summary = "swizzle the workgroup ids for better cache reuse";
  let constructor = "mlir::iree_compiler::createWorkGroupSwizzle()";
  let options = [
    Option<"logTile", "logTile", "unsigned",
            /*default=*/"0",
           "pass the tile value for unit testing">,
  ];
}

#endif // IREE_CODEGEN_COMMON_GPU_PASSES

// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_SPIRV_PASSES
#define IREE_CODEGEN_SPIRV_PASSES

include "mlir/Pass/PassBase.td"

//===---------------------------------------------------------------------===//
// SPIR-V passes (keep alphabetical)
//===---------------------------------------------------------------------===//

def ConvertToSPIRVPass : Pass<"iree-convert-to-spirv", "ModuleOp"> {
  let summary = "Perform the final conversion to SPIR-V dialect";
  let options = [
    Option<"indexBitsOption", "index-bits", "unsigned", /*default=*/"32",
          "Specify the bit widths for SPIR-V indices">,
  ];
}

def SPIRVAnnotateWinogradLoopsPass : InterfacePass<"iree-spirv-annotate-winograd-loops", "mlir::FunctionOpInterface"> {
  let summary = "Annotate innermost Winograd loops with spirv distribute attribute";
}

def SPIRVBreakDownLargeVectorPass : InterfacePass<"iree-spirv-breakdown-large-vector",
                                                  "mlir::FunctionOpInterface"> {
  let summary = "Break down vectors not natively supported by SPIR-V";
}

def SPIRVConvertGPUTargetPass : Pass<"iree-spirv-convert-gpu-target", "mlir::ModuleOp"> {
  let summary = "Convert #iree_gpu.target into #spirv.target_env";
}

def SPIRVEmulateI64Pass :
    InterfacePass<"iree-spirv-emulate-i64", "mlir::FunctionOpInterface"> {
  let summary = "Emulate 64-bit integer ops with 32-bit integer ops";
}

def SPIRVEraseStorageBufferStaticShapePass :
    InterfacePass<"iree-spirv-erase-storage-buffer-static-shape", "mlir::FunctionOpInterface"> {
  let summary = "Turn static shaped storage buffer subspan ops into dynamic shaped ones";
}

def SPIRVFinalVectorLoweringPass : InterfacePass<
    "iree-spirv-final-vector-lowering", "mlir::FunctionOpInterface"> {
  let summary = "Perform final lowering of vectors ops to fit SPIR-V";
}

def SPIRVInitialVectorLoweringPass : InterfacePass<
    "iree-spirv-initial-vector-lowering", "mlir::FunctionOpInterface"> {
  let summary = "Perform initial lowering of vectors ops to fit SPIR-V";
}

def SPIRVLinkExecutablesPass :
    Pass<"iree-spirv-link-executables", "mlir::ModuleOp"> {
  let summary = "Links SPIR-V HAL executables within the top-level program module.";
}

def SPIRVLowerExecutableTargetPass :
    InterfacePass<"iree-spirv-lower-executable-target-pass",
         "mlir::FunctionOpInterface"> {
  let summary = "Lower the executable target to SPIR-V using one of the "
                "IREE::HAL::DispatchLoweringPassPipeline";
  let description = [{
    Main pass to lower executables to scalar + vector code on SPIR-V path.
    Invokes one of the pass pipelines that translate the executable to
    scalar + vector code.
  }];
}

def SPIRVLowerExecutableUsingTransformDialectPass :
    Pass<"iree-spirv-lower-executable-using-transform-dialect", "ModuleOp"> {
  let summary = "Lower the executable target to SPIR-V using transform dialect"
                " followed by some passes to do SPIR-V specific vectorization";
  let description = [{
    Pass to lower executables using Transform dialect on the SPIR-V backend.
    This shouldnt be a separate pass, but it is since there are some
    extra spir-v passes that need to be run as well.
  }];
}

def SPIRVMapMemRefStorageClassPass :
    InterfacePass<"iree-spirv-map-memref-storage-class", "mlir::FunctionOpInterface"> {
  let summary = "Map MemRef memory spaces to SPIR-V storage classes";
}

def SPIRVMaterializeExecutableConditionsPass :
    Pass<"iree-spirv-materialize-executable-conditions",
         "mlir::iree_compiler::IREE::HAL::ExecutableVariantOp"> {
  let summary = "Materialize SPIR-V target requirements of hal.exectuable.variant "
                "ops into hal.executable.condition regions";
}

def SPIRVSelectLoweringStrategyPass :
    Pass<"iree-spirv-select-lowering-strategy-pass", "ModuleOp"> {
  let summary = "Select the IREE::HAL::DispatchLoweringPassPipeline for lowering"
                "to SPIR-V";
}

def SPIRVTileAndDistributePass : InterfacePass<"iree-spirv-tile-and-distribute", "mlir::FunctionOpInterface"> {
  let summary = "Tile and distribute Linalg ops with buffer semantics to "
                "invocations";
}

def SPIRVTileAndPromotePass : InterfacePass<"iree-spirv-tile-and-promote", "mlir::FunctionOpInterface"> {
  let summary = "Promote tiled Linalg ops with buffer semantics to use "
                "workgroup memory and then tile to invocations";
  let options = [
    Option<"promoteCMatrix", "promote-c", "bool", /*default=*/"false",
          "Promote C matrix to use shared memory">,
    Option<"skipThreadLevel", "skip-thread", "bool", /*default=*/"false",
          "Skip tiling and distributing to GPU threads">,
  ];
}

def SPIRVTileToCooperativeOpsPass : InterfacePass<
    "iree-spirv-tile-to-cooperative-ops", "mlir::FunctionOpInterface"> {
  let summary = "Tile Linalg ops with buffer semantics to subgroups and "
                "vectorize to vector ops suitable for lowering to SPIR-V "
                "cooperative ops";
}

def SPIRVTrimExecutableTargetEnvPass :
    Pass<"iree-spirv-trim-executable-target-env",
         "mlir::iree_compiler::IREE::HAL::ExecutableVariantOp"> {
  let summary = "Trim the SPIR-V target environment of a HAL executable "
                "variant to the minimal requirement per the compiled "
                "spirv.module op needs";
}

def SPIRVVectorizeLoadStorePass :
    InterfacePass<"iree-spirv-vectorize-load-store", "mlir::FunctionOpInterface"> {
  let summary = "Vectorize load/store of memrefs for better memory access";
  let description = [{
    Converts memref of scalar to memref of vector of efficent size. This will
    allow to convert memory accesses to vector load/store in SPIR-V without
    having pointer bitcast.
  }];
}

def SPIRVVectorizeToCooperativeOpsPass : InterfacePass<
    "iree-spirv-vectorize-to-cooperative-ops", "mlir::FunctionOpInterface"> {
  let summary = "Tile Linalg ops with buffer semantics to subgroups and "
                "vectorize to vector ops suitable for lowering to SPIR-V "
                "cooperative ops";
}

def SPIRVVectorToGPUSubgroupMMAPass :
    InterfacePass<"iree-spirv-vector-to-gpu-subgroup-mma-ops", "mlir::FunctionOpInterface"> {
  let summary = "Pass to convert vector ops to GPU subgroup MMA ops.";
}

#endif // IREE_CODEGEN_SPIRV_PASSES

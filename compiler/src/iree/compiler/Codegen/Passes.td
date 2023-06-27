// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_PASSES
#define IREE_CODEGEN_PASSES

include "mlir/Pass/PassBase.td"

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

#endif  // IREE_DIALECT_FLOW_PASSES

// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_SPIRV_PASSES
#define IREE_CODEGEN_SPIRV_PASSES

include "mlir/Pass/PassBase.td"

//===---------------------------------------------------------------------===//
// SPIRV passes (keep alphabetical)
//===---------------------------------------------------------------------===//

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

def SPIRVAnnotateWinogradLoops : Pass<"iree-spirv-annotate-winograd-loops", "func::FuncOp"> {
  let summary = "Annotate innermost Winograd loops with spirv distribute attribute";
  let constructor = "mlir::iree_compiler::createSPIRVAnnotateWinogradLoopsPass()";
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

def SPIRVDistribute : Pass<"iree-spirv-distribute", "func::FuncOp"> {
  let summary = "Distribute tiled loop nests to invocations";
  let constructor = "mlir::iree_compiler::createSPIRVDistributePass()";
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

def SPIRVLowerExecutableTarget :
    Pass<"iree-spirv-lower-executable-target-pass",
         "mlir::iree_compiler::IREE::HAL::ExecutableVariantOp"> {
  let summary = "Lower the executable target to SPIR-V using one of the "
                "IREE::HAL::DispatchLoweringPassPipeline";
  let constructor =
      "mlir::iree_compiler::createSPIRVLowerExecutableTargetPass()";
}

def SPIRVMapMemRefStorageClass :
    Pass<"iree-spirv-map-memref-storage-class", "func::FuncOp"> {
  let summary = "Map MemRef memory spaces to SPIR-V storage classes";
  let constructor = "mlir::iree_compiler::createSPIRVMapMemRefStorageClassPass()";
}

def SPIRVSelectLoweringStrategy :
    Pass<"iree-spirv-select-lowering-strategy-pass",
         "mlir::iree_compiler::IREE::HAL::ExecutableVariantOp"> {
  let summary = "Select the IREE::HAL::DispatchLoweringPassPipeline for lowering"
                "to SPIR-V";
  let constructor =
      "mlir::iree_compiler::createSPIRVSelectLoweringStrategyPass()";
}

def SPIRVTile : Pass<"iree-spirv-tile", "func::FuncOp"> {
  let summary = "Tile Linalg ops with tensor semantics to invocations";
  let constructor = "mlir::iree_compiler::createSPIRVTilePass()";
}

def SPIRVTileAndDistribute : Pass<"iree-spirv-tile-and-distribute", "func::FuncOp"> {
  let summary = "Tile and distribute Linalg ops with buffer semantics to "
                "invocations";
  let constructor = "mlir::iree_compiler::createSPIRVTileAndDistributePass()";
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

def SPIRVTileToCooperativeOps : Pass<
    "iree-spirv-tile-to-cooperative-ops", "func::FuncOp"> {
  let summary = "Tile Linalg ops with buffer semantics to subgroups and "
                "vectorize to vector ops suitable for lowering to SPIR-V "
                "cooperative ops";
  let constructor =
    "mlir::iree_compiler::createSPIRVTileToCooperativeOpsPass()";
}

def SPIRVInitialVectorLowering : Pass<
    "iree-spirv-initial-vector-lowering", "func::FuncOp"> {
  let summary = "Perform initial lowering of vectors ops to fit SPIR-V";
  let constructor = "mlir::iree_compiler::createSPIRVInitialVectorLoweringPass()";
}

def SPIRVFinalVectorLowering : Pass<
    "iree-spirv-final-vector-lowering", "func::FuncOp"> {
  let summary = "Perform final lowering of vectors ops to fit SPIR-V";
  let constructor = "mlir::iree_compiler::createSPIRVFinalVectorLoweringPass()";
}

def SPIRVVectorizeLoadStore :
    Pass<"iree-spirv-vectorize-load-store", "ModuleOp"> {
  let summary = "Vectorize load/store of memrefs for better memory access";
  let constructor = "mlir::iree_compiler::createSPIRVVectorizeLoadStore()";
}

def SPIRVVectorizeToCooperativeOps : Pass<
    "iree-spirv-vectorize-to-cooperative-ops", "func::FuncOp"> {
  let summary = "Tile Linalg ops with buffer semantics to subgroups and "
                "vectorize to vector ops suitable for lowering to SPIR-V "
                "cooperative ops";
  let constructor =
    "mlir::iree_compiler::createSPIRVVectorizeToCooperativeOpsPass()";
}

def SPIRVVectorToGPUSubgroupMMA :
    Pass<"iree-spirv-vector-to-gpu-subgroup-mma-ops", "func::FuncOp"> {
  let summary = "Pass to convert vector ops to GPU subgroup MMA ops.";
  let constructor = "mlir::iree_compiler::createSPIRVVectorToGPUSubgroupMMAOpsPass()";
}

#endif // IREE_CODEGEN_SPIRV_PASSES

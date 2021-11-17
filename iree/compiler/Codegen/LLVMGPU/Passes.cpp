// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Passes.h"

#include "iree-dialects/Dialect/LinalgExt/Transforms/Passes.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Dialect/Shape/Transforms/Passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/VectorToGPU/VectorToGPU.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {

static Value gpuAllocationFunction(OpBuilder &builder, Location loc,
                                   ArrayRef<int64_t> staticShape,
                                   Type elementType,
                                   ArrayRef<Value> dynamicSizes) {
  MemRefType allocType = MemRefType::get(staticShape, elementType, {}, 3);
  return builder.create<memref::AllocOp>(loc, allocType, dynamicSizes);
}

void addGPUVectorizationPassPipeline(OpPassManager &pm) {
  //===--------------------------------------------------------------------===//
  // Initial clean up.
  //===--------------------------------------------------------------------===//
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Distribute linalg onto threads within the workgroup.
  pm.addNestedPass<FuncOp>(createLLVMGPUTileAndDistribute());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addNestedPass<FuncOp>(createRemoveSingleIterationLoopPass());

  // Linalg -> vector
  pm.addNestedPass<FuncOp>(createLLVMGPUVectorizationPass());
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(createCSEPass());
  pm.addNestedPass<FuncOp>(createOptimizeVectorTransferPass());
}

void addGPUMatmulSimtPassPipeline(OpPassManager &pm) {
  //===--------------------------------------------------------------------===//
  // Initial clean up.
  //===--------------------------------------------------------------------===//
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Distribute linalg onto threads within the workgroup.
  pm.addNestedPass<FuncOp>(createLLVMGPUTileAndDistribute());
  pm.addNestedPass<FuncOp>(createLLVMGPUDistributeSharedMemoryCopy());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addNestedPass<FuncOp>(createRemoveSingleIterationLoopPass());

  // Linalg -> vector
  pm.addNestedPass<FuncOp>(createLLVMGPUVectorizationPass());
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(createCSEPass());
  pm.addNestedPass<FuncOp>(createOptimizeVectorTransferPass());

  // Pipeline memory operations.
  pm.addNestedPass<FuncOp>(createLLVMGPUPipeliningPass());
}

void addGPUMatmulTensorCorePassPipeline(OpPassManager &pm) {
  //===--------------------------------------------------------------------===//
  // Initial clean up.
  //===--------------------------------------------------------------------===//
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Distribute linalg onto warps within the workgroup.
  pm.addNestedPass<FuncOp>(
      createLLVMGPUTileAndDistribute(/*distributeToWarp=*/true));
  pm.addNestedPass<FuncOp>(createLLVMGPUDistributeSharedMemoryCopy());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addNestedPass<FuncOp>(createRemoveSingleIterationLoopPass());

  // Linalg -> vector
  pm.addNestedPass<FuncOp>(createLLVMGPUTensorCoreVectorizationPass());
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(createCSEPass());
  pm.addNestedPass<FuncOp>(createOptimizeVectorTransferPass());

  // Vector -> MMA ops
  pm.addNestedPass<FuncOp>(memref::createFoldSubViewOpsPass());
  pm.addNestedPass<FuncOp>(createConvertVectorToGPUPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Pipeline memory operations.
  pm.addNestedPass<FuncOp>(createLLVMGPUPipeliningPass());
}

void addGPUSimpleDistributePassPipeline(OpPassManager &pm) {
  //===--------------------------------------------------------------------===//
  // Initial clean up.
  //===--------------------------------------------------------------------===//
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Distribute linalg onto threads within the workgroup.
  pm.addNestedPass<FuncOp>(createLLVMGPUTileAndDistribute());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addNestedPass<FuncOp>(createRemoveSingleIterationLoopPass());
}

static void addLowerToLLVMGPUPasses(OpPassManager &pm, bool useROCM) {
  pm.addPass(createLowerAffinePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // LinalgExt -> SCF
  pm.addNestedPass<FuncOp>(IREE::LinalgExt::createLinalgExtToLoopsPass());

  // Linalg -> SCF
  pm.addNestedPass<FuncOp>(createConvertLinalgToLoopsPass());
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(createCSEPass());

  // Handled tensor-type constants.
  pm.addPass(createTensorConstantBufferizePass());
  pm.addPass(createFoldTensorExtractOpPass());

  pm.addNestedPass<FuncOp>(createLLVMGPUVectorLoweringPass());

  // SCF -> STD
  pm.addNestedPass<FuncOp>(createLowerToCFGPass());
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(createCSEPass());

  pm.addNestedPass<FuncOp>(createStdExpandOpsPass());
  pm.addPass(createLowerAffinePass());

  // Strip out the debug info for the kernel as CUDA driver doesn't diggest PTX
  // debug info well.
  pm.addPass(createStripDebugInfoPass());
  if (useROCM) {
    // convert to ROCDL.
    pm.addPass(createConvertToROCDLPass());
  } else {
    // convert to NVVM.
    pm.addPass(createConvertToNVVMPass());
  }
}

void buildLLVMGPUTransformPassPipeline(OpPassManager &pm, bool useROCM) {
  OpPassManager &bufferizePassPM = pm.nest<ModuleOp>();
  addLinalgBufferizePasses(bufferizePassPM, gpuAllocationFunction);
  pm.addPass(createLLVMGPULowerExecutableTargetPass());
  OpPassManager &nestedModulePM = pm.nest<ModuleOp>();
  //===--------------------------------------------------------------------===//
  // Convert Linalg ops to LLVM+NVVM/ROCDL ops.
  //
  // Post-conditions:
  //   - All Linalg/Loops/GPU/Affine/Standard ops are converted away.
  //   - The module contains the final llvm.module ready to be serialized.
  //===--------------------------------------------------------------------===//
  addLowerToLLVMGPUPasses(nestedModulePM, useROCM);
}

}  // namespace iree_compiler
}  // namespace mlir

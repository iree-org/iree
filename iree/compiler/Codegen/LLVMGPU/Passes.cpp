// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Passes.h"

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"
#include "iree/compiler/Dialect/Shape/Transforms/Passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Dialect/Linalg/Passes.h"
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
  // Convert tensor to buffers.
  addLinalgBufferizePasses(pm, gpuAllocationFunction);
  //===--------------------------------------------------------------------===//
  // Initial clean up.
  //===--------------------------------------------------------------------===//
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Distribute linalg onto threads within the workgroup.
  pm.addNestedPass<FuncOp>(createLLVMGPUTileAndDistributeToThreads());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addNestedPass<FuncOp>(createLLVMGPURemoveSingleIterationLoopPass());

  // Linalg -> vector
  pm.addNestedPass<FuncOp>(createLLVMGPUVectorizationPass());
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(createCSEPass());
  pm.addNestedPass<FuncOp>(createOptimizeVectorTransferPass());
}

void addGPUMatmulSimtPassPipeline(OpPassManager &pm) {
  // Convert tensor to buffers.
  addLinalgBufferizePasses(pm, gpuAllocationFunction);
  //===--------------------------------------------------------------------===//
  // Initial clean up.
  //===--------------------------------------------------------------------===//
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Distribute linalg onto threads within the workgroup.
  pm.addNestedPass<FuncOp>(createLLVMGPUTileAndDistributeToThreads());
  pm.addNestedPass<FuncOp>(createLLVMGPUDistributeSharedMemoryCopy());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addNestedPass<FuncOp>(createLLVMGPURemoveSingleIterationLoopPass());

  // Linalg -> vector
  pm.addNestedPass<FuncOp>(createLLVMGPUVectorizationPass());
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(createCSEPass());
  pm.addNestedPass<FuncOp>(createOptimizeVectorTransferPass());

  // Pipeline memory operations.
  pm.addNestedPass<FuncOp>(createLLVMGPUPipeliningPass());
}

void addGPUSimpleDistributePassPipeline(OpPassManager &pm) {
  // Convert tensor to buffers.
  addLinalgBufferizePasses(pm, gpuAllocationFunction);

  //===--------------------------------------------------------------------===//
  // Initial clean up.
  //===--------------------------------------------------------------------===//
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Distribute linalg onto threads within the workgroup.
  pm.addNestedPass<FuncOp>(createLLVMGPUTileAndDistributeToThreads());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addNestedPass<FuncOp>(createLLVMGPURemoveSingleIterationLoopPass());
}

static void addLowerToLLVMGPUPasses(OpPassManager &pm, bool useROCM) {
  pm.addPass(createLowerAffinePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // LinalgExt -> SCF
  pm.addNestedPass<FuncOp>(linalg_ext::createLinalgExtToLoopsPass());

  // Linalg -> SCF
  pm.addNestedPass<FuncOp>(createConvertLinalgToLoopsPass());
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(createCSEPass());

  // Handled tensor-type constants.
  pm.addPass(createTensorConstantBufferizePass());
  pm.addPass(createFoldTensorExtractOpPass());

  // SCF -> STD
  pm.addNestedPass<FuncOp>(createLowerToCFGPass());
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(createCSEPass());

  pm.addNestedPass<FuncOp>(createLLVMGPUVectorLoweringPass());
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

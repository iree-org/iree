// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Passes.h"

#include "iree-dialects/Dialect/LinalgExt/Passes/Passes.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToGPU/VectorToGPU.h"
#include "mlir/Dialect/Arithmetic/Transforms/Passes.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {

// TODO(thomasraoux): Add a new optional attribute to translate info.
static llvm::cl::opt<unsigned> pipelineDepth("iree-codegen-cuda-pipeline-depth",
                                             llvm::cl::desc("Pipeline depth"),
                                             llvm::cl::init(4));

static Value gpuAllocationFunction(OpBuilder &builder, Location loc,
                                   ArrayRef<int64_t> staticShape,
                                   Type elementType,
                                   ArrayRef<Value> dynamicSizes) {
  MemRefType allocType = MemRefType::get(staticShape, elementType, {}, 3);
  return builder.create<memref::AllocOp>(loc, allocType, dynamicSizes);
}

static void tileAndBufferize(OpPassManager &pm) {
  pm.addNestedPass<FuncOp>(createTileAndDistributeToWorkgroupsPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  addLinalgBufferizePasses(pm, gpuAllocationFunction);

  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
}

//===---------------------------------------------------------------------===//
// Codegen pipelines.
//===---------------------------------------------------------------------===//

void addGPUVectorizationPassPipeline(OpPassManager &pm) {
  tileAndBufferize(pm);

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
  pm.addNestedPass<FuncOp>(createTileAndDistributeToWorkgroupsPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  addLinalgBufferizePasses(pm, gpuAllocationFunction);

  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Distribute linalg onto threads within the workgroup.
  pm.addNestedPass<FuncOp>(createLLVMGPUTileAndDistribute());
  pm.addNestedPass<FuncOp>(createMemrefCopyToLinalgPass());
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
  tileAndBufferize(pm);

  // Distribute linalg onto warps within the workgroup.
  pm.addNestedPass<FuncOp>(
      createLLVMGPUTileAndDistribute(/*distributeToWarp=*/true));
  if (pipelineDepth > 1)
    pm.addNestedPass<FuncOp>(createLLVMGPUMultiBuffering(pipelineDepth));
  pm.addNestedPass<FuncOp>(createMemrefCopyToLinalgPass());
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
  pm.addNestedPass<FuncOp>(createLLVMGPUVectorToGPU());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Pipeline memory operations.
  pm.addNestedPass<FuncOp>(createLLVMGPUPipeliningPass(pipelineDepth));
}

void addGPUSimpleDistributePassPipeline(OpPassManager &pm) {
  tileAndBufferize(pm);

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
  pm.addNestedPass<FuncOp>(createMemrefCopyToLinalgPass());
  pm.addNestedPass<FuncOp>(createConvertLinalgToLoopsPass());
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(createCSEPass());

  // Handled tensor-type constants.
  pm.addPass(arith::createConstantBufferizePass());
  pm.addPass(createFoldTensorExtractOpPass());

  pm.addNestedPass<FuncOp>(createLLVMGPUVectorLoweringPass());

  // SCF -> STD
  pm.addNestedPass<FuncOp>(createConvertSCFToCFPass());
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(createCSEPass());

  // math dialect elementry functions -> polynomial form.
  pm.addNestedPass<FuncOp>(createPolynomialApproximationPass());

  pm.addNestedPass<FuncOp>(arith::createArithmeticExpandOpsPass());
  pm.addNestedPass<FuncOp>(memref::createExpandOpsPass());
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
  pm.nest<ModuleOp>().nest<FuncOp>().addPass(createTypePropagationPass());
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

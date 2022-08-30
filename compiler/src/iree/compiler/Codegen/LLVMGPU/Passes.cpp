// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Passes.h"

#include "iree-dialects/Dialect/LinalgExt/Passes/Passes.h"
#include "iree-dialects/Dialect/LinalgTransform/Passes.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToGPU/VectorToGPU.h"
#include "mlir/Dialect/Arithmetic/Transforms/Passes.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {

static llvm::cl::opt<unsigned> logSwizzleTile(
    "iree-codegen-log-swizzle-tile", llvm::cl::desc("log swizzle tile value"),
    llvm::cl::init(0));

/// Flag used for the transition from wmma to mma.sync. Once we have better
/// performance with mma.sync we can drop wmma support and remove this flag.
llvm::cl::opt<bool> llvmgpuUseMMASync(
    "iree-codegen-llvmgpu-use-mma-sync",
    llvm::cl::desc("use mma sync instead of wmma ops"), llvm::cl::init(false));

static FailureOr<Value> gpuAllocationFn(OpBuilder &builder, Location loc,
                                        MemRefType memRefType,
                                        ValueRange dynamicSizes,
                                        unsigned alignment) {
  MemRefType allocType = MemRefType::get(memRefType.getShape(),
                                         memRefType.getElementType(), {}, 3);
  return builder.create<memref::AllocOp>(loc, allocType, dynamicSizes)
      .getResult();
}

static LogicalResult gpuDeallocationFn(OpBuilder &builder, Location loc,
                                       Value allocation) {
  return success();
}

static LogicalResult gpuCopyFn(OpBuilder &builder, Location loc, Value from,
                               Value to) {
  bool sharedMemCopy =
      from.getType().cast<MemRefType>().getMemorySpaceAsInt() == 3 ||
      to.getType().cast<MemRefType>().getMemorySpaceAsInt() == 3;
  if (sharedMemCopy) builder.create<gpu::BarrierOp>(loc);
  createLinalgCopyOp(builder, loc, from, to);
  if (sharedMemCopy) builder.create<gpu::BarrierOp>(loc);
  return success();
}

static void addBufferizePasses(OpPassManager &passManager) {
  BufferizationOptions::AllocationFn allocationFn = gpuAllocationFn;
  BufferizationOptions::DeallocationFn deallocationFn = gpuDeallocationFn;
  BufferizationOptions::MemCpyFn memcpyFn = gpuCopyFn;
  addIREEComprehensiveBufferizePasses(passManager, allocationFn, deallocationFn,
                                      memcpyFn);
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());
}

static void tileAndDistributeToWorkgroup(OpPassManager &pm) {
  pm.addPass(createTileAndDistributeToWorkgroupsPass());

  auto &nestedModulePM = pm.nest<ModuleOp>();
  nestedModulePM.addNestedPass<func::FuncOp>(
      createConvertToDestinationPassingStylePass());
  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addPass(createCSEPass());
}

static void tileAndBufferize(OpPassManager &pm) {
  tileAndDistributeToWorkgroup(pm);

  auto &nestedModulePM = pm.nest<ModuleOp>();
  addBufferizePasses(nestedModulePM);
}

//===---------------------------------------------------------------------===//
// Codegen pipelines.
//===---------------------------------------------------------------------===//

void addGPUVectorizationPassPipeline(OpPassManager &pm) {
  tileAndDistributeToWorkgroup(pm);

  auto &nestedModulePM = pm.nest<ModuleOp>();
  nestedModulePM.addNestedPass<func::FuncOp>(
      createRemoveSingleIterationLoopPass());
  // Distribute linalg onto threads within the workgroup.
  nestedModulePM.addNestedPass<func::FuncOp>(createLLVMGPUTileTensor(false));
  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addPass(createCSEPass());

  nestedModulePM.addNestedPass<func::FuncOp>(
      createRemoveSingleIterationLoopPass());

  // Linalg -> vector
  nestedModulePM.addNestedPass<func::FuncOp>(createLLVMGPUVectorizationPass());
  nestedModulePM.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  nestedModulePM.addNestedPass<func::FuncOp>(createCSEPass());
  nestedModulePM.addNestedPass<func::FuncOp>(
      createOptimizeVectorTransferPass());
  addBufferizePasses(nestedModulePM);
  nestedModulePM.addNestedPass<func::FuncOp>(createLLVMGPUDistribute());
}

void addGPUMatmulSimtPassPipeline(OpPassManager &pm) {
  tileAndBufferize(pm);

  auto &nestedModulePM = pm.nest<ModuleOp>();
  // Distribute linalg onto threads within the workgroup.
  nestedModulePM.addNestedPass<func::FuncOp>(createLLVMGPUTileAndDistribute());
  nestedModulePM.addNestedPass<func::FuncOp>(createMemrefCopyToLinalgPass());
  nestedModulePM.addNestedPass<func::FuncOp>(
      createGPUDistributeSharedMemoryCopy());
  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addPass(createCSEPass());

  nestedModulePM.addNestedPass<func::FuncOp>(
      createLLVMGPUReduceSharedMemoryBankConflicts());
  nestedModulePM.addNestedPass<func::FuncOp>(
      createRemoveSingleIterationLoopPass());
  nestedModulePM.addNestedPass<func::FuncOp>(
      createWorkGroupSwizzle(logSwizzleTile));
  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addPass(createCSEPass());

  // Linalg -> vector
  nestedModulePM.addNestedPass<func::FuncOp>(createLLVMGPUVectorizationPass());
  nestedModulePM.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  nestedModulePM.addNestedPass<func::FuncOp>(createCSEPass());
  nestedModulePM.addNestedPass<func::FuncOp>(
      createOptimizeVectorTransferPass());

  // Pipeline memory operations.
  nestedModulePM.addNestedPass<func::FuncOp>(createGPUPipeliningPass());
}

void addGPUMatmulTensorCorePassPipeline(OpPassManager &pm,
                                        unsigned pipelineDepth) {
  tileAndBufferize(pm);

  auto &nestedModulePM = pm.nest<ModuleOp>();
  // Distribute linalg onto warps within the workgroup.
  nestedModulePM.addNestedPass<func::FuncOp>(
      createLLVMGPUTileAndDistribute(/*distributeToWarp=*/true));
  nestedModulePM.addNestedPass<func::FuncOp>(
      createRemoveSingleIterationLoopPass());
  if (pipelineDepth > 1)
    nestedModulePM.addNestedPass<func::FuncOp>(
        createLLVMGPUMultiBuffering(pipelineDepth));
  nestedModulePM.addNestedPass<func::FuncOp>(createMemrefCopyToLinalgPass());
  nestedModulePM.addNestedPass<func::FuncOp>(
      createGPUDistributeSharedMemoryCopy());
  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addPass(createCSEPass());

  nestedModulePM.addNestedPass<func::FuncOp>(
      createLLVMGPUReduceSharedMemoryBankConflicts());
  nestedModulePM.addNestedPass<func::FuncOp>(
      createRemoveSingleIterationLoopPass());
  nestedModulePM.addNestedPass<func::FuncOp>(
      createWorkGroupSwizzle(logSwizzleTile));
  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addPass(createCSEPass());

  // Linalg -> vector
  nestedModulePM.addNestedPass<func::FuncOp>(
      createLLVMGPUTensorCoreVectorizationPass());
  nestedModulePM.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  nestedModulePM.addNestedPass<func::FuncOp>(createCSEPass());
  nestedModulePM.addNestedPass<func::FuncOp>(
      createOptimizeVectorTransferPass());

  // Vector -> MMA ops
  nestedModulePM.addNestedPass<func::FuncOp>(
      memref::createFoldSubViewOpsPass());
  nestedModulePM.addNestedPass<func::FuncOp>(createLLVMGPUVectorToGPU());
  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addPass(createCSEPass());

  // Pipeline memory operations.
  nestedModulePM.addNestedPass<func::FuncOp>(
      createGPUPipeliningPass(pipelineDepth));
}

void addGPUWarpReductionPassPipeline(OpPassManager &pm) {
  tileAndDistributeToWorkgroup(pm);
  auto &nestedModulePM = pm.nest<ModuleOp>();

  nestedModulePM.addNestedPass<func::FuncOp>(
      createRemoveSingleIterationLoopPass());

  // Linalg -> vector
  nestedModulePM.addNestedPass<func::FuncOp>(createLLVMGPUVectorizationPass(
      /*nativeVector=*/0, /*generateContract=*/false));
  nestedModulePM.addNestedPass<func::FuncOp>(
      createLoopInvariantCodeMotionPass());
  nestedModulePM.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  nestedModulePM.addNestedPass<func::FuncOp>(createCSEPass());

  addBufferizePasses(nestedModulePM);

  nestedModulePM.addNestedPass<func::FuncOp>(
      createOptimizeVectorTransferPass());

  nestedModulePM.addNestedPass<func::FuncOp>(
      memref::createFoldSubViewOpsPass());
  nestedModulePM.addNestedPass<func::FuncOp>(
      createLoopInvariantCodeMotionPass());
  nestedModulePM.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  nestedModulePM.addNestedPass<func::FuncOp>(createCSEPass());

  // vector -> simt gpu + vector
  nestedModulePM.addNestedPass<func::FuncOp>(
      createConvertVectorReductionToGPUPass());
  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addPass(createCSEPass());
}

void addGPUSimpleDistributePassPipeline(OpPassManager &pm) {
  tileAndBufferize(pm);

  auto &nestedModulePM = pm.nest<ModuleOp>();
  // Distribute linalg onto threads within the workgroup.
  nestedModulePM.addNestedPass<func::FuncOp>(createLLVMGPUTileAndDistribute());
  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addPass(createCSEPass());

  nestedModulePM.addNestedPass<func::FuncOp>(
      createRemoveSingleIterationLoopPass());
}

static void addLowerToLLVMGPUPasses(OpPassManager &pm, bool useROCM) {
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // LinalgExt -> SCF
  pm.addNestedPass<func::FuncOp>(IREE::LinalgExt::createLinalgExtToLoopsPass());

  // Linalg -> SCF
  pm.addNestedPass<func::FuncOp>(createMemrefCopyToLinalgPass());
  pm.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createCSEPass());

  // Pad allocations with dynamic dimension before lowering of SCF and affine
  // but after linalg lowering.
  pm.addNestedPass<func::FuncOp>(createPadDynamicAlloc());

  pm.addPass(createLowerAffinePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Handled tensor-type constants.
  pm.addPass(arith::createConstantBufferizePass());
  pm.addPass(createFoldTensorExtractOpPass());

  pm.addNestedPass<func::FuncOp>(createLLVMGPUVectorLoweringPass());

  // SCF -> STD
  pm.addNestedPass<func::FuncOp>(createConvertSCFToCFPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createCSEPass());

  // math dialect elementry functions -> polynomial form.
  pm.addNestedPass<func::FuncOp>(createPolynomialApproximationPass());

  pm.addNestedPass<func::FuncOp>(arith::createArithmeticExpandOpsPass());
  pm.addNestedPass<func::FuncOp>(memref::createExpandOpsPass());
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

extern llvm::cl::opt<std::string> clGPUCodegenTransformDialectFileName;

void addGPUTransformDialectInterpreterPasses(OpPassManager &passManager) {
  // Give control to the transform dialect.
  passManager.addPass(
      mlir::iree_compiler::createTransformDialectInterpreterPass(
          clGPUCodegenTransformDialectFileName));

  // Dropping the schedule is only needed if we want to embed the transform in
  // the module: we should drop the schedule once applied.
  // This pass does nothing in the case where we apply a separate policy
  // through a file.
  passManager.addPass(createDropSchedulePass());
}

void buildLLVMGPUTransformPassPipeline(OpPassManager &pm, bool useROCM) {
  pm.nest<ModuleOp>().nest<func::FuncOp>().addPass(createTypePropagationPass());
  pm.nest<ModuleOp>().addPass(createBufferizeCopyOnlyDispatchesPass());
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

// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/Passes/Passes.h"

#include "iree-dialects/Dialect/LinalgTransform/Passes.h"
#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/LLVMGPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ComplexToStandard/ComplexToStandard.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToGPU/VectorToGPU.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "iree-llvm-gpu-lowering-pass-pipeline"

namespace mlir {
namespace iree_compiler {

static llvm::cl::opt<unsigned>
    logSwizzleTile("iree-codegen-log-swizzle-tile",
                   llvm::cl::desc("log swizzle tile value"), llvm::cl::init(0));

static FailureOr<Value> gpuAllocationFn(OpBuilder &builder, Location loc,
                                        MemRefType memRefType,
                                        ValueRange dynamicSizes,
                                        unsigned alignment) {
  auto workgroupSpace = gpu::AddressSpaceAttr::get(
      builder.getContext(), gpu::GPUDialect::getWorkgroupAddressSpace());
  MemRefType allocType =
      MemRefType::get(memRefType.getShape(), memRefType.getElementType(),
                      AffineMap(), workgroupSpace);
  return builder.create<memref::AllocOp>(loc, allocType, dynamicSizes)
      .getResult();
}

static LogicalResult gpuCopyFn(OpBuilder &builder, Location loc, Value from,
                               Value to) {
  bool needsBarrier = false;
  if (hasSharedMemoryAddressSpace(llvm::cast<MemRefType>(from.getType()))) {
    needsBarrier = true;
  }
  if (hasSharedMemoryAddressSpace(llvm::cast<MemRefType>(to.getType()))) {
    needsBarrier = true;
  }
  if (needsBarrier)
    builder.create<gpu::BarrierOp>(loc);
  Operation *copy = builder.create<memref::CopyOp>(loc, from, to);
  if (needsBarrier) {
    setMarker(copy, getCopyToWorkgroupMemoryMarker());
    builder.create<gpu::BarrierOp>(loc);
  }
  return success();
}

static void addBufferizePasses(OpPassManager &passManager) {
  BufferizationOptions::AllocationFn allocationFn = gpuAllocationFn;
  BufferizationOptions::MemCpyFn memcpyFn = gpuCopyFn;
  addIREEComprehensiveBufferizePasses(passManager, allocationFn, memcpyFn);
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());
}

static void
tileAndDistributeToWorkgroup(OpPassManager &pm,
                             bool useWARForCooperativeMatrixCodegen = false) {
  pm.addPass(createTileAndDistributeToWorkgroupsPass(
      /*maxWorkgroupParallelDims=*/1,
      linalg::DistributionMethod::CyclicNumProcsEqNumIters));

  auto &nestedModulePM = pm.nest<ModuleOp>();
  nestedModulePM.addNestedPass<func::FuncOp>(
      createConvertToDestinationPassingStylePass(
          useWARForCooperativeMatrixCodegen));
  nestedModulePM.addNestedPass<func::FuncOp>(
      IREE::LinalgExt::createTileAndDecomposeAttentionPass());
  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addPass(createCSEPass());
}

static void tileAndBufferize(OpPassManager &pm) {
  tileAndDistributeToWorkgroup(pm, /*useWARForCooperativeMatrixCodegen=*/true);

  auto &nestedModulePM = pm.nest<ModuleOp>();
  addBufferizePasses(nestedModulePM);
}

static void addGPUVectorizationPasses(OpPassManager &pm) {
  pm.addNestedPass<func::FuncOp>(createDecomposeConvolutionToLowerDimOpsPass());
  GenericVectorizationPassOptions options;
  options.vectorizePadding = true;
  options.vectorizeGatherAccesses = true;
  options.enableCleanup = false;
  options.foldCastIntoContract = true;
  options.maxVectorSize = 4096;
  pm.addNestedPass<func::FuncOp>(createGenericVectorizationPass(options));
  pm.addNestedPass<func::FuncOp>(createHoistRedundantVectorTransfersPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createCSEPass());
}

//===---------------------------------------------------------------------===//
// Codegen pipelines.
//===---------------------------------------------------------------------===//

void addGPUVectorizationPassPipeline(OpPassManager &pm) {
  tileAndDistributeToWorkgroup(pm);

  auto &nestedModulePM = pm.nest<ModuleOp>();

  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addNestedPass<func::FuncOp>(
      createWorkgroupSpecializationPass());
  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addPass(createCSEPass());

  // Distribute linalg onto threads within the workgroup.
  nestedModulePM.addNestedPass<func::FuncOp>(createGPUTensorTile(false));
  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addPass(createCSEPass());

  // Linalg -> vector
  addGPUVectorizationPasses(nestedModulePM);

  // tensor to memref
  addBufferizePasses(nestedModulePM);
  nestedModulePM.addNestedPass<func::FuncOp>(createGPUDistribute());

  // Post bufferization optimizations.
  nestedModulePM.addNestedPass<func::FuncOp>(
      createLoopInvariantCodeMotionPass());
  nestedModulePM.addNestedPass<func::FuncOp>(
      memref::createFoldMemRefAliasOpsPass());
  nestedModulePM.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  nestedModulePM.addNestedPass<func::FuncOp>(createCSEPass());
  nestedModulePM.addNestedPass<func::FuncOp>(
      createOptimizeVectorTransferPass());
  nestedModulePM.addNestedPass<func::FuncOp>(
      createHoistRedundantVectorTransfersPass());
}

void addGPUMatmulSimtPassPipeline(OpPassManager &pm) {
  tileAndDistributeToWorkgroup(pm);
  auto &nestedModulePM = pm.nest<ModuleOp>();

  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addNestedPass<func::FuncOp>(
      createWorkgroupSpecializationPass());
  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addPass(createCSEPass());

  nestedModulePM.addNestedPass<func::FuncOp>(createGPUTensorAlloc());
  nestedModulePM.addNestedPass<func::FuncOp>(createGPUTensorTile(false));

  // Linalg -> vector
  addGPUVectorizationPasses(nestedModulePM);

  // tensor to memref
  addBufferizePasses(nestedModulePM);

  // distribute foreach threads
  nestedModulePM.addNestedPass<func::FuncOp>(createGPUDistribute());

  nestedModulePM.addNestedPass<func::FuncOp>(createMemrefCopyToLinalgPass());
  nestedModulePM.addNestedPass<func::FuncOp>(
      createGPUDistributeSharedMemoryCopy());
  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addPass(createCSEPass());

  nestedModulePM.addNestedPass<func::FuncOp>(
      createGPUReduceSharedMemoryBankConflicts());
  nestedModulePM.addNestedPass<func::FuncOp>(
      createWorkGroupSwizzle(logSwizzleTile));
  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addPass(createCSEPass());

  nestedModulePM.addNestedPass<func::FuncOp>(
      memref::createFoldMemRefAliasOpsPass());
  nestedModulePM.addPass(createCSEPass());
  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addPass(createCSEPass());

  // Even though we vectorize before bufferization we are not able to hoist
  // accumulator load/store out of the K loop until distribution. Therefore we
  // still rely on buffer level transformations for transfer ops hoisting and
  // store to load forwarding. This relies on shacky alias analysis and we need
  // to move this to tensor level once we have better abstractions.
  nestedModulePM.addNestedPass<func::FuncOp>(
      createHoistRedundantVectorTransfersPass());

  // Hoist loop invariant code to avoid pipelining it.
  nestedModulePM.addNestedPass<func::FuncOp>(
      createLoopInvariantCodeMotionPass());
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
        createGPUMultiBuffering(pipelineDepth));
  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addPass(createCSEPass());

  nestedModulePM.addNestedPass<func::FuncOp>(
      createRemoveSingleIterationLoopPass());
  nestedModulePM.addNestedPass<func::FuncOp>(
      createWorkGroupSwizzle(logSwizzleTile));
  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addPass(createCSEPass());

  // Linalg -> vector
  nestedModulePM.addNestedPass<func::FuncOp>(
      createLLVMGPUTensorCoreVectorizationPass());
  nestedModulePM.addNestedPass<func::FuncOp>(
      memref::createFoldMemRefAliasOpsPass());
  nestedModulePM.addNestedPass<func::FuncOp>(createCSEPass());
  nestedModulePM.addNestedPass<func::FuncOp>(
      createOptimizeVectorTransferPass());
  nestedModulePM.addNestedPass<func::FuncOp>(
      createHoistRedundantVectorTransfersPass());

  // Distribute shared memory copies.
  nestedModulePM.addNestedPass<func::FuncOp>(createMemrefCopyToLinalgPass());
  nestedModulePM.addNestedPass<func::FuncOp>(
      createGPUDistributeSharedMemoryCopy());
  nestedModulePM.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  nestedModulePM.addNestedPass<func::FuncOp>(createCSEPass());
  nestedModulePM.addNestedPass<func::FuncOp>(
      createGPUReduceSharedMemoryBankConflicts());

  // Vector -> MMA ops
  nestedModulePM.addNestedPass<func::FuncOp>(
      memref::createFoldMemRefAliasOpsPass());
  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addPass(createCSEPass());
  nestedModulePM.addNestedPass<func::FuncOp>(createLLVMGPUVectorToGPU());
  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addPass(createCSEPass());

  // Hoist loop invariant code to avoid pipelining it.
  nestedModulePM.addNestedPass<func::FuncOp>(
      createLoopInvariantCodeMotionPass());
  // Pipeline memory operations.
  nestedModulePM.addNestedPass<func::FuncOp>(createGPUPipeliningPass(
      /*epiloguePeeling=*/false, pipelineDepth,
      PipeliningSchedulingStrategy::loadGlobalStage0));
  // Optimize shared memory usage.
  nestedModulePM.addNestedPass<func::FuncOp>(
      createLLVMGPUPackSharedMemoryAlloc());
}

void addGPUMatmulTensorCoreMmaSyncPassPipeline(OpPassManager &pm,
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
        createGPUMultiBuffering(pipelineDepth));
  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addPass(createCSEPass());

  nestedModulePM.addNestedPass<func::FuncOp>(
      createRemoveSingleIterationLoopPass());
  nestedModulePM.addNestedPass<func::FuncOp>(
      createWorkGroupSwizzle(logSwizzleTile));
  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addPass(createCSEPass());

  // Linalg -> vector
  nestedModulePM.addNestedPass<func::FuncOp>(
      createLLVMGPUTensorCoreVectorizationPass(GPUTensorCoreType::MMA_SYNC));
  nestedModulePM.addNestedPass<func::FuncOp>(
      memref::createFoldMemRefAliasOpsPass());
  nestedModulePM.addNestedPass<func::FuncOp>(createCSEPass());
  nestedModulePM.addNestedPass<func::FuncOp>(
      createOptimizeVectorTransferPass());
  nestedModulePM.addNestedPass<func::FuncOp>(
      createHoistRedundantVectorTransfersPass());

  // Distribute shared memory copies.
  nestedModulePM.addNestedPass<func::FuncOp>(createMemrefCopyToLinalgPass());
  nestedModulePM.addNestedPass<func::FuncOp>(
      createGPUDistributeSharedMemoryCopy());
  nestedModulePM.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  nestedModulePM.addNestedPass<func::FuncOp>(createCSEPass());

  // Vector -> MMA ops
  nestedModulePM.addNestedPass<func::FuncOp>(
      memref::createFoldMemRefAliasOpsPass());
  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addPass(createCSEPass());
  nestedModulePM.addNestedPass<func::FuncOp>(
      createLLVMGPUVectorToGPU(GPUTensorCoreType::MMA_SYNC));
  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addPass(createCSEPass());

  // Hoist loop invariant code to avoid pipelining it.
  nestedModulePM.addNestedPass<func::FuncOp>(
      createLoopInvariantCodeMotionPass());
  // Pipeline memory operations.
  nestedModulePM.addNestedPass<func::FuncOp>(createGPUPipeliningPass(
      /*epiloguePeeling=*/false, pipelineDepth,
      PipeliningSchedulingStrategy::nvidiaTensorCore));
  // Optimize shared memory usage.
  nestedModulePM.addNestedPass<func::FuncOp>(
      createLLVMGPUPackSharedMemoryAlloc());
}

void addGPUTransposePassPipeline(OpPassManager &pm) {
  tileAndDistributeToWorkgroup(pm);
  auto &nestedModulePM = pm.nest<ModuleOp>();

  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addNestedPass<func::FuncOp>(
      createWorkgroupSpecializationPass());
  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addPass(createCSEPass());

  nestedModulePM.addNestedPass<func::FuncOp>(
      createGPUTensorAlloc(GPUPromoteSharedMemPattern::TransposeOpPattern));
  nestedModulePM.addNestedPass<func::FuncOp>(createGPUTensorTile(false));

  // Linalg -> vector
  addGPUVectorizationPasses(nestedModulePM);
  nestedModulePM.addNestedPass<func::FuncOp>(
      createOptimizeVectorTransferPass());
  nestedModulePM.addNestedPass<func::FuncOp>(
      createHoistRedundantVectorTransfersPass());

  // tensor to memref
  addBufferizePasses(nestedModulePM);

  // distribute foreach threads
  nestedModulePM.addNestedPass<func::FuncOp>(createGPUDistribute());

  nestedModulePM.addNestedPass<func::FuncOp>(createMemrefCopyToLinalgPass());
  nestedModulePM.addNestedPass<func::FuncOp>(
      createGPUDistributeSharedMemoryCopy());
  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addPass(createCSEPass());

  // May or may not need to reduce shared mememory conflicts
  nestedModulePM.addNestedPass<func::FuncOp>(
      createGPUReduceSharedMemoryBankConflicts(/*paddingSizeBits=*/32));
  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addPass(createCSEPass());
}

void addGPUWarpReductionPassPipeline(OpPassManager &pm) {
  tileAndDistributeToWorkgroup(pm);
  auto &nestedModulePM = pm.nest<ModuleOp>();
  nestedModulePM.addNestedPass<func::FuncOp>(
      createRematerializeParallelOpsPass());
  nestedModulePM.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  nestedModulePM.addNestedPass<func::FuncOp>(createGPUTileReductionPass());
  nestedModulePM.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  nestedModulePM.addNestedPass<func::FuncOp>(createCSEPass());

  // Linalg -> vector
  {
    GenericVectorizationPassOptions options;
    options.vectorizePadding = true;
    options.vectorizeGatherAccesses = true;
    options.enableCleanup = false;
    options.generateContract = false;
    options.maxVectorSize = 16384;
    nestedModulePM.addNestedPass<func::FuncOp>(
        createGenericVectorizationPass(options));
    nestedModulePM.addNestedPass<func::FuncOp>(
        createHoistRedundantVectorTransfersPass());
    nestedModulePM.addNestedPass<func::FuncOp>(createCanonicalizerPass());
    nestedModulePM.addNestedPass<func::FuncOp>(createCSEPass());
  }
  nestedModulePM.addNestedPass<func::FuncOp>(
      createLoopInvariantCodeMotionPass());
  nestedModulePM.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  nestedModulePM.addNestedPass<func::FuncOp>(createCSEPass());

  addBufferizePasses(nestedModulePM);

  nestedModulePM.addNestedPass<func::FuncOp>(
      memref::createFoldMemRefAliasOpsPass());
  nestedModulePM.addNestedPass<func::FuncOp>(
      createOptimizeVectorTransferPass());
  nestedModulePM.addNestedPass<func::FuncOp>(
      createHoistRedundantVectorTransfersPass());
  nestedModulePM.addNestedPass<func::FuncOp>(
      createLoopInvariantCodeMotionPass());
  nestedModulePM.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  nestedModulePM.addNestedPass<func::FuncOp>(createCSEPass());
  nestedModulePM.addNestedPass<func::FuncOp>(createForOpCanonicalizationPass());
  nestedModulePM.addNestedPass<func::FuncOp>(createCanonicalizerPass());

  // vector -> simt gpu + vector
  nestedModulePM.addNestedPass<func::FuncOp>(
      createConvertVectorReductionToGPUPass());
  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addPass(createCSEPass());
}

void addGPUPackUnPackPasses(OpPassManager &pm) {
  tileAndDistributeToWorkgroup(pm);
  auto &nestedModulePM = pm.nest<ModuleOp>();

  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addNestedPass<func::FuncOp>(
      createWorkgroupSpecializationPass());
  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addPass(createCSEPass());

  nestedModulePM.addNestedPass<func::FuncOp>(createGPUTensorTile(false));
  nestedModulePM.addNestedPass<func::FuncOp>(
      createDecomposePackUnPackOpsPass(/*tileOuterToOne=*/true));
  addGPUVectorizationPasses(nestedModulePM);

  addBufferizePasses(nestedModulePM);

  // distribute foreach threads
  nestedModulePM.addNestedPass<func::FuncOp>(createGPUDistribute());

  nestedModulePM.addNestedPass<func::FuncOp>(
      createSplitFullPartialTransferPass("linalg-copy"));
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

void addGPUDefaultPassPipeline(OpPassManager &pm) {
  tileAndBufferize(pm);
  auto &nestedModulePM = pm.nest<ModuleOp>();
  nestedModulePM.addNestedPass<func::FuncOp>(
      createRemoveSingleIterationLoopPass());
}

// Sub pipeline to make the address computation more explicit and
// optimize them.
// The idea here is to be less dependent on what the backend is able to
// do by heavy lifting most of the work while we still have the
// information about loops.
// Note: This needs to run before SCF -> CF.
static void addLowerAndOptimzeAddressComputation(OpPassManager &pm) {
  pm.addPass(createExtractAddressComputationGPUPass());
  pm.addNestedPass<func::FuncOp>(memref::createExpandOpsPass());
  pm.addPass(memref::createFoldMemRefAliasOpsPass());
  pm.addPass(memref::createExpandStridedMetadataPass());
  // Hoist loop invariant variables to give decompose affine pass the right loop
  // dependencies.
  pm.addPass(createLoopInvariantCodeMotionPass());
  // Decompose the `affine.apply`s.
  pm.addPass(createDecomposeAffineOpsPass());
  // Get rid of the redundant computations.
  pm.addPass(createCSEPass());
  // Hoist the resulting decompositions.
  pm.addPass(createLoopInvariantCodeMotionPass());
  pm.addPass(createLowerAffinePass());
}

static void addLowerToLLVMGPUPasses(OpPassManager &pm, bool useROCM) {
  pm.addPass(createConvertHALDescriptorTypeToGPUAddressSpacePass());

  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addPass(createLowerUKernelOpsToCallsPass());

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

  // THIS NEEDS TO RUN BEFORE SCF ->CF ON
  addLowerAndOptimzeAddressComputation(pm);
  // THIS NEEDS TO RUN BEFORE SCF ->CF OFF

  // Run checks on shared memory usage.
  // TODO: query this from the target.
  auto getSharedMemoryLimit = [](func::FuncOp) { return 163 * 1024; };
  auto getIndexBitwidth = [](func::FuncOp) { return 64; };
  pm.addPass(
      createGPUCheckResourceUsagePass(getSharedMemoryLimit, getIndexBitwidth));

  // SCF -> STD
  pm.addNestedPass<func::FuncOp>(createConvertSCFToCFPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createCSEPass());

  // Handle complex operation conversion.
  pm.addPass(createConvertComplexToStandardPass());

  // Convert BF16 operations to occur as F32.
  pm.addPass(createConvertBf16ArithToF32Pass());
  pm.addPass(createConvertBf16ToUInt16BuffersPass());

  pm.addNestedPass<func::FuncOp>(arith::createArithExpandOpsPass());

  // math dialect elementry functions -> polynomial form.
  pm.addNestedPass<func::FuncOp>(createPolynomialApproximationPass());

  pm.addNestedPass<func::FuncOp>(memref::createExpandOpsPass());
  pm.addPass(memref::createFoldMemRefAliasOpsPass());
  pm.addPass(memref::createExpandStridedMetadataPass());
  pm.addPass(createEmulateNarrowTypePass());
  pm.addPass(createLowerAffinePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  // Strip out the debug info for the kernel as CUDA driver doesn't diggest PTX
  // debug info well.
  pm.addPass(createStripDebugInfoPass());
  // Cast address spaces of all function arguments to generic
  pm.addPass(createLLVMGPUCastAddressSpaceFunction());
  if (useROCM) {
    // convert to ROCDL.
    pm.addPass(createConvertToROCDLPass());
  } else {
    // convert to NVVM.
    pm.addPass(createConvertToNVVMPass());
  }
}

extern llvm::cl::opt<std::string> clGPUCodegenTransformDialectDebugPayloadTag;
extern llvm::cl::opt<std::string> clGPUCodegenTransformDialectDebugTransformTag;

void addGPUTransformDialectPasses(OpPassManager &passManager) {
  passManager.addPass(
      mlir::iree_compiler::createTransformDialectInterpreterPass());

  // Dropping the schedule is needed:
  //   1. if we want to embed the transform in the module: we should drop the
  //      schedule once applied.
  //   2. if transform.do_not_dce_operands ops are introduced.
  passManager.addPass(createDropSchedulePass());
}

void buildLLVMGPUCodegenConfigurationPassPipeline(OpPassManager &pm) {
  addCommonTargetExecutablePreprocessingPasses(pm);
  pm.addPass(createLLVMGPUSelectLoweringStrategyPass());
}

void buildLLVMGPUCodegenPassPipeline(OpPassManager &pm, bool useROCM) {
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

  LLVM_DEBUG({
    llvm::dbgs() << "Using LLVMGPU pass pipeline:\n";
    pm.printAsTextualPipeline(llvm::dbgs());
    llvm::dbgs() << "\n";
  });
}

//===---------------------------------------------------------------------===//
// Register LLVMGPU Passes
//===---------------------------------------------------------------------===//

namespace {
#define GEN_PASS_REGISTRATION
#include "iree/compiler/Codegen/LLVMGPU/Passes.h.inc"
} // namespace

void registerCodegenLLVMGPUPasses() {
  // Generated.
  registerPasses();

  static PassPipelineRegistration<> LLVMGPUConfigPipeline(
      "iree-codegen-llvmgpu-configuration-pipeline",
      "Runs the translation strategy configuration pipeline on Linalg for GPU",
      [](OpPassManager &passManager) {
        buildLLVMGPUCodegenConfigurationPassPipeline(passManager);
      });

  static PassPipelineRegistration<> LinalgNVVMPipeline(
      "iree-codegen-linalg-to-nvvm-pipeline",
      "Runs the progressive lowering pipeline from Linalg to NVVM",
      [](OpPassManager &passManager) {
        buildLLVMGPUCodegenPassPipeline(passManager, false);
      });

  static PassPipelineRegistration<> LinalgROCDLPipeline(
      "iree-codegen-linalg-to-rocdl-pipeline",
      "Runs the progressive lowering pipeline from Linalg to ROCDL",
      [](OpPassManager &passManager) {
        buildLLVMGPUCodegenPassPipeline(passManager, true);
      });
}

} // namespace iree_compiler
} // namespace mlir

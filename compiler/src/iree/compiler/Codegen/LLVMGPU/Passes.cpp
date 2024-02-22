// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"

#include <cstdint>

#include "iree-dialects/Dialect/LinalgTransform/Passes.h"
#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/LLVMGPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/LLVMGPU/ROCDLPasses.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ComplexToStandard/ComplexToStandard.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToGPU/VectorToGPU.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "iree-llvm-gpu-lowering-pass-pipeline"

namespace mlir::iree_compiler {

constexpr int64_t kDefaultSubgroupSize = 32;

static llvm::cl::opt<unsigned>
    logSwizzleTile("iree-codegen-log-swizzle-tile",
                   llvm::cl::desc("log swizzle tile value"), llvm::cl::init(0));

llvm::cl::opt<int64_t> clLLVMGPUSharedMemoryLimit(
    "iree-llvmgpu-shared-memory-limit",
    llvm::cl::desc("specify the maximum amount of shared memory allowed to be "
                   "allocated for the given target"),
    llvm::cl::init(163 * 1024));

//===----------------------------------------------------------------------===//
// Bufferization Configuration
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// Common Pass Recipes
//===----------------------------------------------------------------------===//

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
      kNumMaxParallelDims,
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
  pm.addNestedPass<func::FuncOp>(createGenericVectorizationPass(options));
  pm.addNestedPass<func::FuncOp>(createOptimizeTensorInsertExtractSlicesPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createCSEPass());
}

//===---------------------------------------------------------------------===//
// Default Vectorization
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
      createOptimizeTensorInsertExtractSlicesPass());
}

//===---------------------------------------------------------------------===//
// MatmulSIMT
//===---------------------------------------------------------------------===//

void addGPUMatmulSimtPassPipeline(OpPassManager &pm) {
  tileAndDistributeToWorkgroup(pm);
  auto &nestedModulePM = pm.nest<ModuleOp>();

  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addNestedPass<func::FuncOp>(
      createWorkgroupSpecializationPass());
  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addPass(createCSEPass());

  nestedModulePM.addNestedPass<func::FuncOp>(
      createGPUTensorTileToSerialLoops());
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
      createOptimizeTensorInsertExtractSlicesPass());

  // Hoist loop invariant code to avoid pipelining it.
  nestedModulePM.addNestedPass<func::FuncOp>(
      createLoopInvariantCodeMotionPass());
  // Pipeline memory operations.
  nestedModulePM.addNestedPass<func::FuncOp>(createGPUPipeliningPass());
}

//===---------------------------------------------------------------------===//
// Matmul Tensor Core
//===---------------------------------------------------------------------===//

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
      createOptimizeTensorInsertExtractSlicesPass());

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

//===---------------------------------------------------------------------===//
// Matmul MMA.Sync
//===---------------------------------------------------------------------===//

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
      createOptimizeTensorInsertExtractSlicesPass());

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

//===---------------------------------------------------------------------===//
// Transpose
//===---------------------------------------------------------------------===//

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
      createOptimizeTensorInsertExtractSlicesPass());

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

//===---------------------------------------------------------------------===//
// Vector Distribution
//===---------------------------------------------------------------------===//

// Matmul pipeline using vector distribution patterns to map to various tensor
// core operations. The current implementation below is unstable and is missing
// a few crucial pieces for performance (primarily software pipelining). The
// current flow is as follows.
//
// 1. Tile + fuse and distribute to workgroups.
// 2. Problem specific tiling, namely tiling the K dimension of the GEMM.
// 3. Vectorize
// 4. Materialize shared memory allocations as vectorized copies.
// 5. Bufferize
//
// * Distribution to warps should happen here, but right now this pipeline
//   is single subgroup. Pending improvements to vector distribution to allow
//   distribution to warps.
//
// 6. Distribute to virtual lanes (i.e. threads in this case).
//
// Note that a few pieces here are subject to change in the immediate future.
// First, the shared memory promotion done here is in a sense a stopgap, as it
// won't compose well with what's available for bufferization/pipelining today.
// Second, distribution to more than one warp depends on either layout changes,
// or explicit distribution using `scf.forall`. For now this keeps it simple
// and gives us a starting point for generating code for matmuls in the first
// place.

// We use vector ops to do the copy for this pipeline because distribution is
// vector based.
static LogicalResult gpuVectorCopyFn(OpBuilder &builder, Location loc,
                                     Value from, Value to) {
  bool needsBarrier = false;
  MemRefType fromType = llvm::cast<MemRefType>(from.getType());
  if (hasSharedMemoryAddressSpace(fromType)) {
    needsBarrier = true;
  }
  if (hasSharedMemoryAddressSpace(llvm::cast<MemRefType>(to.getType()))) {
    needsBarrier = true;
  }
  if (needsBarrier)
    builder.create<gpu::BarrierOp>(loc);
  VectorType vectorType =
      VectorType::get(fromType.getShape(), fromType.getElementType());
  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  SmallVector<Value> indices(vectorType.getRank(), c0);
  SmallVector<bool> inBounds(vectorType.getRank(), true);
  Value read = builder.create<vector::TransferReadOp>(loc, vectorType, from,
                                                      indices, inBounds);
  builder.create<vector::TransferWriteOp>(loc, read, to, indices, inBounds);
  if (needsBarrier) {
    builder.create<gpu::BarrierOp>(loc);
  }
  return success();
}

static void addVectorBufferizePasses(OpPassManager &passManager) {
  BufferizationOptions::AllocationFn allocationFn = gpuAllocationFn;
  BufferizationOptions::MemCpyFn memcpyFn = gpuCopyFn;
  addIREEComprehensiveBufferizePasses(passManager, allocationFn, memcpyFn);
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());
}

void addGPUVectorDistributePassPipeline(OpPassManager &pm) {
  tileAndDistributeToWorkgroup(pm);
  auto &nestedModulePM = pm.nest<ModuleOp>();
  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addPass(createCSEPass());

  // Problem specific (reduction) tiling.
  nestedModulePM.addNestedPass<func::FuncOp>(
      createGPUTensorTileToSerialLoops());

  nestedModulePM.addNestedPass<func::FuncOp>(
      createOptimizeTensorInsertExtractSlicesPass());

  // Linalg -> Vector
  addGPUVectorizationPasses(nestedModulePM);

  // Allocate tensors for copies to shared memory.
  nestedModulePM.addNestedPass<func::FuncOp>(createGPUVectorAlloc());

  // Tensor -> Memref
  addVectorBufferizePasses(nestedModulePM);
  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addPass(createCSEPass());
  nestedModulePM.addNestedPass<func::FuncOp>(
      createHoistStaticallyBoundAllocationsPass());

  // Vector SIMD -> Vector SIMT
  nestedModulePM.addNestedPass<func::FuncOp>(createLLVMGPUVectorDistribute());
  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addPass(createCSEPass());

  nestedModulePM.addNestedPass<func::FuncOp>(
      createGPUReduceSharedMemoryBankConflicts());
  nestedModulePM.addNestedPass<func::FuncOp>(
      memref::createFoldMemRefAliasOpsPass());
  nestedModulePM.addPass(createCSEPass());
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
    options.enableVectorMasking = true;
    options.useConfiguredVectorSizes = false;
    options.vectorizePadding = true;
    options.vectorizeGatherAccesses = true;
    options.enableCleanup = false;
    options.generateContract = false;
    nestedModulePM.addNestedPass<func::FuncOp>(
        createGenericVectorizationPass(options));
    nestedModulePM.addNestedPass<func::FuncOp>(
        createOptimizeTensorInsertExtractSlicesPass());
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
      createOptimizeTensorInsertExtractSlicesPass());
  nestedModulePM.addNestedPass<func::FuncOp>(
      createLoopInvariantCodeMotionPass());
  nestedModulePM.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  nestedModulePM.addNestedPass<func::FuncOp>(createCSEPass());
  nestedModulePM.addNestedPass<func::FuncOp>(createForOpCanonicalizationPass());
  nestedModulePM.addNestedPass<func::FuncOp>(createCanonicalizerPass());

  auto getSubgroupSizeFn = [](mlir::FunctionOpInterface func) -> int {
    auto moduleOp = func->getParentOfType<ModuleOp>();
    llvm::StringMap<IREE::HAL::ExecutableExportOp> exportOps =
        getAllEntryPoints(moduleOp);
    IREE::HAL::ExecutableExportOp exportOp = exportOps.lookup(func.getName());
    std::optional<int64_t> maybeSubgroupSize = getSubgroupSize(exportOp);
    return maybeSubgroupSize.value_or(kDefaultSubgroupSize);
  };

  // vector -> simt gpu + vector
  nestedModulePM.addNestedPass<func::FuncOp>(
      createConvertVectorReductionToGPUPass(/*expandSubgroupReduction=*/true,
                                            getSubgroupSizeFn));
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

void addGPUDefaultPassPipeline(OpPassManager &pm, bool enableMicrokernels) {
  tileAndDistributeToWorkgroup(pm, /*useWARForCooperativeMatrixCodegen=*/true);
  auto &nestedModulePM = pm.nest<ModuleOp>();
  if (enableMicrokernels) {
    nestedModulePM.addPass(createGPULowerToUKernelsPass());
  }
  nestedModulePM.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  nestedModulePM.addNestedPass<func::FuncOp>(createCSEPass());

  addBufferizePasses(nestedModulePM);
  nestedModulePM.addNestedPass<func::FuncOp>(
      createRemoveSingleIterationLoopPass());
}

void addGPUBaseLoweringPassPipeline(OpPassManager &pm) {
  auto &nestedModulePM = pm.nest<ModuleOp>();

  nestedModulePM.addNestedPass<func::FuncOp>(
      createConvertToDestinationPassingStylePass(
          /*useWARForCooperativeMatrixCodegen=*/false));
  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addPass(createCSEPass());

  addBufferizePasses(nestedModulePM);
  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addPass(createCSEPass());

  nestedModulePM.addNestedPass<func::FuncOp>(
      IREE::LinalgExt::createLinalgExtToLoopsPass());
  nestedModulePM.addNestedPass<func::FuncOp>(createMemrefCopyToLinalgPass());
  nestedModulePM.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());
  nestedModulePM.addNestedPass<func::FuncOp>(
      createRemoveSingleIterationLoopPass());
  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addPass(createCSEPass());
}

// Add passes to make the address computation more explicit and optimize them.
//
// The idea here is to be less dependent on what the LLVM backend is able to do,
// by heavy lifting most of the work while we still have the information about
// loops.
//
// Note that this needs to run before SCF -> CF.
static void addLowerAndOptimizeAddressComputationPasses(OpPassManager &pm) {
  pm.addPass(createExtractAddressComputationGPUPass());
  pm.addNestedPass<func::FuncOp>(memref::createExpandOpsPass());
  pm.addPass(memref::createFoldMemRefAliasOpsPass());
  pm.addPass(memref::createExpandStridedMetadataPass());
  // Hoist loop invariant variables to give affine decomposition pass the right
  // loop dependencies.
  pm.addPass(createLoopInvariantCodeMotionPass());
  // Decompose affine ops.
  pm.addPass(createDecomposeAffineOpsPass());
  // Get rid of the redundant computations.
  pm.addPass(createCSEPass());
  // Hoist the resulting decompositions.
  pm.addPass(createLoopInvariantCodeMotionPass());
  pm.addPass(createLowerAffinePass());
}

static void addLowerToLLVMGPUPasses(OpPassManager &pm, bool forROCDL) {
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

  // Pad allocations with dynamic dimension after linalg lowering but before
  // lowering SCF and affine ops.
  pm.addNestedPass<func::FuncOp>(createPadDynamicAlloc());

  pm.addPass(createLowerAffinePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Handled tensor constants.
  pm.addPass(arith::createConstantBufferizePass());
  pm.addPass(createFoldTensorExtractOpPass());

  pm.addNestedPass<func::FuncOp>(createLLVMGPUVectorLoweringPass());

  // This pass needs to run before SCF -> CF.
  addLowerAndOptimizeAddressComputationPasses(pm);

  // Run checks on shared memory usage.
  // TODO: query this from the target.
  int64_t limit = clLLVMGPUSharedMemoryLimit;
  auto getSharedMemoryLimit = [limit](mlir::FunctionOpInterface) {
    return limit;
  };
  auto getIndexBitwidth = [](mlir::FunctionOpInterface) { return 64; };
  pm.addPass(
      createGPUCheckResourceUsagePass(getSharedMemoryLimit, getIndexBitwidth));

  // SCF -> CF
  pm.addNestedPass<func::FuncOp>(createConvertSCFToCFPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createCSEPass());

  // Handle complex operation conversion.
  pm.addPass(createConvertComplexToStandardPass());

  // Convert BF16 operations to occur as F32.
  pm.addPass(createConvertBf16ArithToF32Pass());
  pm.addPass(createConvertBf16ToUInt16BuffersPass());

  // Convert math dialect elementry functions to polynomial form.
  pm.addNestedPass<func::FuncOp>(createPolynomialApproximationPass());

  pm.addNestedPass<func::FuncOp>(memref::createExpandOpsPass());
  pm.addPass(memref::createFoldMemRefAliasOpsPass());
  pm.addPass(memref::createExpandStridedMetadataPass());
  pm.addPass(createEmulateNarrowTypePass());
  pm.addPass(affine::createAffineExpandIndexOpsPass());
  pm.addPass(createLowerAffinePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Strip out the debug info for the kernel.
  pm.addPass(createStripDebugInfoPass());
  // Cast address spaces of all function arguments to generic.
  pm.addPass(createLLVMGPUCastAddressSpaceFunction());

  if (forROCDL) {
    // convert to ROCDL.
    pm.addPass(createConvertToROCDLPass());
  } else {
    // convert to NVVM.
    pm.addPass(createConvertToNVVMPass());
  }
}

extern llvm::cl::opt<std::string> clGPUCodegenTransformDialectDebugPayloadTag;
extern llvm::cl::opt<std::string> clGPUCodegenTransformDialectDebugTransformTag;

void addGPUTransformDialectPasses(OpPassManager &passManager,
                                  StringRef entryPoint) {
  passManager.addPass(
      mlir::iree_compiler::createTransformDialectInterpreterPass(entryPoint));

  // Dropping the schedule is needed:
  //   1. if we want to embed the transform in the module: we should drop the
  //      schedule once applied.
  //   2. if transform.do_not_dce_operands ops are introduced.
  passManager.addPass(createDropSchedulePass());
}

//===----------------------------------------------------------------------===//
// Common Pass Pipelines
//===----------------------------------------------------------------------===//

void buildLLVMGPUCodegenConfigurationPassPipeline(OpPassManager &pm) {
  addCommonTargetExecutablePreprocessingPasses(pm);
  auto &nestedModulePM = pm.nest<ModuleOp>();
  nestedModulePM.addNestedPass<func::FuncOp>(createGPUGeneralizeNamedOpsPass());
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

//===----------------------------------------------------------------------===//
// ROCDL Pass Pipelines
//===----------------------------------------------------------------------===//

void buildROCDLCodegenConfigurationPassPipeline(OpPassManager &pm) {
  addCommonTargetExecutablePreprocessingPasses(pm);
  auto &nestedModulePM = pm.nest<ModuleOp>();
  nestedModulePM.addNestedPass<func::FuncOp>(createGPUGeneralizeNamedOpsPass());
  pm.addPass(createROCDLSelectLoweringStrategyPass());
}

void buildROCDLCodegenPassPipeline(OpPassManager &pm) {
  pm.addPass(createROCDLLowerExecutableTargetPass());
  OpPassManager &nestedModulePM = pm.nest<ModuleOp>();
  addLowerToLLVMGPUPasses(nestedModulePM, /*forROCDL=*/true);

  LLVM_DEBUG({
    llvm::dbgs() << "Using ROCDL pass pipeline:\n";
    pm.printAsTextualPipeline(llvm::dbgs());
    llvm::dbgs() << "\n";
  });
}

//===---------------------------------------------------------------------===//
// Common Pass Registration
//===---------------------------------------------------------------------===//

namespace common {
#define GEN_PASS_REGISTRATION
#include "iree/compiler/Codegen/LLVMGPU/Passes.h.inc"
} // namespace common

void registerCodegenLLVMGPUPasses() {
  // Generated.
  common::registerPasses();

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

//===---------------------------------------------------------------------===//
// ROCDL Pass Registration
//===---------------------------------------------------------------------===//

namespace rocdl {
#define GEN_PASS_REGISTRATION
#include "iree/compiler/Codegen/LLVMGPU/ROCDLPasses.h.inc"
} // namespace rocdl

void registerCodegenROCDLPasses() {
  // Generated.
  rocdl::registerPasses();

  static PassPipelineRegistration<> ROCDLConfigPipeline(
      "iree-codegen-rocdl-configuration-pipeline",
      "Runs pass pipeline to select a suitable lowering strategy for ROCDL",
      [](OpPassManager &passManager) {
        buildROCDLCodegenConfigurationPassPipeline(passManager);
      });

  static PassPipelineRegistration<> LinalgROCDLPipeline(
      "iree-codegen-linalg-to-rocdl-pipeline2",
      "Runs pass pipeline to progressively lower Linalg to ROCDL",
      [](OpPassManager &passManager) {
        buildROCDLCodegenPassPipeline(passManager);
      });
}

} // namespace mlir::iree_compiler

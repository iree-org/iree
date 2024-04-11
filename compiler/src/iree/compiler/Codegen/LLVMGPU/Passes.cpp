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
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/LLVMGPU/KernelConfig.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/LLVMGPU/ROCDLPasses.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "iree/compiler/Utils/PassUtils.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ComplexToStandard/ComplexToStandard.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToGPU/VectorToGPU.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "iree-llvm-gpu-lowering-pass-pipeline"

namespace mlir::iree_compiler {

constexpr int64_t kDefaultSubgroupSize = 32;

static llvm::cl::opt<ReorderWorkgrupsStrategy> clReorderWorkgroupsStrategy(
    "iree-codegen-reorder-workgroups-strategy",
    llvm::cl::desc("Reorder workgroup IDs using the selected strategy"),
    llvm::cl::values(clEnumValN(ReorderWorkgrupsStrategy::None, "none",
                                "No workgroup reordering"),
                     clEnumValN(ReorderWorkgrupsStrategy::Swizzle, "swizzle",
                                "Swizzle"),
                     clEnumValN(ReorderWorkgrupsStrategy::Transpose,
                                "transpose", "Transpose")),
    llvm::cl::init(ReorderWorkgrupsStrategy::None));

static llvm::cl::opt<unsigned> clReorderWorkgroupsLogSwizzleTile(
    "iree-codegen-reorder-workgroups-log-swizzle-tile",
    llvm::cl::desc("Reorder workgroups: log tile size to use"),
    llvm::cl::init(3));

static llvm::cl::opt<int64_t> clLLVMGPUSharedMemoryLimit(
    "iree-llvmgpu-shared-memory-limit",
    llvm::cl::desc("specify the maximum amount of shared memory allowed to be "
                   "allocated for the given target"),
    llvm::cl::init(163 * 1024));

static llvm::cl::opt<bool> clLLVMGPUEnablePrefetch(
    "iree-llvmgpu-enable-prefetch",
    llvm::cl::desc("Enable prefetch in the vector distribute pipeline"),
    llvm::cl::init(false));

//===----------------------------------------------------------------------===//
// Bufferization Configuration
//===----------------------------------------------------------------------===//

static bool hasThreadMapping(scf::ForallOp forall) {
  if (!forall.getMapping().has_value()) {
    return false;
  }
  return llvm::any_of(*forall.getMapping(),
                      llvm::IsaPred<gpu::GPUThreadMappingAttr>);
  ;
}

// All pipelines that use this allocation function distribute scf.forall ops
// after bufferizing. This means that to differentiate between an allocation in
// function memory and workgroup memory, we need to look for a parent
// scf.forall op with a thread mapping. If not present, we allocate workgroup
// memory. Pipelines that choose to distribute in a different order will have
// to use a different allocation function.
static FailureOr<Value> gpuAllocationFn(OpBuilder &builder, Location loc,
                                        MemRefType memRefType,
                                        ValueRange dynamicSizes,
                                        unsigned alignment) {
  Block *insertionBlock = builder.getInsertionBlock();
  Operation *parent = insertionBlock->getParentOp();
  scf::ForallOp enclosingForall = dyn_cast<scf::ForallOp>(parent);
  if (!enclosingForall) {
    enclosingForall = parent->getParentOfType<scf::ForallOp>();
  }
  gpu::AddressSpaceAttr addressSpace;
  if (enclosingForall && hasThreadMapping(enclosingForall)) {
    addressSpace = gpu::AddressSpaceAttr::get(
        builder.getContext(), gpu::GPUDialect::getPrivateAddressSpace());
  } else {
    addressSpace = gpu::AddressSpaceAttr::get(
        builder.getContext(), gpu::GPUDialect::getWorkgroupAddressSpace());
  }
  MemRefType allocType =
      MemRefType::get(memRefType.getShape(), memRefType.getElementType(),
                      AffineMap(), addressSpace);
  return builder.create<memref::AllocOp>(loc, allocType, dynamicSizes)
      .getResult();
}

// Barriers are only needed when copying to/from workgroup memory. The only
// other kind of memory that can be allocated is function memory, which is local
// to a thread.
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

// Returns success when workgroup reordering is supported for `funcOp`.
// On ROCm, we require workgroup counts to be static.
static LogicalResult canReorderWorkgroups(FunctionOpInterface funcOp) {
  auto target = IREE::HAL::ExecutableTargetAttr::lookup(funcOp);
  if (!target) {
    return failure();
  }
  if (target.getBackend() != "rocm")
    return success();

  // Workgroup reordering on ROCm currently requires all workgrup counts to be
  // static.
  SmallVector<int64_t> workgroupCounts = getStaticNumWorkgroups(funcOp);
  if (workgroupCounts.empty())
    return failure();

  // This is further restricted to 2D+ grids as we reorder along the X and Y
  // workgroup IDs.
  return success(workgroupCounts.size() >= 2);
}

//===----------------------------------------------------------------------===//
// Common Pass Recipes
//===----------------------------------------------------------------------===//

static void addBufferizePasses(OpPassManager &funcPassManager) {
  BufferizationOptions::AllocationFn allocationFn = gpuAllocationFn;
  BufferizationOptions::MemCpyFn memcpyFn = gpuCopyFn;
  addIREEComprehensiveBufferizePasses(funcPassManager, allocationFn, memcpyFn);
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
}

static void
tileAndDistributeToWorkgroup(OpPassManager &funcPassManager,
                             bool useWARForCooperativeMatrixCodegen = false) {
  funcPassManager.addPass(createTileAndDistributeToWorkgroupsPass(
      kNumMaxParallelDims,
      linalg::DistributionMethod::CyclicNumProcsEqNumIters));

  funcPassManager.addPass(createConvertToDestinationPassingStylePass(
      useWARForCooperativeMatrixCodegen));
  // TODO(#16421): Disable decomposition due to failure in bufferization.
  // funcPassManager.addPass(
  //     IREE::LinalgExt::createTileAndDecomposeAttentionPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
}

static void tileAndBufferize(OpPassManager &funcPassManager) {
  tileAndDistributeToWorkgroup(funcPassManager,
                               /*useWARForCooperativeMatrixCodegen=*/true);
  addBufferizePasses(funcPassManager);
}

static void addGPUVectorizationPasses(OpPassManager &funcPassManager) {
  funcPassManager.addPass(createDecomposeConvolutionToLowerDimOpsPass());
  GenericVectorizationPassOptions options;
  options.vectorizePadding = true;
  options.vectorizeGatherAccesses = true;
  options.enableCleanup = false;
  options.foldCastIntoContract = true;
  funcPassManager.addPass(createGenericVectorizationPass(options));
  funcPassManager.addPass(createOptimizeTensorInsertExtractSlicesPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
}

//===---------------------------------------------------------------------===//
// Default Vectorization
//===---------------------------------------------------------------------===//

void addGPUVectorizationPassPipeline(OpPassManager &funcPassManager) {
  tileAndDistributeToWorkgroup(funcPassManager);

  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createWorkgroupSpecializationPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Distribute linalg onto threads within the workgroup.
  funcPassManager.addPass(createGPUTensorTile(false));
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Linalg -> vector
  addGPUVectorizationPasses(funcPassManager);

  // tensor to memref
  addBufferizePasses(funcPassManager);
  funcPassManager.addPass(createGPUDistribute());

  // Post bufferization optimizations.
  funcPassManager.addPass(createLoopInvariantCodeMotionPass());
  funcPassManager.addPass(memref::createFoldMemRefAliasOpsPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
  funcPassManager.addPass(createOptimizeVectorTransferPass());
  funcPassManager.addPass(createOptimizeTensorInsertExtractSlicesPass());
}

//===---------------------------------------------------------------------===//
// MatmulSIMT
//===---------------------------------------------------------------------===//

void addGPUMatmulSimtPassPipeline(OpPassManager &funcPassManager) {
  tileAndDistributeToWorkgroup(funcPassManager);

  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createWorkgroupSpecializationPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  funcPassManager.addPass(createGPUTensorTileToSerialLoops());
  funcPassManager.addPass(createGPUTensorAlloc());
  funcPassManager.addPass(createGPUTensorTile(false));

  // Linalg -> vector
  addGPUVectorizationPasses(funcPassManager);

  // tensor to memref
  addBufferizePasses(funcPassManager);

  // distribute foreach threads
  funcPassManager.addPass(createGPUDistribute());

  funcPassManager.addPass(createMemrefCopyToLinalgPass());
  funcPassManager.addPass(createGPUDistributeSharedMemoryCopy());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  funcPassManager.addPass(createGPUReduceSharedMemoryBankConflicts());
  funcPassManager.addPass(createReorderWorkgroups(
      clReorderWorkgroupsStrategy, clReorderWorkgroupsLogSwizzleTile,
      canReorderWorkgroups));
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  funcPassManager.addPass(memref::createFoldMemRefAliasOpsPass());
  funcPassManager.addPass(createCSEPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Even though we vectorize before bufferization we are not able to hoist
  // accumulator load/store out of the K loop until distribution. Therefore we
  // still rely on buffer level transformations for transfer ops hoisting and
  // store to load forwarding. This relies on shacky alias analysis and we need
  // to move this to tensor level once we have better abstractions.
  funcPassManager.addPass(createOptimizeTensorInsertExtractSlicesPass());

  // Hoist loop invariant code to avoid pipelining it.
  funcPassManager.addPass(createLoopInvariantCodeMotionPass());
  // Pipeline memory operations.
  funcPassManager.addPass(createGPUPipeliningPass());
}

//===---------------------------------------------------------------------===//
// Matmul Tensor Core
//===---------------------------------------------------------------------===//

void addGPUMatmulTensorCorePassPipeline(OpPassManager &funcPassManager,
                                        unsigned pipelineDepth) {
  tileAndBufferize(funcPassManager);

  // Distribute linalg onto warps within the workgroup.
  funcPassManager.addPass(
      createLLVMGPUTileAndDistribute(/*distributeToWarp=*/true));
  funcPassManager.addPass(createRemoveSingleIterationLoopPass());
  if (pipelineDepth > 1)
    funcPassManager.addPass(createGPUMultiBuffering(pipelineDepth));
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  funcPassManager.addPass(createRemoveSingleIterationLoopPass());
  funcPassManager.addPass(createReorderWorkgroups(
      clReorderWorkgroupsStrategy, clReorderWorkgroupsLogSwizzleTile,
      canReorderWorkgroups));
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Linalg -> vector
  funcPassManager.addPass(createLLVMGPUTensorCoreVectorizationPass());
  funcPassManager.addPass(memref::createFoldMemRefAliasOpsPass());
  funcPassManager.addPass(createCSEPass());
  funcPassManager.addPass(createOptimizeVectorTransferPass());
  funcPassManager.addPass(createOptimizeTensorInsertExtractSlicesPass());

  // Distribute shared memory copies.
  funcPassManager.addPass(createMemrefCopyToLinalgPass());
  funcPassManager.addPass(createGPUDistributeSharedMemoryCopy());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
  funcPassManager.addPass(createGPUReduceSharedMemoryBankConflicts());

  // Vector -> MMA ops
  funcPassManager.addPass(memref::createFoldMemRefAliasOpsPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
  funcPassManager.addPass(createLLVMGPUVectorToGPU());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Hoist loop invariant code to avoid pipelining it.
  funcPassManager.addPass(createLoopInvariantCodeMotionPass());
  // Pipeline memory operations.
  funcPassManager.addPass(createGPUPipeliningPass(
      /*epiloguePeeling=*/false, pipelineDepth,
      PipeliningSchedulingStrategy::loadGlobalStage0));
  // Optimize shared memory usage.
  funcPassManager.addPass(createLLVMGPUPackSharedMemoryAlloc());
}

//===---------------------------------------------------------------------===//
// Matmul MMA.Sync
//===---------------------------------------------------------------------===//

void addGPUMatmulTensorCoreMmaSyncPassPipeline(OpPassManager &funcPassManager,
                                               unsigned pipelineDepth) {
  tileAndBufferize(funcPassManager);

  // Distribute linalg onto warps within the workgroup.
  funcPassManager.addPass(
      createLLVMGPUTileAndDistribute(/*distributeToWarp=*/true));
  funcPassManager.addPass(createRemoveSingleIterationLoopPass());
  if (pipelineDepth > 1)
    funcPassManager.addPass(createGPUMultiBuffering(pipelineDepth));
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  funcPassManager.addPass(createRemoveSingleIterationLoopPass());
  funcPassManager.addPass(createReorderWorkgroups(
      clReorderWorkgroupsStrategy, clReorderWorkgroupsLogSwizzleTile,
      canReorderWorkgroups));
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Linalg -> vector
  funcPassManager.addPass(
      createLLVMGPUTensorCoreVectorizationPass(GPUTensorCoreType::MMA_SYNC));
  funcPassManager.addPass(memref::createFoldMemRefAliasOpsPass());
  funcPassManager.addPass(createCSEPass());
  funcPassManager.addPass(createOptimizeVectorTransferPass());
  funcPassManager.addPass(createOptimizeTensorInsertExtractSlicesPass());

  // Distribute shared memory copies.
  funcPassManager.addPass(createMemrefCopyToLinalgPass());
  funcPassManager.addPass(createGPUDistributeSharedMemoryCopy());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Vector -> MMA ops
  funcPassManager.addPass(memref::createFoldMemRefAliasOpsPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
  funcPassManager.addPass(
      createLLVMGPUVectorToGPU(GPUTensorCoreType::MMA_SYNC));
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Hoist loop invariant code to avoid pipelining it.
  funcPassManager.addPass(createLoopInvariantCodeMotionPass());
  // Pipeline memory operations.
  funcPassManager.addPass(createGPUPipeliningPass(
      /*epiloguePeeling=*/false, pipelineDepth,
      PipeliningSchedulingStrategy::nvidiaTensorCore));
  // Optimize shared memory usage.
  funcPassManager.addPass(createLLVMGPUPackSharedMemoryAlloc());
}

//===---------------------------------------------------------------------===//
// Transpose
//===---------------------------------------------------------------------===//

void addGPUTransposePassPipeline(OpPassManager &funcPassManager) {
  tileAndDistributeToWorkgroup(funcPassManager);

  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createWorkgroupSpecializationPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  funcPassManager.addPass(
      createGPUTensorAlloc(GPUPromoteSharedMemPattern::TransposeOpPattern));
  funcPassManager.addPass(createGPUTensorTile(false));

  // Linalg -> vector
  addGPUVectorizationPasses(funcPassManager);
  funcPassManager.addPass(createOptimizeVectorTransferPass());
  funcPassManager.addPass(createOptimizeTensorInsertExtractSlicesPass());

  // tensor to memref
  addBufferizePasses(funcPassManager);

  // distribute foreach threads
  funcPassManager.addPass(createGPUDistribute());

  funcPassManager.addPass(createMemrefCopyToLinalgPass());
  funcPassManager.addPass(createGPUDistributeSharedMemoryCopy());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // May or may not need to reduce shared mememory conflicts
  funcPassManager.addPass(
      createGPUReduceSharedMemoryBankConflicts(/*paddingSizeBits=*/32));
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
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

static void addVectorBufferizePasses(OpPassManager &funcPassManager) {
  BufferizationOptions::AllocationFn allocationFn = gpuAllocationFn;
  BufferizationOptions::MemCpyFn memcpyFn = gpuCopyFn;
  addIREEComprehensiveBufferizePasses(funcPassManager, allocationFn, memcpyFn);
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
}

void addGPUVectorDistributePassPipeline(OpPassManager &funcPassManager) {
  tileAndDistributeToWorkgroup(funcPassManager);
  funcPassManager.addPass(createReorderWorkgroups(
      clReorderWorkgroupsStrategy, clReorderWorkgroupsLogSwizzleTile,
      canReorderWorkgroups));
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Problem specific (reduction) tiling.
  funcPassManager.addPass(createGPUTensorTileToSerialLoops());

  // Generalize all named ops so that we can fold away unit extent dims. By this
  // point, all tiling is finished so the tiling configurations on those ops can
  // be safely dropped. This additionally allows vectorization of convolution to
  // `vector.contract` as filter dimensions are expected to be tiled to 1 by
  // this point.
  funcPassManager.addPass(createLinalgGeneralizeNamedOpsPass());
  LinalgFoldUnitExtentDimsPassOptions options;
  options.useRankReducingSlices = true;
  funcPassManager.addPass(mlir::createLinalgFoldUnitExtentDimsPass(options));
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  funcPassManager.addPass(createOptimizeTensorInsertExtractSlicesPass());
  funcPassManager.addPass(createOptimizeTensorInsertExtractSlicesPass());

  // Linalg -> Vector
  addGPUVectorizationPasses(funcPassManager);

  // Allocate tensors for copies to shared memory.
  funcPassManager.addPass(createGPUVectorAlloc());

  // Tensor -> Memref
  addVectorBufferizePasses(funcPassManager);
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
  funcPassManager.addPass(createHoistStaticallyBoundAllocationsPass());

  // Vector SIMD -> Vector SIMT
  funcPassManager.addPass(createLLVMGPUCastTypeToFitMMAPass());
  funcPassManager.addPass(createLLVMGPUVectorDistribute());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
  funcPassManager.addPass(
      createGPUReduceSharedMemoryBankConflicts(/*paddingSizeBits=*/64));

  funcPassManager.addPass(createGPUReduceSharedMemoryBankConflicts());
  if (clLLVMGPUEnablePrefetch) {
    funcPassManager.addPass(createLLVMGPUPrefetchSharedMemoryPass());
  }
  funcPassManager.addPass(memref::createFoldMemRefAliasOpsPass());
  funcPassManager.addPass(createCSEPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
}

void addGPUWarpReductionPassPipeline(OpPassManager &funcPassManager) {
  tileAndDistributeToWorkgroup(funcPassManager);
  funcPassManager.addPass(createRematerializeParallelOpsPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createGPUTileReductionPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Linalg -> vector
  {
    GenericVectorizationPassOptions options;
    options.enableVectorMasking = true;
    options.useConfiguredVectorSizes = false;
    options.vectorizePadding = true;
    options.vectorizeGatherAccesses = true;
    options.enableCleanup = false;
    options.generateContract = false;
    funcPassManager.addPass(createGenericVectorizationPass(options));
    funcPassManager.addPass(createOptimizeTensorInsertExtractSlicesPass());
    funcPassManager.addPass(createCanonicalizerPass());
    funcPassManager.addPass(createCSEPass());
  }
  funcPassManager.addPass(createLoopInvariantCodeMotionPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  addBufferizePasses(funcPassManager);

  funcPassManager.addPass(memref::createFoldMemRefAliasOpsPass());
  funcPassManager.addPass(createOptimizeVectorTransferPass());
  funcPassManager.addPass(createOptimizeTensorInsertExtractSlicesPass());
  funcPassManager.addPass(createLoopInvariantCodeMotionPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
  funcPassManager.addPass(createForOpCanonicalizationPass());
  funcPassManager.addPass(createCanonicalizerPass());

  auto getSubgroupSizeFn = [](mlir::FunctionOpInterface func) -> int {
    // TODO: This kind of call back function is a really really bad idea
    // This should be easier to resolve than doing this.
    if (std::optional<int64_t> maybeSubgroupSize = getSubgroupSize(func)) {
      return maybeSubgroupSize.value();
    }
    return kDefaultSubgroupSize;
  };

  // vector -> simt gpu + vector
  funcPassManager.addPass(createConvertVectorReductionToGPUPass(
      /*expandSubgroupReduction=*/true, getSubgroupSizeFn));
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
}

void addGPUPackUnPackPasses(OpPassManager &funcPassManager) {
  tileAndDistributeToWorkgroup(funcPassManager);

  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createWorkgroupSpecializationPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  funcPassManager.addPass(createGPUTensorTile(false));
  funcPassManager.addPass(
      createDecomposePackUnPackOpsPass(/*tileOuterToOne=*/true));
  addGPUVectorizationPasses(funcPassManager);

  addBufferizePasses(funcPassManager);

  // distribute foreach threads
  funcPassManager.addPass(createGPUDistribute());

  funcPassManager.addPass(createSplitFullPartialTransferPass("linalg-copy"));
}

void addGPUSimpleDistributePassPipeline(OpPassManager &funcPassManager) {
  tileAndBufferize(funcPassManager);

  // Distribute linalg onto threads within the workgroup.
  funcPassManager.addPass(createLLVMGPUTileAndDistribute());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  funcPassManager.addPass(createRemoveSingleIterationLoopPass());
}

void addGPUDefaultPassPipeline(OpPassManager &funcPassManager,
                               bool enableMicrokernels) {
  tileAndDistributeToWorkgroup(funcPassManager,
                               /*useWARForCooperativeMatrixCodegen=*/true);
  if (enableMicrokernels) {
    funcPassManager.addPass(createGPULowerToUKernelsPass());
  }
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  addBufferizePasses(funcPassManager);
  funcPassManager.addPass(createRemoveSingleIterationLoopPass());
}

void addGPUBaseLoweringPassPipeline(OpPassManager &funcPassManager) {
  funcPassManager.addPass(createConvertToDestinationPassingStylePass(
      /*useWARForCooperativeMatrixCodegen=*/false));
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  addBufferizePasses(funcPassManager);
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  funcPassManager.addPass(IREE::LinalgExt::createLinalgExtToLoopsPass());
  funcPassManager.addPass(createMemrefCopyToLinalgPass());
  funcPassManager.addPass(createConvertLinalgToLoopsPass());
  funcPassManager.addPass(createRemoveSingleIterationLoopPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
}

// Add passes to make the address computation more explicit and optimize them.
//
// The idea here is to be less dependent on what the LLVM backend is able to do,
// by heavy lifting most of the work while we still have the information about
// loops.
//
// Note that this needs to run before SCF -> CF.
static void
addLowerAndOptimizeAddressComputationPasses(FunctionLikeNest &funcPassManager) {
  funcPassManager.addPass(createExtractAddressComputationGPUPass)
      .addPass(memref::createExpandOpsPass)
      .addPass(memref::createFoldMemRefAliasOpsPass)
      .addPass(memref::createExpandStridedMetadataPass)
      // Hoist loop invariant variables to give affine decomposition pass the
      // right loop dependencies.
      .addPass(createLoopInvariantCodeMotionPass)
      // Decompose affine ops.
      .addPass(createDecomposeAffineOpsPass)
      // Get rid of the redundant computations.
      .addPass(createCSEPass)
      // Hoist the resulting decompositions.
      .addPass(createLoopInvariantCodeMotionPass)
      .addPass(createLowerAffinePass);
}

static void addLowerToLLVMGPUPasses(OpPassManager &modulePassManager,
                                    bool forROCDL) {
  modulePassManager.addPass(
      createConvertHALDescriptorTypeToGPUAddressSpacePass());
  modulePassManager.addPass(createCanonicalizerPass());
  modulePassManager.addPass(createCSEPass());

  modulePassManager.addPass(createLowerUKernelOpsToCallsPass());

  FunctionLikeNest(modulePassManager)
      // LinalgExt -> SCF
      .addPass(IREE::LinalgExt::createLinalgExtToLoopsPass)

      // Linalg -> SCF
      .addPass(createMemrefCopyToLinalgPass)
      .addPass(createConvertLinalgToLoopsPass)
      .addPass(createCanonicalizerPass)
      .addPass(createCSEPass)

      // Pad allocations with dynamic dimension after linalg lowering but before
      // lowering SCF and affine ops.
      .addPass(createPadDynamicAlloc)

      .addPass(createLowerAffinePass)
      .addPass(createCanonicalizerPass)
      .addPass(createCSEPass);

  // Handled tensor constants.
  modulePassManager.addPass(arith::createConstantBufferizePass());

  FunctionLikeNest funcPassManager(modulePassManager);
  funcPassManager.addPass(createFoldTensorExtractOpPass)
      .addPass(createLLVMGPUVectorLoweringPass);

  // This pass needs to run before SCF -> CF.
  addLowerAndOptimizeAddressComputationPasses(funcPassManager);

  // Run checks on shared memory usage.
  funcPassManager
      .addPass([&]() {
        // TODO: query this from the target.
        auto getSharedMemoryLimit = [](mlir::FunctionOpInterface entryPoint) {
          return getTargetSharedMemoryLimitInBytes(entryPoint);
        };
        auto getIndexBitwidth = [](mlir::FunctionOpInterface) { return 64; };
        return createGPUCheckResourceUsagePass(getSharedMemoryLimit,
                                               getIndexBitwidth);
      })
      // SCF -> CF
      .addPass(createConvertSCFToCFPass)
      .addPass(createCanonicalizerPass)
      .addPass(createCSEPass)
      // Handle complex operation conversion.
      .addPass(createConvertComplexToStandardPass)
      // Convert BF16 operations to occur as F32.
      .addPass(createConvertBf16ArithToF32Pass)
      .addPass(createConvertBf16ToUInt16BuffersPass)
      // Convert math dialect elementry functions to polynomial form.
      .addPass(createPolynomialApproximationPass)
      .addPass(memref::createExpandOpsPass)
      .addPass(memref::createFoldMemRefAliasOpsPass)
      .addPass(memref::createExpandStridedMetadataPass)
      .addPass(createEmulateNarrowTypePass)
      .addPass(affine::createAffineExpandIndexOpsPass)
      .addPass(createLowerAffinePass)
      .addPass(createCanonicalizerPass)
      .addPass(createCSEPass);

  // Strip out the debug info for the kernel.
  modulePassManager.addPass(createStripDebugInfoPass());
  // Cast address spaces of all function arguments to generic.
  modulePassManager.addPass(createLLVMGPUCastAddressSpaceFunction());

  if (forROCDL) {
    // convert to ROCDL.
    modulePassManager.addPass(createConvertToROCDLPass());
  } else {
    // convert to NVVM.
    modulePassManager.addPass(createConvertToNVVMPass());
  }
}

extern llvm::cl::opt<std::string> clGPUCodegenTransformDialectDebugPayloadTag;
extern llvm::cl::opt<std::string> clGPUCodegenTransformDialectDebugTransformTag;

void addGPUTransformDialectPasses(OpPassManager &funcPassManager,
                                  StringRef entryPoint) {
  funcPassManager.addPass(
      mlir::iree_compiler::createTransformDialectInterpreterPass(entryPoint));

  // Dropping the schedule is needed:
  //   1. if we want to embed the transform in the module: we should drop the
  //      schedule once applied.
  //   2. if transform.do_not_dce_operands ops are introduced.
  funcPassManager.addPass(createDropSchedulePass());
}

//===----------------------------------------------------------------------===//
// Common Pass Pipelines
//===----------------------------------------------------------------------===//

static void buildLLVMGPUCodegenConfigurationPassPipelineImpl(
    OpPassManager &modulePassManager) {
  {
    FunctionLikeNest funcPassManager(modulePassManager);
    funcPassManager.addPass(createGPUGeneralizeNamedOpsPass);
    addCommonTargetExecutablePreprocessingPasses(funcPassManager);
  }
  modulePassManager.addPass(createMaterializeUserConfigsPass());

  modulePassManager.addPass(createLLVMGPUSelectLoweringStrategyPass());
}

void buildLLVMGPUCodegenConfigurationPassPipeline(
    OpPassManager &variantPassManager) {
  buildLLVMGPUCodegenConfigurationPassPipelineImpl(
      variantPassManager.nest<ModuleOp>());
}

void buildLLVMGPUCodegenPassPipeline(OpPassManager &variantPassManager,
                                     bool useROCM) {
  OpPassManager &modulePassManager = variantPassManager.nest<ModuleOp>();
  modulePassManager.addPass(createLowerExecutableUsingTransformDialectPass());
  FunctionLikeNest(modulePassManager)
      .addPass(createLLVMGPULowerExecutableTargetPass);
  variantPassManager.addPass(createReconcileTranslationInfoPass());
  //===--------------------------------------------------------------------===//
  // Convert Linalg ops to LLVM+NVVM/ROCDL ops.
  //
  // Post-conditions:
  //   - All Linalg/Loops/GPU/Affine/Standard ops are converted away.
  //   - The module contains the final llvm.module ready to be serialized.
  //===--------------------------------------------------------------------===//
  addLowerToLLVMGPUPasses(modulePassManager, useROCM);

  LLVM_DEBUG({
    llvm::dbgs() << "Using LLVMGPU pass pipeline:\n";
    variantPassManager.printAsTextualPipeline(llvm::dbgs());
    llvm::dbgs() << "\n";
  });
}

//===----------------------------------------------------------------------===//
// ROCDL Pass Pipelines
//===----------------------------------------------------------------------===//

static void buildROCDLCodegenConfigurationPassPipelineImpl(
    OpPassManager &modulePassManager) {
  {
    FunctionLikeNest funcPassManager(modulePassManager);
    funcPassManager.addPass(createGPUGeneralizeNamedOpsPass);
    addCommonTargetExecutablePreprocessingPasses(funcPassManager);
  }
  modulePassManager.addPass(createMaterializeUserConfigsPass());

  modulePassManager.addPass(createROCDLSelectLoweringStrategyPass());
}

void buildROCDLCodegenConfigurationPassPipeline(
    OpPassManager &variantPassManager) {
  OpPassManager &modulePassManager = variantPassManager.nest<ModuleOp>();
  buildROCDLCodegenConfigurationPassPipelineImpl(modulePassManager);
}

void buildROCDLCodegenPassPipeline(OpPassManager &variantPassManager) {
  OpPassManager &modulePassManager = variantPassManager.nest<ModuleOp>();
  modulePassManager.addPass(createLowerExecutableUsingTransformDialectPass());
  FunctionLikeNest(modulePassManager)
      .addPass(createROCDLLowerExecutableTargetPass);
  variantPassManager.addPass(createReconcileTranslationInfoPass());
  addLowerToLLVMGPUPasses(modulePassManager, /*forROCDL=*/true);

  LLVM_DEBUG({
    llvm::dbgs() << "Using ROCDL pass pipeline:\n";
    variantPassManager.printAsTextualPipeline(llvm::dbgs());
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
      "Runs the translation strategy configuration pipeline on Linalg for GPU "
      "on all functions in a module",
      [](OpPassManager &modulePassManager) {
        buildLLVMGPUCodegenConfigurationPassPipelineImpl(modulePassManager);
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
      [](OpPassManager &modulePassManager) {
        buildROCDLCodegenConfigurationPassPipelineImpl(modulePassManager);
      });

  static PassPipelineRegistration<> LinalgROCDLPipeline(
      "iree-codegen-linalg-to-rocdl-pipeline2",
      "Runs pass pipeline to progressively lower Linalg to ROCDL",
      [](OpPassManager &passManager) {
        buildROCDLCodegenPassPipeline(passManager);
      });

  static PassPipelineRegistration<> LLVMGPUBufferizePipeline(
      "iree-codegen-llvmgpu-bufferization-pipeline",
      "Runs pass pipeline to bufferize for llvmgpu backends",
      [](OpPassManager &passManager) { addBufferizePasses(passManager); });
}

} // namespace mlir::iree_compiler

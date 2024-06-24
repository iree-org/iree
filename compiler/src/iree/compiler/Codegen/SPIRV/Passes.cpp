// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- Passes.cpp - Pipelines from Linalg ops to SPIR-V -------------------===//
//
// This file contains various pipelines to lower IREE HAL executables containing
// Linalg ops to SPIR-V.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"

#include "iree-dialects/Dialect/LinalgTransform/Passes.h"
#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/SPIRV/KernelConfig.h"
#include "iree/compiler/Codegen/SPIRV/Passes.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "iree/compiler/Utils/PassUtils.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/Support/Debug.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ComplexToStandard/ComplexToStandard.h"
#include "mlir/Conversion/MemRefToSPIRV/MemRefToSPIRV.h"
#include "mlir/Conversion/MemRefToSPIRV/MemRefToSPIRVPass.h"
#include "mlir/Conversion/TosaToArith/TosaToArith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "iree-spirv-lowering-pass-pipeline"

namespace mlir::iree_compiler {

static llvm::cl::opt<int> clSPIRVIndexingBits(
    "iree-spirv-index-bits",
    llvm::cl::desc("Set the bit width of indices in SPIR-V."),
    llvm::cl::init(32));

//===----------------------------------------------------------------------===//
// Bufferization Configuration
//===----------------------------------------------------------------------===//

static FailureOr<Value> gpuAllocateWorkgroupMemoryFn(OpBuilder &builder,
                                                     Location loc,
                                                     MemRefType memRefType,
                                                     ValueRange dynamicSizes,
                                                     unsigned alignment) {
  auto workgroupSpace = gpu::AddressSpaceAttr::get(
      builder.getContext(), gpu::GPUDialect::getWorkgroupAddressSpace());
  MemRefType allocType =
      MemRefType::get(memRefType.getShape(), memRefType.getElementType(),
                      AffineMap(), workgroupSpace);
  auto allocOp = builder.create<memref::AllocOp>(
      loc, allocType, dynamicSizes, builder.getI64IntegerAttr(alignment));
  return allocOp.getResult();
}

static FailureOr<Value> gpuAllocateFunctionMemoryFn(OpBuilder &builder,
                                                    Location loc,
                                                    MemRefType memRefType,
                                                    ValueRange dynamicSizes,
                                                    unsigned alignment) {
  std::optional<unsigned> space =
      spirv::mapVulkanStorageClassToMemorySpace(spirv::StorageClass::Function);
  MemRefType allocType = MemRefType::get(
      memRefType.getShape(), memRefType.getElementType(), {}, *space);
  auto allocaOp = builder.create<memref::AllocaOp>(
      loc, allocType, dynamicSizes, builder.getI64IntegerAttr(alignment));
  return allocaOp.getResult();
}

static LogicalResult gpuCopyFn(OpBuilder &builder, Location loc, Value from,
                               Value to) {
  auto fromType = llvm::cast<MemRefType>(from.getType());
  auto toType = llvm::cast<MemRefType>(to.getType());

  bool needsBarrier = hasSharedMemoryAddressSpace(fromType) ||
                      hasSharedMemoryAddressSpace(toType);
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

static void addTileAndDistributeToWorkgroupsPasses(
    OpPassManager &funcPassManager,
    bool useFuseTensorPadWithConsumerPass = false,
    bool useWARForCooperativeMatrixCodegen = false) {
  funcPassManager.addPass(createTileAndDistributeToWorkgroupsPass(
      kNumMaxParallelDims,
      linalg::DistributionMethod::CyclicNumProcsEqNumIters));
  if (useFuseTensorPadWithConsumerPass) {
    funcPassManager.addPass(createFuseTensorPadWithConsumerPass());
  }
  funcPassManager.addPass(createConvertToDestinationPassingStylePass(
      useWARForCooperativeMatrixCodegen));
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
}

/// Adds passes to lower vector ops to meet SPIR-V requirements.
void addSPIRVVectorLoweringPasses(OpPassManager &funcPassManager) {
  funcPassManager.addPass(createSPIRVInitialVectorLoweringPass());
  funcPassManager.addPass(createOptimizeTensorInsertExtractSlicesPass());
  funcPassManager.addPass(createSPIRVFinalVectorLoweringPass());
}

static void addBufferizePasses(OpPassManager &funcPassManager,
                               BufferizationOptions::AllocationFn fn) {
  BufferizationOptions::AllocationFn allocationFn = fn;
  BufferizationOptions::MemCpyFn memcpyFn = gpuCopyFn;
  addIREEComprehensiveBufferizePasses(funcPassManager, allocationFn, memcpyFn);
}

static void
addSPIRVBufferizePasses(OpPassManager &funcPassManager,
                        BufferizationOptions::AllocationFn allocationFn) {
  // Resolve dim ops first so that we don't have compute Linalg ops lingering on
  // becuase of dim op usage. This avoids bufferizing those compute ops just for
  // their shape dimensions.
  funcPassManager.addPass(memref::createResolveShapedTypeResultDimsPass());
  addBufferizePasses(funcPassManager, allocationFn);
  // Distribute immediately after bufferization to avoid losing attribute
  // annotations in subsequent transformations. This is a bit fragile right now
  // but we expect upstream for loops to eventually recognize distribution as a
  // first-class attribute then we don't need this.
  funcPassManager.addPass(createGPUDistributeScfForPass());
  funcPassManager.addPass(memref::createResolveShapedTypeResultDimsPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
  funcPassManager.addPass(createCleanupBufferAllocViewPass());
}

/// Adds passes to materialize structured ops as loops. This replaces structured
/// ops with loop nests containing payloads, so it should be invoked after
/// tiling and vectorization and before buffer transformations.
static void addLoopMaterializationPasses(OpPassManager &funcPassManager) {
  funcPassManager.addPass(IREE::LinalgExt::createLinalgExtToLoopsPass());
  funcPassManager.addPass(createMemrefCopyToLinalgPass());
  funcPassManager.addPass(createConvertLinalgToLoopsPass());
  funcPassManager.addPass(createRemoveSingleIterationLoopPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
}

/// Adds passes to lowering MemRefs. This folds MemRef subviews, flattens n-D
/// MemRef into 1-D ones, vectorizes load/store when possible, and performs
/// cross loop nest optimizations. This should be invoked after structured op
/// lowering and before final SPIR-V conversion.
static void addMemRefLoweringPasses(OpPassManager &modulePassManager) {
  FunctionLikeNest funcPassManager(modulePassManager);

  funcPassManager.addPass(createCanonicalizerPass)
      .addPass(createCSEPass)
      .addPass(createConvertComplexToStandardPass)

      // Math dialect elementry functions -> polynomial form.
      .addPass(createPolynomialApproximationPass)

      .addPass(createPadDynamicAlloc);

  // Check to make sure we are not exceeding shared memory usage limit.
  auto getSharedMemoryLimit = [](mlir::FunctionOpInterface fn) {
    IREE::GPU::TargetAttr target = getGPUTargetAttr(fn);
    return target.getWgp().getMaxWorkgroupMemoryBytes();
  };
  // TODO: query this from the target.
  auto getIndexBitwidth = [](mlir::FunctionOpInterface) { return 32; };
  funcPassManager
      .addPass([&]() {
        return createGPUCheckResourceUsagePass(getSharedMemoryLimit,
                                               getIndexBitwidth);
      })

      // Fold load/store from/to subview ops into the original memref when
      // possible. In SPIR-V we don't use memref descriptor so it's not possible
      // to handle subview ops.
      .addPass(memref::createFoldMemRefAliasOpsPass)
      .addPass(createEmulateNarrowTypePass)
      .addPass(memref::createExpandOpsPass)
      .addPass(createCanonicalizerPass)
      .addPass(createCSEPass)

      // Turn scalar load/store from memrefs into vectorized ones if possible.
      // This gives better memory access patterns, which is very important for
      // perf.
      .addPass(createSPIRVVectorizeLoadStore)
      // Perform optimizations that need to across the scf.for region boundary.
      .addPass(createForOpCanonicalizationPass)
      // Perform various vector-level cross-op optimizations like load-store
      // forwarding, shape casting and casting op cancelling.
      .addPass([&]() { return createOptimizeVectorTransferPass(); })
      .addPass(createSPIRVBreakDownLargeVectorPass)

      // Perform optimizations that need to across the scf.for region boundary.
      .addPass(createForOpCanonicalizationPass)
      .addPass(createCanonicalizerPass)
      .addPass(createCSEPass)
      .addPass([&]() { return createOptimizeVectorTransferPass(); });

  // Turn multi-dimension memref into one-dimension. This is needed for
  // SPIR-V because we don't use upstream memref descriptors.
  modulePassManager.addPass(createFlattenMemRefSubspanPass());

  FunctionLikeNest(modulePassManager)
      .addPass(createSPIRVEraseStorageBufferStaticShapePass);
}

/// Adds passes to perform the final SPIR-V conversion.
static void addSPIRVLoweringPasses(OpPassManager &modulePassManager) {
  FunctionLikeNest(modulePassManager)
      .addPass(createCanonicalizerPass)
      .addPass(createCSEPass)
      .addPass(createLowerAffinePass)

      // Lower ApplyScale before the i64 Emulation Pass so that new 64-bit ops
      // are also emulated if not supported by the target.
      .addPass([&]() {
        return tosa::createTosaToArith(/*includeApplyRescale=*/true,
                                       /*use32BitApplyRescale=*/true);
      })
      .addPass(createCanonicalizerPass)
      .addPass(createCSEPass)
      .addPass(createSPIRVMapMemRefStorageClassPass)
      .addPass(createSPIRVEmulateI64Pass)
      .addPass(createConvertBf16ArithToF32Pass)
      .addPass(createConvertBf16ToUInt16BuffersPass)
      .addPass(createCanonicalizerPass)
      .addPass(createCSEPass);

  modulePassManager.addPass(createSPIRVConvertGPUTargetPass());
  modulePassManager.addPass(createConvertToSPIRVPass(clSPIRVIndexingBits));

  auto getTargetEnv = [](spirv::ModuleOp moduleOp) {
    return moduleOp->getParentOfType<mlir::ModuleOp>()
        ->getAttrOfType<spirv::TargetEnvAttr>(spirv::getTargetEnvAttrName());
  };

  OpPassManager &spirvModulePassManager =
      modulePassManager.nest<spirv::ModuleOp>();
  spirvModulePassManager.addPass(
      spirv::createUnifyAliasedResourcePass(getTargetEnv));
  spirvModulePassManager.addPass(spirv::createSPIRVLowerABIAttributesPass());
  spirvModulePassManager.addPass(createCanonicalizerPass());
  spirvModulePassManager.addPass(createCSEPass());
  spirvModulePassManager.addPass(spirv::createSPIRVRewriteInsertsPass());
  spirvModulePassManager.addPass(spirv::createSPIRVCanonicalizeGLPass());
  spirvModulePassManager.addPass(spirv::createSPIRVUpdateVCEPass());
}

//===----------------------------------------------------------------------===//
// Pass Pipelines
//===----------------------------------------------------------------------===//

void addSPIRVBaseLoweringPassPipeline(OpPassManager &funcPassManager) {
  funcPassManager.addPass(createConvertToDestinationPassingStylePass(
      /*useWARForCooperativeMatrixCodegen=*/false));
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  addBufferizePasses(funcPassManager, gpuAllocateWorkgroupMemoryFn);
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  addLoopMaterializationPasses(funcPassManager);
}

void addSPIRVBaseDistributePassPipeline(OpPassManager &funcPassManager) {
  addTileAndDistributeToWorkgroupsPasses(funcPassManager);

  addBufferizePasses(funcPassManager, gpuAllocateWorkgroupMemoryFn);

  // Tile and distribute to GPU invocations.
  funcPassManager.addPass(createSPIRVTileAndDistributePass());
  funcPassManager.addPass(createMemrefCopyToLinalgPass());
  funcPassManager.addPass(createGPUDistributeSharedMemoryCopyPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  addLoopMaterializationPasses(funcPassManager);
}

void addSPIRVBaseVectorizePassPipeline(OpPassManager &funcPassManager) {
  addTileAndDistributeToWorkgroupsPasses(
      funcPassManager, /*useFuseTensorPadWithConsumerPass=*/true);

  funcPassManager.addPass(createFoldAffineMinInDistributedLoopsPass());
  funcPassManager.addPass(memref::createResolveShapedTypeResultDimsPass());

  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Tile to GPU invocations and vectorize.
  funcPassManager.addPass(createGPUCreateFastSlowPathPass());
  funcPassManager.addPass(createGPUTilePass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
  {
    GenericVectorizationPassOptions options;
    options.vectorizeGatherAccesses = true;
    funcPassManager.addPass(createGenericVectorizationPass(options));
  }
  addSPIRVVectorLoweringPasses(funcPassManager);
  funcPassManager.addPass(createForOpCanonicalizationPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Bufferize and distribute.
  addSPIRVBufferizePasses(funcPassManager, gpuAllocateFunctionMemoryFn);

  // Generate loop nests for all remaining ops and remove trivial loops.
  addLoopMaterializationPasses(funcPassManager);

  // Perform various vector-level cross-op optimizations like load-store
  // forwarding, shape casting and casting op cancelling.
  funcPassManager.addPass(createOptimizeVectorTransferPass());
}

void addSPIRVWinogradVectorizePassPipeline(OpPassManager &funcPassManager) {
  addTileAndDistributeToWorkgroupsPasses(
      funcPassManager, /*useFuseTensorPadWithConsumerPass=*/true);

  funcPassManager.addPass(createFoldAffineMinInDistributedLoopsPass());
  funcPassManager.addPass(memref::createResolveShapedTypeResultDimsPass());

  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  funcPassManager.addPass(createGPUTilePass());
  funcPassManager.addPass(
      IREE::LinalgExt::createDecomposeWinogradTransformPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Tile to GPU invocations and vectorize.
  funcPassManager.addPass(createSPIRVAnnotateWinogradLoopsPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
  {
    GenericVectorizationPassOptions options;
    options.vectorizeGatherAccesses = true;
    options.enableCleanup = true;
    funcPassManager.addPass(createGenericVectorizationPass(options));
  }
  addSPIRVVectorLoweringPasses(funcPassManager);
  funcPassManager.addPass(createForOpCanonicalizationPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Bufferize and distribute.
  addSPIRVBufferizePasses(funcPassManager, gpuAllocateFunctionMemoryFn);

  // Generate loop nests for all remaining ops and remove trivial loops.
  addLoopMaterializationPasses(funcPassManager);

  // Perform various vector-level cross-op optimizations like load-store
  // forwarding, shape casting and casting op cancelling.
  funcPassManager.addPass(createOptimizeVectorTransferPass());
}

void addSPIRVCooperativeMatrixVectorizePassPipeline(
    OpPassManager &funcPassManager, unsigned pipelineDepth,
    unsigned storeStage) {
  addTileAndDistributeToWorkgroupsPasses(
      funcPassManager, /*useFuseTensorPadWithConsumerPass=*/false,
      /*useWARForCooperativeMatrixCodegen=*/true);

  addBufferizePasses(funcPassManager, gpuAllocateWorkgroupMemoryFn);

  // Tile to GPU workgroups and promote.
  funcPassManager.addPass(createSPIRVTileAndPromotePass(
      /*promoteCMatrix=*/true, /*skipThreadLevel=*/true));
  funcPassManager.addPass(createRemoveSingleIterationLoopPass());
  // Run canonicalization patterns to propagate constant shape sizes after
  // removing trip-one loops.
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Tile and distribute to GPU subgroups.
  funcPassManager.addPass(createSPIRVTileToCooperativeOpsPass());
  funcPassManager.addPass(createRemoveSingleIterationLoopPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Multi-buffer depending on pipeline depth and distribute to shared memory.
  if (pipelineDepth > 0) {
    funcPassManager.addPass(createGPUMultiBufferingPass(
        GPUMultiBufferingPassOptions{pipelineDepth + 1}));
  }
  funcPassManager.addPass(createMemrefCopyToLinalgPass());
  funcPassManager.addPass(createGPUDistributeSharedMemoryCopyPass());

  // Reduce bank conflicts by padding.
  {
    GPUReduceBankConflictsPassOptions options = {};
    options.paddingBits = detail::bankConflictReductionPaddingBits;
    funcPassManager.addPass(createGPUReduceBankConflictsPass(options));
  }

  // Performs high-level n-D mechanical vectorization. This does not perform
  // unrolling or lowering, which is done later.
  {
    GenericVectorizationPassOptions options;
    funcPassManager.addPass(createGenericVectorizationPass(options));
  }

  // With subview ops, vector hoisting won't kick in. So fold memref subview ops
  // before performing vector unrolling and hoisting.
  funcPassManager.addPass(memref::createFoldMemRefAliasOpsPass());

  // Vectorize to cooperative ops.
  funcPassManager.addPass(createSPIRVVectorizeToCooperativeOpsPass());
  funcPassManager.addPass(createCSEPass());
  funcPassManager.addPass(createOptimizeTensorInsertExtractSlicesPass());
  funcPassManager.addPass(createRemoveSingleIterationLoopPass());

  // Run canonicalization patterns to propagate constant shape sizes after
  // removing trip-one loops.
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Perform various vector-level cross-op optimizations like load-store
  // forwarding, shape casting and casting op cancelling.
  funcPassManager.addPass(createOptimizeVectorTransferPass());

  funcPassManager.addPass(createForOpCanonicalizationPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
  funcPassManager.addPass(createSPIRVVectorToGPUSubgroupMMAOpsPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
  addSPIRVVectorLoweringPasses(funcPassManager);

  if (pipelineDepth > 0) {
    PipeliningSchedulingStrategy schedule =
        storeStage == 0 ? PipeliningSchedulingStrategy::loadStoreStage0
                        : PipeliningSchedulingStrategy::loadGlobalStage0;
    GPUPipeliningPassOptions pipelieningOptions = {};
    pipelieningOptions.epiloguePeeling = true;
    pipelieningOptions.depth = pipelineDepth;
    pipelieningOptions.scheduleIndex = llvm::to_underlying(schedule);
    funcPassManager.addPass(createGPUPipeliningPass(pipelieningOptions));
  }
}

void addSPIRVMatmulPromoteVectorizePassPipeline(OpPassManager &funcPassManager,
                                                unsigned pipelineDepth,
                                                unsigned storeStage) {
  // Guards against 0 for consistency with older user provided tuning configs.
  pipelineDepth = pipelineDepth ? pipelineDepth : 1;
  LLVM_DEBUG(llvm::dbgs() << "Non-zero Pipeline Depth: " << pipelineDepth
                          << "\n";);
  addTileAndDistributeToWorkgroupsPasses(
      funcPassManager, /*useFuseTensorPadWithConsumerPass=*/false,
      /*useWARForCooperativeMatrixCodegen=*/true);

  // Promote to workgroups and tile to threads.
  funcPassManager.addPass(createGPUTensorTileToSerialLoopsPass());
  funcPassManager.addPass(createGPUTensorAlloc());
  funcPassManager.addPass(createGPUTensorTilePass());

  // Performs high-level n-D mechanical vectorization. This does not perform
  // unrolling or lowering, which is done later.
  {
    GenericVectorizationPassOptions options;
    options.vectorizePadding = true;
    options.vectorizeGatherAccesses = true;
    options.enableCleanup = false;
    options.maxVectorSize = 4096;
    funcPassManager.addPass(createGenericVectorizationPass(options));
    funcPassManager.addPass(createOptimizeTensorInsertExtractSlicesPass());
    funcPassManager.addPass(createCanonicalizerPass());
    funcPassManager.addPass(createCSEPass());
  }

  // Bufferize.
  addBufferizePasses(funcPassManager, gpuAllocateWorkgroupMemoryFn);

  // Distribute scf.forall to GPU threads.
  funcPassManager.addPass(createGPUDistributePass());

  if (pipelineDepth > 1 || storeStage == 0) {
    GPUMultiBufferingPassOptions multibufferingOptions = {
        storeStage == 0 ? pipelineDepth + 1 : pipelineDepth};
    funcPassManager.addPass(createGPUMultiBufferingPass(multibufferingOptions));
  }

  funcPassManager.addPass(createMemrefCopyToLinalgPass());
  funcPassManager.addPass(createGPUDistributeSharedMemoryCopyPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  {
    GPUReduceBankConflictsPassOptions options = {};
    options.paddingBits = detail::bankConflictReductionPaddingBits;
    funcPassManager.addPass(createGPUReduceBankConflictsPass(options));
  }

  // With subview ops, vector hoisting won't kick in. So fold memref subview ops
  // before performing vector unrolling and hoisting.
  funcPassManager.addPass(memref::createFoldMemRefAliasOpsPass());

  funcPassManager.addPass(createSPIRVInitialVectorLoweringPass());
  funcPassManager.addPass(createCSEPass());
  funcPassManager.addPass(createOptimizeTensorInsertExtractSlicesPass());
  funcPassManager.addPass(createSPIRVFinalVectorLoweringPass());

  funcPassManager.addPass(createForOpCanonicalizationPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
  funcPassManager.addPass(createOptimizeVectorTransferPass());

  // Hoist loop invariant code to avoid pipelining it.
  funcPassManager.addPass(createLoopInvariantCodeMotionPass());
  PipeliningSchedulingStrategy schedule =
      storeStage == 0 ? PipeliningSchedulingStrategy::loadStoreStage0
                      : PipeliningSchedulingStrategy::loadGlobalStage0;
  GPUPipeliningPassOptions pipelieningOptions = {};
  pipelieningOptions.epiloguePeeling = true;
  pipelieningOptions.depth = pipelineDepth;
  pipelieningOptions.scheduleIndex = llvm::to_underlying(schedule);
  funcPassManager.addPass(createGPUPipeliningPass(pipelieningOptions));

  addLoopMaterializationPasses(funcPassManager);
}

void addSPIRVSubgroupReducePassPipeline(OpPassManager &funcPassManager) {
  addTileAndDistributeToWorkgroupsPasses(
      funcPassManager, /*useFuseTensorPadWithConsumerPass=*/true);

  // Fuse input parallel ops into the reduction op so that we don't need to
  // create temporary allocations during bufferization.
  funcPassManager.addPass(createRematerializeParallelOpsPass());
  funcPassManager.addPass(createCanonicalizerPass());

  funcPassManager.addPass(createGPUTileReductionPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Performs high-level n-D mechanical vectorization. This does not perform
  // unrolling or lowering, which is done later.
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

  // Bufferize and distribute.
  // We bufferize before distributing to threads there; so we are still at the
  // block level. Therefore, need to allocate workgroup memory.
  addSPIRVBufferizePasses(funcPassManager, gpuAllocateWorkgroupMemoryFn);

  // Perform various vector-level cross-op optimizations like load-store
  // forwarding, shape casting and casting op cancelling.
  funcPassManager.addPass(createOptimizeVectorTransferPass());

  // Simplify the IR for vector distribution.
  funcPassManager.addPass(memref::createFoldMemRefAliasOpsPass());
  funcPassManager.addPass(createLoopInvariantCodeMotionPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
  funcPassManager.addPass(createForOpCanonicalizationPass());
  funcPassManager.addPass(createCanonicalizerPass());

  auto getWarpSize = [](mlir::FunctionOpInterface func) -> int {
    // TODO: This kind of call back function is a really really bad idea
    // This should be easier to resolve than doing this.
    return *getGPUSubgroupSize(func, /*pickLargest=*/true);
  };

  // Handle vector reduction operations specifically.
  funcPassManager.addPass(createConvertVectorReductionToGPUPass(
      /*expandSubgroupReduction=*/false, getWarpSize));
  // Perform normal vector unrolling and lowering transformations. This breaks
  // vectors down to native machine size.
  addSPIRVVectorLoweringPasses(funcPassManager);
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
}

//===----------------------------------------------------------------------===//
// Entry Point
//===----------------------------------------------------------------------===//

static void buildSPIRVCodegenConfigurationPassPipelineImpl(
    OpPassManager &modulePassManager) {
  {
    FunctionLikeNest funcPassManager(modulePassManager);
    funcPassManager.addPass(createGPUGeneralizeNamedOpsPass);
    addCommonTargetExecutablePreprocessingPasses(funcPassManager);
  }
  modulePassManager.addPass(createMaterializeUserConfigsPass());

  modulePassManager.addPass(createSPIRVSelectLoweringStrategyPass());
}

void buildSPIRVCodegenConfigurationPassPipeline(
    OpPassManager &variantPassManager) {
  OpPassManager &modulePassManager = variantPassManager.nest<ModuleOp>();
  buildSPIRVCodegenConfigurationPassPipelineImpl(modulePassManager);
}

void buildSPIRVCodegenPassPipeline(OpPassManager &variantPassManager) {
  {
    OpPassManager &modulePassManager = variantPassManager.nest<ModuleOp>();
    modulePassManager.addPass(
        createSPIRVLowerExecutableUsingTransformDialectPass());
    FunctionLikeNest(modulePassManager)
        .addPass(createSPIRVLowerExecutableTargetPass);
    addMemRefLoweringPasses(modulePassManager);
  }
  variantPassManager.addPass(createReconcileTranslationInfoPass());

  {
    OpPassManager &modulePassManager = variantPassManager.nest<ModuleOp>();
    addSPIRVLoweringPasses(modulePassManager);
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Using SPIR-V pass pipeline:\n";
    variantPassManager.printAsTextualPipeline(llvm::dbgs());
    llvm::dbgs() << "\n";
  });
}

// NOTE: this runs on the top-level program module containing all hal.executable
// ops.
void buildSPIRVLinkingPassPipeline(OpPassManager &modulePassManager) {
  auto &executablePassManager =
      modulePassManager.nest<IREE::HAL::ExecutableOp>();
  // Trim the allowed target environment (version/capability/extension/etc.) to
  // the minimal requirement needed by compiled spirv.module ops. This helps to
  // increase the chance of linking different variant ops together.
  executablePassManager.addNestedPass<IREE::HAL::ExecutableVariantOp>(
      createSPIRVTrimExecutableTargetEnvPass());
  // Materialize the minimal required target environment into proper device
  // queries to execute in the runtime.
  executablePassManager.addNestedPass<IREE::HAL::ExecutableVariantOp>(
      createSPIRVMaterializeExecutableConditionsPass());
  // Link together executables. This may produce some IR duplication.
  modulePassManager.addPass(createSPIRVLinkExecutablesPass());

  // Cleanup IR duplication.
  modulePassManager.addNestedPass<IREE::HAL::ExecutableOp>(
      mlir::createCanonicalizerPass());
}

//===---------------------------------------------------------------------===//
// Register SPIR-V Passes
//===---------------------------------------------------------------------===//

namespace {
#define GEN_PASS_REGISTRATION
#include "iree/compiler/Codegen/SPIRV/Passes.h.inc"
} // namespace

void registerCodegenSPIRVPasses() {
  // Generated.
  registerPasses();

  static PassPipelineRegistration<> SPIRVConfigPipeline(
      "iree-codegen-spirv-configuration-pipeline",
      "Runs the pipeline for configuring the lowering from linalg to SPIR-V on "
      "all functions in a module",
      [](OpPassManager &modulePassManager) {
        buildSPIRVCodegenConfigurationPassPipelineImpl(modulePassManager);
      });

  static PassPipelineRegistration<> LinalgSPIRVPipeline(
      "iree-codegen-linalg-to-spirv-pipeline",
      "Runs the progressive lowering pipeline from linalg to SPIR-V",
      [](OpPassManager &variantPassManager) {
        buildSPIRVCodegenPassPipeline(variantPassManager);
      });
}

} // namespace mlir::iree_compiler

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
#include "iree/compiler/Codegen/Dialect/GPU/TargetUtils/ConfigUtils.h"
#include "iree/compiler/Codegen/Dialect/GPU/Transforms/Passes.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/Transforms/Passes.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/LLVMGPU/ROCDLPasses.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "iree/compiler/Utils/PassUtils.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ComplexToStandard/ComplexToStandard.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
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

static llvm::cl::opt<ReorderWorkgroupsStrategy> clReorderWorkgroupsStrategy(
    "iree-codegen-reorder-workgroups-strategy",
    llvm::cl::desc("Reorder workgroup IDs using the selected strategy"),
    llvm::cl::values(clEnumValN(ReorderWorkgroupsStrategy::None, "none",
                                "No workgroup reordering"),
                     clEnumValN(ReorderWorkgroupsStrategy::Transpose,
                                "transpose", "Transpose")),
    llvm::cl::init(ReorderWorkgroupsStrategy::None));

static llvm::cl::opt<int64_t> clLLVMGPUSharedMemoryLimit(
    "iree-llvmgpu-shared-memory-limit",
    llvm::cl::desc("specify the maximum amount of shared memory allowed to be "
                   "allocated for the given target"),
    llvm::cl::init(163 * 1024));

static llvm::cl::opt<bool> clLLVMGPUEnableSharedMemoryReuse(
    "iree-llvmgpu-enable-shared-memory-reuse",
    llvm::cl::desc(
        "Enable shared memory reuse in the vector distribute pipeline"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> clDistributeToWorkgroupsUsingForall(
    "iree-llvmgpu-test-distribute-to-workgroups-using-forall",
    llvm::cl::desc("Use scf.forall for distribution to workgroups"),
    llvm::cl::init(false), llvm::cl::Hidden);

static llvm::cl::opt<IREE::Codegen::WorkgroupId>
    clSetWorkgroupDistributionAlong(
        "iree-llvmgpu-set-workgroup-distribution-along",
        llvm::cl::desc(
            "Constrain the workgroup distribution along grid dimensions."),
        llvm::cl::values(clEnumValN(IREE::Codegen::WorkgroupId::IdX, "x",
                                    "Constrain the workgroup distribution to "
                                    "use only workgroups along x."),
                         clEnumValN(IREE::Codegen::WorkgroupId::IdY, "y",
                                    "Constrain the workgroup distribution to "
                                    "use only workgroups along x and y."),
                         clEnumValN(IREE::Codegen::WorkgroupId::IdZ, "z",
                                    "Constrain the workgroup distribution to "
                                    "use only workgroups along x, y and z.")),
        llvm::cl::init(IREE::Codegen::WorkgroupId::IdZ)

    );

//===----------------------------------------------------------------------===//
// Bufferization Configuration
//===----------------------------------------------------------------------===//

static bool hasThreadMapping(scf::ForallOp forall) {
  if (!forall.getMapping().has_value()) {
    return false;
  }
  return llvm::any_of(*forall.getMapping(),
                      llvm::IsaPred<gpu::GPUThreadMappingAttr>);
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
  if (enclosingForall && hasThreadMapping(enclosingForall)) {
    auto addressSpace = gpu::AddressSpaceAttr::get(
        builder.getContext(), gpu::GPUDialect::getPrivateAddressSpace());
    auto allocType =
        MemRefType::get(memRefType.getShape(), memRefType.getElementType(),
                        AffineMap(), addressSpace);
    return builder.create<memref::AllocaOp>(loc, allocType, dynamicSizes)
        .getResult();
  }

  auto addressSpace = gpu::AddressSpaceAttr::get(
      builder.getContext(), gpu::GPUDialect::getWorkgroupAddressSpace());
  auto allocType =
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

// Returns success when workgroup reordering is supported / enabled for
// `funcOp`. On ROCm, we require workgroup counts to be static.
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
  if (llvm::any_of(workgroupCounts, ShapedType::isDynamic))
    return failure();

  // This is further restricted to 2D+ grids as we reorder along the X and Y
  // workgroup IDs.
  return success(workgroupCounts.size() >= 2);
}

// Reconciles workgroup reordering strategy based on the pipeline `option` and
// the CLI flag.
static ReorderWorkgroupsStrategy getReorderWorkgroupsStrategy(
    const std::optional<ReorderWorkgroupsStrategy> &option) {
  return option.value_or(clReorderWorkgroupsStrategy);
}

//===----------------------------------------------------------------------===//
// Common Pass Recipes
//===----------------------------------------------------------------------===//

static void addBufferizePasses(OpPassManager &funcPassManager) {
  funcPassManager.addPass(createROCDLConfigureBufferInstructionsPass());
  funcPassManager.addPass(createGPUBubbleResourceCastsPass());
  BufferizationOptions::AllocationFn allocationFn = gpuAllocationFn;
  BufferizationOptions::MemCpyFn memcpyFn = gpuCopyFn;
  addIREEComprehensiveBufferizePasses(funcPassManager, allocationFn, memcpyFn);
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
}

static void tileAndDistributeToWorkgroup(
    OpPassManager &funcPassManager, bool useForall,
    std::optional<ConvertToDestinationPassingStylePassOptions>
        convertToDpsOptions = ConvertToDestinationPassingStylePassOptions{},
    ReorderWorkgroupsStrategy strategy = ReorderWorkgroupsStrategy::None) {
  if (useForall) {
    assert((strategy == ReorderWorkgroupsStrategy::None ||
            strategy == ReorderWorkgroupsStrategy::Transpose) &&
           "Only None and Transpose reorder strategies are supported with "
           "forall distribution.");
    funcPassManager.addPass(createTileAndDistributeToWorkgroupsWithReordering(
        strategy == ReorderWorkgroupsStrategy::Transpose));
  } else {
    funcPassManager.addPass(createTileAndDistributeToWorkgroupsPass(
        kNumMaxParallelDims,
        linalg::DistributionMethod::CyclicNumProcsEqNumIters));
    funcPassManager.addPass(createCSEPass());
    if (convertToDpsOptions) {
      funcPassManager.addPass(
          createConvertToDestinationPassingStylePass(*convertToDpsOptions));
    }
  }

  // TODO(#16421): Disable decomposition due to failure in bufferization.
  // funcPassManager.addPass(
  //     IREE::LinalgExt::createTileAndDecomposeAttentionPass());
  funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
}

static void tileAndBufferize(OpPassManager &funcPassManager) {
  ConvertToDestinationPassingStylePassOptions options;
  options.useWARForCooperativeMatrixCodegen = true;
  tileAndDistributeToWorkgroup(funcPassManager, /*useForall=*/true,
                               /*convertToDpsOptions=*/options);
  addBufferizePasses(funcPassManager);
}

static void addGPUVectorizationPasses(OpPassManager &funcPassManager,
                                      bool vectorizeCopies = true,
                                      bool enableMasking = false) {
  funcPassManager.addPass(createDecomposeConvolutionToLowerDimOpsPass());
  funcPassManager.addPass(IREE::LinalgExt::createDecomposeIm2colPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
  funcPassManager.addPass(
      IREE::VectorExt::createVectorizeIREEVectorExtOpsPass());
  // Vectorize.
  GenericVectorizationPassOptions options;
  options.vectorizePadding = true;
  options.vectorizeCopies = vectorizeCopies;
  options.vectorizeGatherAccesses = true;
  options.enableCleanup = false;
  options.foldCastIntoContract = true;
  options.enableVectorMasking = enableMasking;
  funcPassManager.addPass(createGenericVectorizationPass(options));
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
  // Run subset hoisting to convert iter_args to vectors.
  funcPassManager.addPass(createOptimizeTensorInsertExtractSlicesPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
}

//===---------------------------------------------------------------------===//
// Default Vectorization
//===---------------------------------------------------------------------===//

void addGPUVectorizationPassPipeline(OpPassManager &funcPassManager) {
  tileAndDistributeToWorkgroup(funcPassManager, /*useForall=*/true);

  funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
  funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Distribute linalg onto threads within the workgroup.
  funcPassManager.addPass(createGPUTensorTilePass());
  funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Linalg -> vector
  addGPUVectorizationPasses(funcPassManager);

  // tensor to memref
  addBufferizePasses(funcPassManager);
  funcPassManager.addPass(createGPUDistributePass());

  // Post bufferization optimizations.
  funcPassManager.addPass(createPropagateDispatchSizeBoundsPass());
  funcPassManager.addPass(createIREELoopInvariantCodeMotionPass());
  funcPassManager.addPass(memref::createFoldMemRefAliasOpsPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
  funcPassManager.addPass(createOptimizeVectorTransferPass());
  funcPassManager.addPass(createOptimizeTensorInsertExtractSlicesPass());
}

//===---------------------------------------------------------------------===//
// Tile and Fuse
//===---------------------------------------------------------------------===//

static FailureOr<Value> gpuRequireMemSpaceAllocationFn(OpBuilder &builder,
                                                       Location loc,
                                                       MemRefType memRefType,
                                                       ValueRange dynamicSizes,
                                                       unsigned alignment) {
  Attribute memorySpace = memRefType.getMemorySpace();
  // Bail out if the memref type specifies a nonnull memory space that is not
  // #gpu.address_space.
  if (memorySpace &&
      !llvm::isa<gpu::AddressSpaceAttr, amdgpu::AddressSpaceAttr>(
          memorySpace)) {
    return failure();
  }

  MemRefType allocType = memRefType;
  auto privateSpace = gpu::AddressSpaceAttr::get(
      builder.getContext(), gpu::GPUDialect::getPrivateAddressSpace());
  auto workgroupSpace = gpu::AddressSpaceAttr::get(
      builder.getContext(), gpu::GPUDialect::getWorkgroupAddressSpace());
  if (!memorySpace) {
    allocType =
        MemRefType::get(memRefType.getShape(), memRefType.getElementType(),
                        AffineMap(), privateSpace);
    memorySpace = privateSpace;
  }

  if (memorySpace == privateSpace) {
    return builder.create<memref::AllocaOp>(loc, allocType, dynamicSizes)
        .getResult();
  }
  allocType =
      MemRefType::get(memRefType.getShape(), memRefType.getElementType(),
                      AffineMap(), workgroupSpace);
  return builder.create<memref::AllocOp>(loc, allocType, dynamicSizes)
      .getResult();
}

static void addGPUBufferizePasses(OpPassManager &funcPassManager) {
  funcPassManager.addPass(createEliminateEmptyTensorsPass());
  funcPassManager.addPass(bufferization::createEmptyTensorToAllocTensorPass());
  funcPassManager.addPass(createGPUInferMemorySpacePass());
  funcPassManager.addPass(createROCDLConfigureBufferInstructionsPass());
  funcPassManager.addPass(createGPUBubbleResourceCastsPass());
  funcPassManager.addPass(createGPUAllocPrivateMemoryForDPSOpsPass());
  BufferizationOptions::AllocationFn allocationFn =
      gpuRequireMemSpaceAllocationFn;
  BufferizationOptions::MemCpyFn memcpyFn = [](OpBuilder &builder, Location loc,
                                               Value from, Value to) {
    builder.create<memref::CopyOp>(loc, from, to);
    return success();
  };
  funcPassManager.addPass(
      createIREEComprehensiveBufferizePass(allocationFn, memcpyFn));

  // Convert linalg.copy to direct loads. This has to be before any
  // canonicalization.
  funcPassManager.addPass(createGPULowerToGlobalLoadsPass());

  addIREEPostBufferizationPasses(funcPassManager);

  funcPassManager.addPass(createROCDLBufferInstructionsOptimizationPass());

  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
}

/// Control function for decomposing pack and unpack ops. Returns true if the
/// op is a PackOp with a DispatchTensorLoadOp producer, or an UnPackOp with
/// only DispatchTensorStoreOp consumers.
LogicalResult isAtBoundary(Operation *op) {
  if (isa<linalg::PackOp>(op)) {
    if (isa_and_nonnull<IREE::TensorExt::DispatchTensorLoadOp>(
            op->getOperand(0).getDefiningOp())) {
      return success();
    }
  } else if (isa<linalg::UnPackOp>(op)) {
    if (llvm::all_of(op->getUsers(), [](Operation *user) {
          return isa<IREE::TensorExt::DispatchTensorStoreOp>(user);
        })) {
      return success();
    }
  }
  return failure();
}

void addGPUTileAndFusePassPipeline(OpPassManager &funcPassManager,
                                   const GPUPipelineOptions &pipelineOptions) {
  if (pipelineOptions.useIgemmConvolution) {
    funcPassManager.addPass(createConvolutionToIGEMMPass());
  }
  // TODO (nirvedhmeshram) : Can remove this pass after
  // https://github.com/iree-org/iree/issues/19546 is fixed.
  funcPassManager.addPass(createConvertAccGEMMToGEMMPass());
  tileAndDistributeToWorkgroup(funcPassManager, /*useForall=*/true,
                               /*convertToDpsOptions=*/std::nullopt);

  // Step 0. Apply any user annotated lowering strategies. This runs first as
  // steps 1 - 4 are essentially applying patterns based on the lowering config,
  // so a custom strategy runs first circumventing that.
  //
  // In the future there may be cases where we want the custom strategy run at
  // later points in the pipeline.
  funcPassManager.addPass(createLoweringConfigInterpreterPass());
  funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Step 1. Promote matmul operands and pack to intrinsic shapes.
  funcPassManager.addPass(createGPUPadOperandsPass());
  funcPassManager.addPass(createGPUPromoteMatmulOperandsPass());
  funcPassManager.addPass(createGPUPackToIntrinsicsPass());
  // Decompose packs and unpacks that are at the function boundary.
  funcPassManager.addPass(createDecomposeBoundaryPackUnPackOpsPass());

  // Step 1.5. Expand result shapes of MultiMmaOps before tiling, and
  // propagate reshapes to the function boundary.
  {
    IREE::GPU::ConcretizeMmaShapesPassOptions options;
    options.concretizeInputs = false;
    options.concretizeResult = true;
    funcPassManager.addPass(IREE::GPU::createConcretizeMmaShapesPass());
  }
  funcPassManager.addPass(createPropagateReshapesByExpansionPass());

  // Step 2. Tile and fuse tileable ops to reduction loops.
  {
    GPUApplyTilingLevelPassOptions options;
    options.tilingLevel = IREE::GPU::TilingLevel::Reduction;
    funcPassManager.addPass(createGPUApplyTilingLevelPass(options));
    funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
    funcPassManager.addPass(createCSEPass());
  }

  // Step 3. Decompose pack and unpack ops and propagate the resulting reshapes.
  funcPassManager.addPass(createDecomposePackUnPackOpsPass(
      DecomposePackUnPackOpsPassOptions{/*tileOuterToOne=*/false,
                                        /*useOnlyReshapes=*/true}));

  // Step 3.5. Expand the inner dimensions of MultiMma ops in preparation for
  // distribution to lanes.
  {
    IREE::GPU::ConcretizeMmaShapesPassOptions options;
    options.concretizeInputs = true;
    options.concretizeResult = false;
    funcPassManager.addPass(IREE::GPU::createConcretizeMmaShapesPass());
  }

  funcPassManager.addPass(createPropagateReshapesByExpansionPass());
  funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Step 4. Tile and fuse tileable ops to subgroups/threads.
  {
    GPUApplyTilingLevelPassOptions options;
    options.tilingLevel = IREE::GPU::TilingLevel::Thread;
    funcPassManager.addPass(createGPUApplyTilingLevelPass(options));
    funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
    funcPassManager.addPass(createCSEPass());
  }
  {
    GPUApplyTilingLevelPassOptions options;
    options.tilingLevel = IREE::GPU::TilingLevel::Subgroup;
    funcPassManager.addPass(createGPUApplyTilingLevelPass(options));
  }
  funcPassManager.addPass(IREE::GPU::createDistributeMmaToLanesPass());

  // Step 4.5. Things that need to happen right after distribution to threads.
  funcPassManager.addPass(createGPULowerToUKernelsPass());

  // Normalize loop bounds for later lowerings.
  funcPassManager.addPass(iree_compiler::createNormalizeLoopBoundsPass(
      NormalizeLoopBoundsPassOptions{/*normalizeFor=*/false,
                                     /*normalizeForall=*/true}));
  funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // TODO: This LICM instance is load bearing due to brittleness of the
  // hoisting and fusion pass, as well as a lack of a fallback distribution
  // pass.
  funcPassManager.addPass(createIREELoopInvariantCodeMotionPass());

  // Drop resource casts if needed. This is the last possible place to do so
  // before greedy fusion.
  funcPassManager.addPass(createGPUBubbleResourceCastsPass());
  {
    OptimizeTensorInsertExtractSlicesPassOptions options;
    options.foldIdentitySlices = true;
    funcPassManager.addPass(
        createOptimizeTensorInsertExtractSlicesPass(options));
  }

  // Step 5. Greedily fuse parallel loops and hoist from serial loops.
  funcPassManager.addPass(createGPUFuseAndHoistParallelLoopsPass());
  funcPassManager.addPass(createGPUGreedilyDistributeToThreadsPass());
  funcPassManager.addPass(createTileLargeTensorsPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
  funcPassManager.addPass(createPropagateDispatchSizeBoundsPass());
  funcPassManager.addPass(createIREELoopInvariantCodeMotionPass());
  funcPassManager.addPass(IREE::GPU::createCombineBarrierRegionsPass());

  // Step 6. Lower special ops and vectorize.
  funcPassManager.addPass(IREE::GPU::createVectorizeIREEGPUOpsPass());
  addGPUVectorizationPasses(funcPassManager, /*vectorizeCopies=*/false,
                            /*enableMasking=*/true);
  funcPassManager.addPass(createCleanupBufferAllocViewPass());
  funcPassManager.addPass(createGPUCombineValueBarriersPass());

  // Step 7. Bufferize.
  addGPUBufferizePasses(funcPassManager);

  // Step 8. Resolve remaining parallel loops.
  funcPassManager.addPass(createGPUDistributeCopyUsingForallPass());
  funcPassManager.addPass(iree_compiler::createNormalizeLoopBoundsPass(
      NormalizeLoopBoundsPassOptions{/*normalizeFor=*/false,
                                     /*normalizeForall=*/true}));
  funcPassManager.addPass(createGPUVerifyDistributionPass());
  funcPassManager.addPass(createGPUDistributeForallPass());

  // Vectorize copies that came out of bufferization.
  funcPassManager.addPass(createVectorizeMemrefCopyPass());

  // Step 8. Unroll operations to native intrinsic widths.
  funcPassManager.addPass(IREE::GPU::createUnrollToIntrinsicsPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Step 9. Remaining post-bufferization optimizations/lowerings.
  funcPassManager.addPass(createPropagateDispatchSizeBoundsPass());
  funcPassManager.addPass(IREE::GPU::createLowerIREEGPUOpsPass());
  funcPassManager.addPass(createUnrollAnnotatedLoopsPass());
  funcPassManager.addPass(createIREELoopInvariantCodeMotionPass());
  if (pipelineOptions.enableReduceSharedMemoryBankConflicts) {
    GPUReduceBankConflictsPassOptions options = {};
    options.paddingBits = 64;
    funcPassManager.addPass(createGPUReduceBankConflictsPass(options));
  }
  if (pipelineOptions.prefetchSharedMemory) {
    funcPassManager.addPass(createFissionTransferOpsInControlFlowPass());
    funcPassManager.addPass(createHoistStaticallyBoundAllocationsPass());
    funcPassManager.addPass(createRemoveSingleIterationLoopPass());
    funcPassManager.addPass(createLLVMGPUPrefetchSharedMemoryPass());
  }

  funcPassManager.addPass(memref::createFoldMemRefAliasOpsPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
  {
    OptimizeVectorTransferPassOptions options;
    // Disable redundant vector transfer hoisting because it does not
    // properly consider distributed code on memrefs.
    options.redundantHoisting = false;
    funcPassManager.addPass(createOptimizeVectorTransferPass());
  }
  funcPassManager.addPass(createHoistStaticallyBoundAllocationsPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
}

//===---------------------------------------------------------------------===//
// Winograd Vectorize
//===---------------------------------------------------------------------===//

void addGPUWinogradVectorizePassPipeline(OpPassManager &funcPassManager) {
  tileAndDistributeToWorkgroup(funcPassManager, /*useForall=*/true);

  funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
  funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Distribute linalg onto threads within the workgroup.
  funcPassManager.addPass(createGPUTilePass());
  funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
  funcPassManager.addPass(
      IREE::LinalgExt::createDecomposeWinogradTransformPass());

  // Linalg -> vector
  addGPUVectorizationPasses(funcPassManager);

  // tensor to memref
  addBufferizePasses(funcPassManager);
  GPUDistributeScfForPassOptions options;
  options.useBlockDims = false;
  funcPassManager.addPass(createGPUDistributeScfForPass(options));

  // Post bufferization optimizations.
  funcPassManager.addPass(createPropagateDispatchSizeBoundsPass());
  funcPassManager.addPass(createIREELoopInvariantCodeMotionPass());
  funcPassManager.addPass(memref::createFoldMemRefAliasOpsPass());
  funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
  funcPassManager.addPass(createOptimizeVectorTransferPass());
  funcPassManager.addPass(createOptimizeTensorInsertExtractSlicesPass());
}

//===---------------------------------------------------------------------===//
// Matmul Tensor Core
//===---------------------------------------------------------------------===//

void addGPUMatmulTensorCorePassPipeline(OpPassManager &funcPassManager,
                                        const GPUPipelineOptions &options,
                                        unsigned pipelineDepth) {
  tileAndBufferize(funcPassManager);

  // Distribute linalg onto warps within the workgroup.
  funcPassManager.addPass(
      createLLVMGPUTileAndDistributePass(/*distributeToWarp=*/true));
  funcPassManager.addPass(createPropagateDispatchSizeBoundsPass());
  funcPassManager.addPass(createRemoveSingleIterationLoopPass());
  if (pipelineDepth > 1) {
    funcPassManager.addPass(createGPUMultiBufferingPass(
        GPUMultiBufferingPassOptions{pipelineDepth}));
  }
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  funcPassManager.addPass(createRemoveSingleIterationLoopPass());

  ReorderWorkgroupsStrategy reorderStrategy =
      getReorderWorkgroupsStrategy(options.reorderStrategy);
  funcPassManager.addPass(
      createReorderWorkgroups(reorderStrategy, canReorderWorkgroups));

  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Linalg -> vector
  funcPassManager.addPass(
      createLLVMGPUTensorCoreVectorizationPass(GPUTensorCoreType::WMMA));
  funcPassManager.addPass(memref::createFoldMemRefAliasOpsPass());
  funcPassManager.addPass(createCSEPass());
  funcPassManager.addPass(createOptimizeVectorTransferPass());
  funcPassManager.addPass(createOptimizeTensorInsertExtractSlicesPass());

  // Distribute shared memory copies.
  funcPassManager.addPass(createMemrefCopyToLinalgPass());
  funcPassManager.addPass(createGPUDistributeSharedMemoryCopyPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
  if (options.enableReduceSharedMemoryBankConflicts) {
    funcPassManager.addPass(createGPUReduceBankConflictsPass());
  }

  // Vector -> MMA ops
  funcPassManager.addPass(memref::createFoldMemRefAliasOpsPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
  funcPassManager.addPass(
      createLLVMGPUVectorToGPUPass(GPUTensorCoreType::WMMA));
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Hoist loop invariant code to avoid pipelining it.
  funcPassManager.addPass(createPropagateDispatchSizeBoundsPass());
  funcPassManager.addPass(createIREELoopInvariantCodeMotionPass());
  // Pipeline memory operations.
  GPUPipeliningPassOptions pipelieningOptions = {};
  pipelieningOptions.epiloguePeeling = false;
  pipelieningOptions.depth = pipelineDepth;
  pipelieningOptions.scheduleIndex =
      llvm::to_underlying(PipeliningSchedulingStrategy::loadGlobalStage0);
  funcPassManager.addPass(createGPUPipeliningPass(pipelieningOptions));
  // Optimize shared memory usage.
  funcPassManager.addPass(createLLVMGPUPackSharedMemoryAllocPass());
}

//===---------------------------------------------------------------------===//
// Matmul MMA.Sync
//===---------------------------------------------------------------------===//

void addGPUMatmulTensorCoreMmaSyncPassPipeline(
    OpPassManager &funcPassManager, const GPUPipelineOptions &options,
    unsigned pipelineDepth) {
  tileAndBufferize(funcPassManager);

  // Distribute linalg onto warps within the workgroup.
  funcPassManager.addPass(
      createLLVMGPUTileAndDistributePass(/*distributeToWarp=*/true));
  funcPassManager.addPass(createPropagateDispatchSizeBoundsPass());
  funcPassManager.addPass(createRemoveSingleIterationLoopPass());
  if (pipelineDepth > 1) {
    funcPassManager.addPass(createGPUMultiBufferingPass(
        GPUMultiBufferingPassOptions{pipelineDepth}));
  }
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  funcPassManager.addPass(createRemoveSingleIterationLoopPass());

  ReorderWorkgroupsStrategy reorderStrategy =
      getReorderWorkgroupsStrategy(options.reorderStrategy);
  funcPassManager.addPass(
      createReorderWorkgroups(reorderStrategy, canReorderWorkgroups));

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
  funcPassManager.addPass(createGPUDistributeSharedMemoryCopyPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Vector -> MMA ops
  funcPassManager.addPass(memref::createFoldMemRefAliasOpsPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
  funcPassManager.addPass(
      createLLVMGPUVectorToGPUPass(GPUTensorCoreType::MMA_SYNC));
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Hoist loop invariant code to avoid pipelining it.
  funcPassManager.addPass(createPropagateDispatchSizeBoundsPass());
  funcPassManager.addPass(createIREELoopInvariantCodeMotionPass());
  // Pipeline memory operations.
  GPUPipeliningPassOptions pipelieningOptions = {};
  pipelieningOptions.epiloguePeeling = false;
  pipelieningOptions.depth = pipelineDepth;
  pipelieningOptions.scheduleIndex =
      llvm::to_underlying(PipeliningSchedulingStrategy::nvidiaTensorCore);
  funcPassManager.addPass(createGPUPipeliningPass(pipelieningOptions));
  // Optimize shared memory usage.
  funcPassManager.addPass(createLLVMGPUPackSharedMemoryAllocPass());
}

//===---------------------------------------------------------------------===//
// Transpose
//===---------------------------------------------------------------------===//

void addGPUTransposePassPipeline(OpPassManager &funcPassManager,
                                 const GPUPipelineOptions &options) {
  tileAndDistributeToWorkgroup(funcPassManager, /*useForall=*/true);

  funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
  funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  funcPassManager.addPass(
      createGPUTensorAlloc(GPUPromoteSharedMemPattern::TransposeOpPattern));
  funcPassManager.addPass(createGPUTensorTilePass());

  // Linalg -> vector
  addGPUVectorizationPasses(funcPassManager);
  funcPassManager.addPass(createOptimizeVectorTransferPass());
  funcPassManager.addPass(createOptimizeTensorInsertExtractSlicesPass());

  // tensor to memref
  addBufferizePasses(funcPassManager);

  // distribute foreach threads
  funcPassManager.addPass(createGPUDistributePass());

  funcPassManager.addPass(createMemrefCopyToLinalgPass());
  funcPassManager.addPass(createGPUDistributeSharedMemoryCopyPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  if (options.enableReduceSharedMemoryBankConflicts) {
    // May or may not need to reduce shared mememory conflicts.
    GPUReduceBankConflictsPassOptions options = {};
    options.paddingBits = 32;
    funcPassManager.addPass(createGPUReduceBankConflictsPass(options));
  }

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
  funcPassManager.addPass(createROCDLConfigureBufferInstructionsPass());
  BufferizationOptions::AllocationFn allocationFn = gpuAllocationFn;
  BufferizationOptions::MemCpyFn memcpyFn = gpuCopyFn;
  addIREEComprehensiveBufferizePasses(funcPassManager, allocationFn, memcpyFn);
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
}

void addGPUVectorDistributePassPipeline(OpPassManager &funcPassManager,
                                        const GPUPipelineOptions &options,
                                        bool usePadToModelSharedMemcpy) {

  ReorderWorkgroupsStrategy reorderStrategy =
      getReorderWorkgroupsStrategy(options.reorderStrategy);

  tileAndDistributeToWorkgroup(funcPassManager, /*useForall=*/true,
                               /*convertToDpsOptions=*/std::nullopt,
                               /*reorderStrategy=*/reorderStrategy);

  if (usePadToModelSharedMemcpy) {
    funcPassManager.addPass(createLLVMGPUPromoteMatmulToFitMMAPass());
  }

  funcPassManager.addPass(
      IREE::LinalgExt::createConvertAttentionToOnlineAttentionPass());

  funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
  funcPassManager.addPass(createGPUPromoteMatmulOperandsPass());

  // Tile to reduction loops.
  {
    GPUApplyTilingLevelPassOptions options;
    options.tilingLevel = IREE::GPU::TilingLevel::Reduction;
    options.allowZeroSlices = true;
    funcPassManager.addPass(createGPUApplyTilingLevelPass(options));
    funcPassManager.addPass(affine::createLoopCoalescingPass());
    funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
    funcPassManager.addPass(createCSEPass());
  }

  // Tile to reduction loops.
  {
    GPUApplyTilingLevelPassOptions options;
    options.tilingLevel = IREE::GPU::TilingLevel::PartialReduction;
    options.allowZeroSlices = true;
    funcPassManager.addPass(createGPUApplyTilingLevelPass(options));
    funcPassManager.addPass(affine::createLoopCoalescingPass());
    funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
    funcPassManager.addPass(createCSEPass());
  }

  funcPassManager.addPass(IREE::LinalgExt::createDecomposeAttentionPass());
  funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Set anchors at tensor level for vector distribution later and hoist out
  // loop invariant anchors.
  funcPassManager.addPass(createDecomposeHorizontallyFusedGemmsPass());
  funcPassManager.addPass(createLLVMGPUConfigureTensorLayoutsPass());
  funcPassManager.addPass(createIREELoopInvariantCodeMotionPass());

  // Generalize all named ops so that we can fold away unit extent dims. By this
  // point, all tiling is finished so the tiling configurations on those ops can
  // be safely dropped. This additionally allows vectorization of convolution to
  // `vector.contract` as filter dimensions are expected to be tiled to 1 by
  // this point.
  funcPassManager.addPass(createLinalgGeneralizeNamedOpsPass());
  if (!usePadToModelSharedMemcpy) {
    LinalgFoldUnitExtentDimsPassOptions options;
    options.useRankReducingSlices = true;
    funcPassManager.addPass(IREE::LinalgExt::createFoldUnitExtentDimsPass());
    funcPassManager.addPass(
        IREE::VectorExt::createVectorExtFoldUnitExtentDimsPass());
    funcPassManager.addPass(mlir::createLinalgFoldUnitExtentDimsPass(options));
    funcPassManager.addPass(createCanonicalizerPass());
    funcPassManager.addPass(createCSEPass());
  }

  funcPassManager.addPass(createOptimizeTensorInsertExtractSlicesPass());

  // Linalg -> Vector
  addGPUVectorizationPasses(funcPassManager);

  // Allocate tensors for copies to shared memory.
  funcPassManager.addPass(createGPUVectorAllocPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
  funcPassManager.addPass(createGPUCombineValueBarriersPass());

  // Tensor -> Memref
  addVectorBufferizePasses(funcPassManager);
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
  funcPassManager.addPass(createHoistStaticallyBoundAllocationsPass());

  // Preprocessing for vector distribution.
  funcPassManager.addPass(createLLVMGPUCastTypeToFitMMAPass());

  // Vector SIMD -> Vector SIMT
  funcPassManager.addPass(createLLVMGPUVectorDistributePass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  if (options.enableReduceSharedMemoryBankConflicts) {
    GPUReduceBankConflictsPassOptions options = {};
    options.paddingBits = 64;
    funcPassManager.addPass(createGPUReduceBankConflictsPass(options));
  }
  if (options.prefetchSharedMemory) {
    funcPassManager.addPass(createLLVMGPUPrefetchSharedMemoryPass());
  }
  if (clLLVMGPUEnableSharedMemoryReuse) {
    funcPassManager.addPass(createHoistStaticallyBoundAllocationsPass());
    funcPassManager.addPass(createGPUReuseSharedMemoryAllocsPass());
  }
  funcPassManager.addPass(memref::createFoldMemRefAliasOpsPass());
  funcPassManager.addPass(createCSEPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
}

void addGPUWarpReductionPassPipeline(OpPassManager &funcPassManager,
                                     bool forROCDL) {
  tileAndDistributeToWorkgroup(
      funcPassManager, /*useForall=*/clDistributeToWorkgroupsUsingForall);
  funcPassManager.addPass(createRematerializeParallelOpsPass());
  funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
  funcPassManager.addPass(createGPUTileReductionPass());
  funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
  funcPassManager.addPass(createPropagateDispatchSizeBoundsPass());

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
  funcPassManager.addPass(createIREELoopInvariantCodeMotionPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  addBufferizePasses(funcPassManager);

  funcPassManager.addPass(memref::createFoldMemRefAliasOpsPass());
  funcPassManager.addPass(createOptimizeVectorTransferPass());
  funcPassManager.addPass(createOptimizeTensorInsertExtractSlicesPass());
  funcPassManager.addPass(createIREELoopInvariantCodeMotionPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
  funcPassManager.addPass(createForOpCanonicalizationPass());
  funcPassManager.addPass(createCanonicalizerPass());

  // vector -> simt gpu + vector
  VectorReductionToGPUPassOptions options;
  options.expandSubgroupReduction = !forROCDL;
  funcPassManager.addPass(createVectorReductionToGPUPass(options));
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
  funcPassManager.addPass(affine::createLoopCoalescingPass());
  funcPassManager.addPass(createCanonicalizerPass());
}

void addGPUSimpleDistributePassPipeline(OpPassManager &funcPassManager) {
  tileAndBufferize(funcPassManager);

  // Distribute linalg onto threads within the workgroup.
  funcPassManager.addPass(createLLVMGPUTileAndDistributePass(
      /*distributeToWarp=*/clDistributeToWorkgroupsUsingForall));
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  funcPassManager.addPass(createPropagateDispatchSizeBoundsPass());
  funcPassManager.addPass(createRemoveSingleIterationLoopPass());
}

void addGPUDefaultPassPipeline(OpPassManager &funcPassManager,
                               const GPUPipelineOptions &options) {
  ConvertToDestinationPassingStylePassOptions dpsOptions;
  dpsOptions.useWARForCooperativeMatrixCodegen = true;
  tileAndDistributeToWorkgroup(funcPassManager, /*useForall=*/true,
                               /*convertToDpsOptions=*/dpsOptions);
  if (options.enableUkernels) {
    funcPassManager.addPass(createGPULowerToUKernelsPass());
  }
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  addBufferizePasses(funcPassManager);
  funcPassManager.addPass(createPropagateDispatchSizeBoundsPass());
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
  funcPassManager.addPass(createPropagateDispatchSizeBoundsPass());
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
      // Lower any remaining vector.transfer_read and vector.transfer_write ops,
      // since some of the following patterns have trouble dealing with their
      // full complexity.
      .addPass(createVectorTransferLoweringPass)
      .addPass(memref::createFoldMemRefAliasOpsPass)
      // Resolve swizzling hints before lowering affine ops but after
      // lowering vector (transfer) ops.
      .addPass(createResolveSwizzleHintsPass)
      // Canonicalize and CSE to attempt to deduplicate swizzle computation.
      .addPass(createCanonicalizerPass)
      .addPass(createCSEPass)
      .addPass(createIREEExpandStridedMetadataPass)
      .addPass(createPropagateDispatchSizeBoundsPass)
      // Hoist loop invariant variables to give affine decomposition pass the
      // right loop dependencies.
      .addPass(createIREELoopInvariantCodeMotionPass)
      // Decompose affine ops.
      .addPass(createDecomposeAffineOpsPass)
      // Get rid of the redundant computations.
      .addPass(createCSEPass)
      // Hoist the resulting decompositions.
      .addPass(createIREELoopInvariantCodeMotionPass)
      .addPass(affine::createAffineExpandIndexOpsPass)
      .addPass(createLowerAffinePass)
      .addPass([]() {
        return IREE::Util::createOptimizeIntArithmeticPass(
            IREE::Util::OptimizeIntArithmeticPassOptions{/*narrowToI32=*/true});
      })
      // Do another round of LICM now that we've lowered and optimized
      // arithmetic
      .addPass(createCSEPass)
      .addPass(createIREELoopInvariantCodeMotionPass);
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
      .addPass(createPadDynamicAllocPass)
      // Hoist any newly static allocations from PadDynamicAlloc.
      .addPass(createHoistStaticallyBoundAllocationsPass)

      .addPass(createLowerAffinePass)
      .addPass(createCanonicalizerPass)
      .addPass(createCSEPass);

  // Handled tensor constants.
  modulePassManager.addPass(createIREEBufferizeConstantsPass());

  FunctionLikeNest funcPassManager(modulePassManager);
  funcPassManager.addPass(createFoldTensorExtractOpPass)
      .addPass(createLLVMGPUVectorLoweringPass)
      .addPass(createExpandGPUOpsPass)
      // Barrier elimination before we reach unstructured control flow.
      .addPass(createGpuEliminateBarriers);

  // This pass needs to run before SCF -> CF.
  addLowerAndOptimizeAddressComputationPasses(funcPassManager);

  // Run checks on shared memory usage.
  funcPassManager
      .addPass([&] {
        auto getIndexBitwidth = [](mlir::FunctionOpInterface) { return 64; };
        return createGPUCheckResourceUsagePass(getIndexBitwidth);
      })
      // SCF -> CF
      .addPass(createSCFToControlFlowPass)
      .addPass(createCanonicalizerPass)
      .addPass(createCSEPass)
      // Handle complex operation conversion.
      .addPass(createConvertComplexToStandardPass)
      // Math dialect ops rewrites, approximations, casts.
      .addPass(createMathTransformPass)
      .addPass(memref::createExpandOpsPass)
      .addPass(memref::createFoldMemRefAliasOpsPass)
      .addPass([]() {
        IREEExpandStridedMetadataPassOptions options;
        options.allowSubviewExpansion = true;
        return createIREEExpandStridedMetadataPass(options);
      })
      .addPass(createEmulateNarrowTypePass)
      .addPass(affine::createAffineExpandIndexOpsPass)
      .addPass(createLowerAffinePass);

  // Strip out the debug info for the kernel.
  modulePassManager.addPass(createStripDebugInfoPass());
  // Cast address spaces of all function arguments to generic.
  modulePassManager.addPass(createLLVMGPUCastAddressSpaceFunctionPass());
  modulePassManager.addPass(IREE::Util::createDropCompilerHintsPass(
      IREE::Util::DropCompilerHintsPassOptions{/*keepAssumeInt=*/true}));

  if (forROCDL) {
    // convert to ROCDL.
    funcPassManager.addPass(createConvertUnsupportedFloatArithPass);
    modulePassManager.addPass(createConvertToROCDLPass());
    modulePassManager.addNestedPass<LLVM::LLVMFuncOp>(
        createROCDLAnnotateKernelForTranslationPass());
  } else {
    // convert to NVVM.
    modulePassManager.addPass(createConvertToNVVMPass());
  }
}

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
    funcPassManager.addPass(createMaterializeDeviceEncodingPass);
    // TODO(#20160): Combine the EncodingToPaddingPasses with the
    // MaterializeDeviceEncodingPass.
    addEncodingToPaddingPasses(funcPassManager);
    funcPassManager.addPass(createGPUGeneralizeNamedOpsPass);
    funcPassManager.addPass(createROCDLConfigureBufferInstructionsPass);
    addCommonTargetExecutablePreprocessingPasses(funcPassManager);
    // This materializes into 'nop' in the absence of pad encoding layout
    // attributes.
    funcPassManager.addPass(createBlockDynamicDimensionsPass);
    funcPassManager.addPass(createConfigTrackingCanonicalizerPass);
    funcPassManager.addPass(createCSEPass);
  }
  modulePassManager.addPass(createMaterializeTuningSpecsPass());
  modulePassManager.addPass(createMaterializeUserConfigsPass());
  modulePassManager.addPass(createLLVMGPUSelectLoweringStrategyPass());
}

void buildLLVMGPUCodegenConfigurationPassPipeline(
    OpPassManager &variantPassManager) {
  variantPassManager.addPass(createSpecializeExportsPass());
  buildLLVMGPUCodegenConfigurationPassPipelineImpl(
      variantPassManager.nest<ModuleOp>());
}

void buildLLVMGPUCodegenPassPipeline(OpPassManager &variantPassManager,
                                     bool useROCM) {
  // LLVMGPUSelectLoweringStrategyPass may have created ExecutableObjectAttr.
  // Hoisting them now deduplicates them and ensures that rewrite patterns don't
  // need to think about explicitly copying them over to new ops.
  variantPassManager.addPass(IREE::HAL::createHoistExecutableObjectsPass());
  {
    OpPassManager &modulePassManager = variantPassManager.nest<ModuleOp>();
    modulePassManager.addPass(createLowerExecutableUsingTransformDialectPass());
    LLVMGPULowerExecutableTargetPassOptions options;
    options.forROCDL = useROCM;
    FunctionLikeNest(modulePassManager)
        .addPass(
            [&] { return createLLVMGPULowerExecutableTargetPass(options); })
        .addPass(createVerifyWorkgroupDistributionPass);
  }
  {
    ReconcileTranslationInfoPassOptions options;
    options.distributeAlong = clSetWorkgroupDistributionAlong;
    variantPassManager.addPass(createReconcileTranslationInfoPass(options));
  }

  //===--------------------------------------------------------------------===//
  // Convert Linalg ops to LLVM+NVVM/ROCDL ops.
  //
  // Post-conditions:
  //   - All Linalg/Loops/GPU/Affine/Standard ops are converted away.
  //   - The module contains the final llvm.module ready to be serialized.
  //===--------------------------------------------------------------------===//
  addLowerToLLVMGPUPasses(variantPassManager.nest<ModuleOp>(), useROCM);

  LLVM_DEBUG({
    llvm::dbgs() << "Using LLVMGPU pass pipeline:\n";
    variantPassManager.printAsTextualPipeline(llvm::dbgs());
    llvm::dbgs() << "\n";
  });
}

// NOTE: this runs on the top-level program module containing all
// hal.executable ops.
void buildLLVMGPULinkingPassPipeline(OpPassManager &modulePassManager,
                                     std::optional<std::string> target) {
  // Link together executables. This may produce some IR duplication.
  LLVMGPULinkExecutablesPassOptions linkOptions;
  linkOptions.target = target.value_or("");
  modulePassManager.addPass(createLLVMGPULinkExecutablesPass(linkOptions));

  // Cleanup IR duplication.
  modulePassManager.addNestedPass<IREE::HAL::ExecutableOp>(
      mlir::createCanonicalizerPass());

  // Assign final executable constant and import ordinals.
  auto &variantPassManager = modulePassManager.nest<IREE::HAL::ExecutableOp>()
                                 .nest<IREE::HAL::ExecutableVariantOp>();
  variantPassManager.addPass(createLLVMGPUAssignConstantOrdinalsPass());
}

//===----------------------------------------------------------------------===//
// ROCDL Pass Pipelines
//===----------------------------------------------------------------------===//

static void buildROCDLCodegenConfigurationPassPipelineImpl(
    OpPassManager &modulePassManager) {
  {
    FunctionLikeNest funcPassManager(modulePassManager);
    funcPassManager.addPass(createGPUGeneralizeNamedOpsPass);
    funcPassManager.addPass(createROCDLConfigureBufferInstructionsPass);
    addCommonTargetExecutablePreprocessingPasses(funcPassManager);
  }
  modulePassManager.addPass(createMaterializeTuningSpecsPass());
  modulePassManager.addPass(createMaterializeUserConfigsPass());

  modulePassManager.addPass(createROCDLSelectLoweringStrategyPass());
}

void buildROCDLCodegenConfigurationPassPipeline(
    OpPassManager &variantPassManager) {
  variantPassManager.addPass(createSpecializeExportsPass());
  OpPassManager &modulePassManager = variantPassManager.nest<ModuleOp>();
  buildROCDLCodegenConfigurationPassPipelineImpl(modulePassManager);
}

void buildROCDLCodegenPassPipeline(OpPassManager &variantPassManager) {
  {
    OpPassManager &modulePassManager = variantPassManager.nest<ModuleOp>();
    modulePassManager.addPass(createLowerExecutableUsingTransformDialectPass());
    FunctionLikeNest(modulePassManager)
        .addPass(createROCDLLowerExecutableTargetPass)
        .addPass(createVerifyWorkgroupDistributionPass);
  }
  variantPassManager.addPass(createReconcileTranslationInfoPass());
  variantPassManager.addPass(createLowerAffinePass());
  variantPassManager.addPass(IREE::Util::createDropCompilerHintsPass(
      IREE::Util::DropCompilerHintsPassOptions{/*keepAssumeInt=*/true}));

  addLowerToLLVMGPUPasses(variantPassManager.nest<ModuleOp>(),
                          /*forROCDL=*/true);

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

  static PassPipelineRegistration<> LLVMGPULinkingPipeline(
      "iree-codegen-llvmgpu-linking-pipeline",
      "Runs the LLVMGPU HAL executable linking pipeline",
      [](OpPassManager &modulePassManager) {
        buildLLVMGPULinkingPassPipeline(modulePassManager);
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

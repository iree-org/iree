// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"

#include <cstdint>

#include "iree-dialects/Dialect/LinalgTransform/Passes.h"
#include "iree/compiler/Codegen/Common/CombineLayoutTransformation.h"
#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
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
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ComplexToStandard/ComplexToStandard.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/AMDGPU/Transforms/Passes.h"
#include "mlir/Dialect/Affine/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
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

static llvm::cl::opt<bool> clCombineLayoutTransformation(
    "iree-llvmgpu-test-combine-layout-transformation",
    llvm::cl::desc("Combine relayout ops during dispatch configuration"),
    llvm::cl::init(true), llvm::cl::Hidden);

static llvm::cl::opt<bool> clROCDLLoadToTransposeLoad(
    "iree-llvmgpu-test-load-to-transpose-load",
    llvm::cl::desc("Enable amdgpu.transpose_load targeting for ROCDL"),
    llvm::cl::init(true), llvm::cl::Hidden);

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
        llvm::cl::init(IREE::Codegen::WorkgroupId::IdX)

    );

static llvm::cl::opt<bool> clPatchFuncOps(
    "iree-llvmgpu-debug-patch-func-ops",
    llvm::cl::desc(
        "Perform the patches on func ops for debugging purpose. It should be "
        "used with `--iree-codegen-debug-patched-func-ops-file-name`."),
    llvm::cl::init(false), llvm::cl::Hidden);

static llvm::cl::opt<bool> clLLVMGPUEnableSmallFloatEmulation(
    "iree-llvmgpu-enable-small-float-emulation",
    llvm::cl::desc(
        "Enable software emulation for fp4/fp8 types without hardware support. "
        "When disabled (default), unsupported types will cause a compile "
        "error."),
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
    return memref::AllocaOp::create(builder, loc, allocType, dynamicSizes)
        .getResult();
  }

  auto addressSpace = gpu::AddressSpaceAttr::get(
      builder.getContext(), gpu::GPUDialect::getWorkgroupAddressSpace());
  auto allocType =
      MemRefType::get(memRefType.getShape(), memRefType.getElementType(),
                      AffineMap(), addressSpace);
  return memref::AllocOp::create(builder, loc, allocType, dynamicSizes)
      .getResult();
}

// Barriers are only needed when copying to/from workgroup memory. The only
// other kind of memory that can be allocated is function memory, which is local
// to a thread.
static LogicalResult gpuCopyFn(OpBuilder &builder, Location loc, Value from,
                               Value to) {
  bool needsBarrier = false;
  if (hasSharedMemoryAddressSpace(cast<MemRefType>(from.getType()))) {
    needsBarrier = true;
  }
  if (hasSharedMemoryAddressSpace(cast<MemRefType>(to.getType()))) {
    needsBarrier = true;
  }
  if (needsBarrier) {
    // This barrier is only on workgroup memory since (at time of writing) no
    // code writes to global memory in a way that would require global writes to
    // be visible after a barrier for correctness (ex. if a global array was
    // being used to communicate without atomics).
    gpu::BarrierOp::create(builder, loc, gpu::AddressSpace::Workgroup);
  }
  Operation *copy = memref::CopyOp::create(builder, loc, from, to);
  if (needsBarrier) {
    setMarker(copy, getCopyToWorkgroupMemoryMarker());
    gpu::BarrierOp::create(builder, loc, gpu::AddressSpace::Workgroup);
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
  if (target.getBackend() != "rocm") {
    return success();
  }

  // Workgroup reordering on ROCm currently requires all workgrup counts to be
  // static.
  SmallVector<int64_t> workgroupCounts = getStaticNumWorkgroups(funcOp);
  if (llvm::any_of(workgroupCounts, ShapedType::isDynamic)) {
    return failure();
  }

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
                                      bool enableMasking = false,
                                      bool foldIdentitySlices = false) {
  funcPassManager.addPass(createDecomposeConvolutionToLowerDimOpsPass());
  funcPassManager.addPass(IREE::LinalgExt::createDecomposeIm2colPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
  funcPassManager.addPass(
      IREE::VectorExt::createVectorizeIREEVectorExtOpsPass());
  // Vectorize.
  GenericVectorizationPassOptions options;
  options.vectorizeCopies = vectorizeCopies;
  options.enableCleanup = false;
  options.foldCastIntoContract = true;
  options.enableVectorMasking = enableMasking;
  funcPassManager.addPass(createGenericVectorizationPass(options));
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
  // Run subset hoisting to convert iter_args to vectors.
  OptimizeTensorInsertExtractSlicesPassOptions optimizeSlicesOptions;
  optimizeSlicesOptions.foldIdentitySlices = foldIdentitySlices;
  funcPassManager.addPass(
      createOptimizeTensorInsertExtractSlicesPass(optimizeSlicesOptions));
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
  funcPassManager.addPass(createIREECodegenFoldMemRefAliasOpsPass());
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
      !isa<gpu::AddressSpaceAttr, amdgpu::AddressSpaceAttr>(memorySpace)) {
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
    return memref::AllocaOp::create(builder, loc, allocType, dynamicSizes)
        .getResult();
  }
  allocType =
      MemRefType::get(memRefType.getShape(), memRefType.getElementType(),
                      AffineMap(), workgroupSpace);
  return memref::AllocOp::create(builder, loc, allocType, dynamicSizes)
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
    memref::CopyOp::create(builder, loc, from, to);
    return success();
  };
  funcPassManager.addPass(
      createIREEComprehensiveBufferizePass(allocationFn, memcpyFn));

  addIREEPostBufferizationPasses(funcPassManager);

  funcPassManager.addPass(createROCDLBufferInstructionsOptimizationPass());

  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  funcPassManager.addPass(createAMDGPULowerCoalescedDMAToGatherLDSPass());
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
    if (llvm::all_of(op->getUsers(),
                     llvm::IsaPred<IREE::TensorExt::DispatchTensorStoreOp>)) {
      return success();
    }
  }
  return failure();
}

void addGPUTileAndFusePassPipeline(OpPassManager &funcPassManager,
                                   const GPUPipelineOptions &pipelineOptions,
                                   bool forROCDL) {
  funcPassManager.addPass(createGPUPadConvsPass());
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
  funcPassManager.addPass(createLowerTensorUKernelsPass());
  funcPassManager.addPass(createLoweringConfigInterpreterPass());
  funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Step 1. Promote matmul operands and pack to intrinsic shapes.
  funcPassManager.addPass(createGPUPadOperandsPass());
  funcPassManager.addPass(createGPUPromoteMatmulOperandsPass());
  funcPassManager.addPass(createGPUTileAndConvertConvToMatmulPass());
  funcPassManager.addPass(createGPUPackToIntrinsicsPass());
  // Decompose packs and unpacks that are at the function boundary.
  funcPassManager.addPass(createDecomposeBoundaryPackUnPackOpsPass());

  // Step 1.5. Expand result shapes of MultiMmaOps before tiling, and
  // propagate reshapes to the function boundary.
  {
    IREE::GPU::ExpandUndistributedInnerTilesPassOptions options;
    options.expandInputs = false;
    options.expandOutputs = true;
    // Note: options not passed in was previous behavior from PR #18179.
    funcPassManager.addPass(
        IREE::GPU::createExpandUndistributedInnerTilesPass());
  }
  funcPassManager.addPass(createPropagateReshapesByExpansionPass());

  // Step 2. Tile and fuse tileable ops to reduction loops.
  {
    GPUApplyTilingLevelPassOptions options;
    options.tilingLevel = IREE::GPU::TilingLevel::Reduction;
    options.normalizeLoops = pipelineOptions.useIgemmConvolution;
    funcPassManager.addPass(createGPUApplyTilingLevelPass(options));
    funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
    funcPassManager.addPass(createCSEPass());
  }

  // Convert global load DMAs after reduction tiling but before pack
  // decomposition. DecomposePackUnPackOps introduces linalg.transpose which
  // breaks the source tracing in the coalesced DMA conversion.
  funcPassManager.addPass(createGPUConvertToCoalescedDMAPass());

  // Step 3. Decompose pack and unpack ops and propagate the resulting reshapes.
  funcPassManager.addPass(createDecomposePackUnPackOpsPass(
      DecomposePackUnPackOpsPassOptions{/*tileOuterToOne=*/false,
                                        /*useOnlyReshapes=*/true}));

  // Step 3.5. Expand the inner dimensions of MultiMma ops in preparation for
  // distribution to lanes.
  {
    IREE::GPU::ExpandUndistributedInnerTilesPassOptions options;
    options.expandInputs = true;
    options.expandOutputs = false;
    // Note: options not passed in was previous behavior from PR #18179.
    funcPassManager.addPass(
        IREE::GPU::createExpandUndistributedInnerTilesPass());
  }

  funcPassManager.addPass(createPropagateReshapesByExpansionPass());
  funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Step 4. Tile and fuse tileable ops to subgroups/threads.
  {
    GPUApplyTilingLevelPassOptions options;
    options.tilingLevel = IREE::GPU::TilingLevel::Thread;
    options.normalizeLoops = pipelineOptions.useIgemmConvolution;
    // TileAndFuse currently relies on no consumer fusion to order fusion.
    // Disable consumer fusion to maintain this.
    // TODO: Fix this by choosing which consumers to fuse to what.
    options.fuseConsumers = false;
    funcPassManager.addPass(createGPUApplyTilingLevelPass(options));
    funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
    funcPassManager.addPass(createCSEPass());
  }
  {
    GPUApplyTilingLevelPassOptions options;
    options.tilingLevel = IREE::GPU::TilingLevel::Subgroup;
    // TileAndFuse currently relies on no consumer fusion to order fusion.
    // Disable consumer fusion to maintain this.
    // TODO: Fix this by choosing which consumers to fuse to what.
    options.fuseConsumers = false;
    funcPassManager.addPass(createGPUApplyTilingLevelPass(options));
  }
  funcPassManager.addPass(IREE::GPU::createDistributeInnerTiledToLanesPass());

  // Step 4.5. Things that need to happen right after distribution to threads.
  funcPassManager.addPass(createLowerBitcodeUKernelsPass());

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
  CombineResultLayoutTransformationPassOptions combineLayoutOptions;
  combineLayoutOptions.scope =
      IREE::Codegen::RelayoutCombinationScope::Workgroup;
  funcPassManager.addPass(
      createCombineResultLayoutTransformationPass(combineLayoutOptions));
  funcPassManager.addPass(createGPUGreedilyDistributeToThreadsPass());
  funcPassManager.addPass(createTileLargeTensorsPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
  funcPassManager.addPass(createPropagateDispatchSizeBoundsPass());
  funcPassManager.addPass(createIREELoopInvariantCodeMotionPass());
  funcPassManager.addPass(createGPUCombineValueSemanticBarriersPass());

  // Step 6. Lower special ops and vectorize.
  funcPassManager.addPass(
      IREE::LinalgExt::createVectorizeIREELinalgExtOpsPass());
  funcPassManager.addPass(IREE::GPU::createVectorizeIREEGPUOpsPass());
  addGPUVectorizationPasses(funcPassManager, /*vectorizeCopies=*/false,
                            /*enableMasking=*/true,
                            /*foldIdentitySlices=*/true);
  funcPassManager.addPass(createCleanupBufferAllocViewPass());
  funcPassManager.addPass(createGPUCombineValueSemanticBarriersPass());

  // Step 7. Bufferize.
  addGPUBufferizePasses(funcPassManager);

  // Step 8. Resolve remaining parallel loops.
  funcPassManager.addPass(IREE::LinalgExt::createDecomposeMapStorePass());
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
  if (forROCDL && clROCDLLoadToTransposeLoad) {
    funcPassManager.addPass(createROCDLLoadToTransposeLoadPass());
  }

  // Step 9. Remaining post-bufferization optimizations/lowerings.
  funcPassManager.addPass(createFlattenSwizzleHintAllocsPass());
  funcPassManager.addPass(createPropagateDispatchSizeBoundsPass());
  funcPassManager.addPass(IREE::GPU::createLowerIREEGPUOpsPass());
  funcPassManager.addPass(createUnrollAnnotatedLoopsPass());
  funcPassManager.addPass(createIREELoopInvariantCodeMotionPass());
  if (pipelineOptions.enableReduceSharedMemoryBankConflicts) {
    GPUReduceBankConflictsPassOptions options = {};
    options.paddingBits = 64;
    funcPassManager.addPass(createGPUReduceBankConflictsPass(options));
  }
  funcPassManager.addPass(createHoistStaticallyBoundAllocationsPass());
  if (forROCDL && pipelineOptions.prefetchNumStages >= 2) {
    funcPassManager.addPass(createFissionTransferOpsInControlFlowPass());
    funcPassManager.addPass(createRemoveSingleIterationLoopPass());
    ROCDLPrefetchSharedMemoryPassOptions prefetchOpts;
    prefetchOpts.numStages = pipelineOptions.prefetchNumStages;
    funcPassManager.addPass(createROCDLPrefetchSharedMemoryPass(prefetchOpts));
  }

  funcPassManager.addPass(createIREECodegenFoldMemRefAliasOpsPass());
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
  funcPassManager.addPass(createIREECodegenFoldMemRefAliasOpsPass());
  funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
  funcPassManager.addPass(createOptimizeVectorTransferPass());
  funcPassManager.addPass(createOptimizeTensorInsertExtractSlicesPass());
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
  MemRefType fromType = cast<MemRefType>(from.getType());
  if (hasSharedMemoryAddressSpace(fromType)) {
    needsBarrier = true;
  }
  if (hasSharedMemoryAddressSpace(cast<MemRefType>(to.getType()))) {
    needsBarrier = true;
  }
  if (needsBarrier) {
    // See notes in `gpuCopyFn` abut the address space argument here.
    gpu::BarrierOp::create(builder, loc, gpu::AddressSpace::Workgroup);
  }
  VectorType vectorType =
      VectorType::get(fromType.getShape(), fromType.getElementType());
  Value c0 = arith::ConstantIndexOp::create(builder, loc, 0);
  SmallVector<Value> indices(vectorType.getRank(), c0);
  SmallVector<bool> inBounds(vectorType.getRank(), true);
  Value read =
      vector::TransferReadOp::create(builder, loc, vectorType, from, indices,
                                     /*padding=*/std::nullopt, inBounds);
  vector::TransferWriteOp::create(builder, loc, read, to, indices, inBounds);
  if (needsBarrier) {
    gpu::BarrierOp::create(builder, loc, gpu::AddressSpace::Workgroup);
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
                                        bool forROCDL) {
  ReorderWorkgroupsStrategy reorderStrategy =
      getReorderWorkgroupsStrategy(options.reorderStrategy);

  tileAndDistributeToWorkgroup(funcPassManager, /*useForall=*/true,
                               /*convertToDpsOptions=*/std::nullopt,
                               /*reorderStrategy=*/reorderStrategy);

  // Some of the elementwise fusion can benefit from this pass.
  funcPassManager.addPass(createRematerializeParallelOpsPass());

  funcPassManager.addPass(
      IREE::LinalgExt::createConvertAttentionToOnlineAttentionPass());

  funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
  funcPassManager.addPass(createGPUPromoteMatmulOperandsPass());

  funcPassManager.addPass(createGPUExpandDimensionsPass());

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
    GPUApplyPaddingLevelPassOptions padOptions;
    padOptions.tilingLevel = IREE::GPU::TilingLevel::PartialReduction;
    funcPassManager.addPass(createGPUApplyPaddingLevelPass(padOptions));
    GPUApplyTilingLevelPassOptions options;
    options.tilingLevel = IREE::GPU::TilingLevel::PartialReduction;
    options.allowZeroSlices = true;
    funcPassManager.addPass(createGPUApplyTilingLevelPass(options));
    funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
    funcPassManager.addPass(createCSEPass());
    // Post tiling, the tensor.pad multiples can be simplified to static
    // sizes, run dim simplification to infer and propagate these sizes.
    funcPassManager.addPass(memref::createResolveShapedTypeResultDimsPass());
    funcPassManager.addPass(affine::createSimplifyAffineMinMaxPass());
    funcPassManager.addPass(memref::createReifyResultShapesPass());
    funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
    funcPassManager.addPass(createCSEPass());
    funcPassManager.addPass(affine::createLoopCoalescingPass());
    funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
    funcPassManager.addPass(createCSEPass());
  }

  // Tile to serial loops.
  {
    GPUApplyTilingLevelPassOptions options;
    options.tilingLevel = IREE::GPU::TilingLevel::Serial;
    options.allowZeroSlices = true;
    funcPassManager.addPass(createGPUApplyTilingLevelPass(options));
    funcPassManager.addPass(affine::createLoopCoalescingPass());
    funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
    funcPassManager.addPass(createCSEPass());
  }

  funcPassManager.addPass(IREE::LinalgExt::createDecomposeAttentionPass());
  funcPassManager.addPass(createConfigTrackingCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Convert convolutions to matmuls by tiling filter dimensions.
  funcPassManager.addPass(createGPUTileAndConvertConvToMatmulPass());

  // Set anchors at tensor level for vector distribution later and hoist out
  // loop invariant anchors.
  funcPassManager.addPass(createDecomposeHorizontallyFusedGemmsPass());
  funcPassManager.addPass(createLLVMGPUConfigureTensorLayoutsPass());
  funcPassManager.addPass(createIREELoopInvariantCodeMotionPass());

  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
  funcPassManager.addPass(createOptimizeTensorInsertExtractSlicesPass());
  funcPassManager.addPass(tensor::createFoldTensorSubsetOpsPass());

  // Linalg -> Vector
  funcPassManager.addPass(
      IREE::LinalgExt::createVectorizeIREELinalgExtOpsPass());
  addGPUVectorizationPasses(funcPassManager, /*vectorizeCopies=*/true,
                            /*enableMasking=*/true);

  // Allocate tensors for copies to shared memory.
  funcPassManager.addPass(createGPUVectorAllocPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
  funcPassManager.addPass(createGPUCombineValueSemanticBarriersPass());

  // Tensor -> Memref
  addVectorBufferizePasses(funcPassManager);
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
  funcPassManager.addPass(createHoistStaticallyBoundAllocationsPass());

  // Preprocessing for vector distribution.
  funcPassManager.addPass(createLLVMGPUCastTypeToFitMMAPass());

  // Vector SIMD -> Vector SIMT
  funcPassManager.addPass(createLLVMGPUVectorDistributePass());
  funcPassManager.addPass(IREE::LinalgExt::createDecomposeMapStorePass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  if (options.enableReduceSharedMemoryBankConflicts) {
    GPUReduceBankConflictsPassOptions options = {};
    options.paddingBits = 64;
    funcPassManager.addPass(createGPUReduceBankConflictsPass(options));
  }
  if (forROCDL && options.prefetchNumStages >= 2) {
    ROCDLPrefetchSharedMemoryPassOptions prefetchOpts;
    prefetchOpts.numStages = options.prefetchNumStages;
    funcPassManager.addPass(createROCDLPrefetchSharedMemoryPass(prefetchOpts));
  }
  if (clLLVMGPUEnableSharedMemoryReuse) {
    funcPassManager.addPass(createHoistStaticallyBoundAllocationsPass());
    funcPassManager.addPass(createGPUReuseSharedMemoryAllocsPass());
  }
  funcPassManager.addPass(createIREECodegenFoldMemRefAliasOpsPass());
  funcPassManager.addPass(createCSEPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
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
    funcPassManager.addPass(createLowerBitcodeUKernelsPass());
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
  funcPassManager
      .addPass(createExtractAddressComputationGPUPass)
      // Lower any remaining vector.transfer_read and vector.transfer_write ops,
      // since some of the following patterns have trouble dealing with their
      // full complexity.
      .addPass(createVectorTransferLoweringPass)
      .addPass(createIREECodegenFoldMemRefAliasOpsPass)
      // Propagate constants close to loads/stores to improve the ability for
      // swizzling to CSE.
      .addPass(createPropagateConstantOffsetsPass)
      // Propagating constants introduces CSE opportunities.
      .addPass(createCSEPass)
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
      .addPass(createIREECodegenAffineExpandIndexOpsPass)
      .addPass(createIREECodegenLowerAffinePass)
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
                                    bool forROCDL, bool preserveDebugInfo) {
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

      .addPass(createCanonicalizerPass)
      .addPass(createCSEPass);

  // Handled tensor constants.
  modulePassManager.addPass(createIREEBufferizeConstantsPass());

  FunctionLikeNest funcPassManager(modulePassManager);
  funcPassManager.addPass(createFoldTensorExtractOpPass)
      .addPass(createExpandGPUOpsPass)
      // Barrier elimination before we reach unstructured control flow.
      .addPass(createGpuEliminateBarriers)
      .addPass(createCanonicalizerPass)
      .addPass(createCSEPass);

  // This pass needs to run before SCF -> CF.
  addLowerAndOptimizeAddressComputationPasses(funcPassManager);
  funcPassManager.addPass(createLLVMGPUVectorLoweringPass)
      .addPass(createCanonicalizerPass)
      .addPass(createCSEPass);

  if (forROCDL) {
    // This pass needs to run after the LLVMGPUVectorLoweringPass.
    funcPassManager.addPass(amdgpu::createAmdgpuMaskedloadToLoadPass);
    // This pass needs to run before the ResolveSwizzleHints pass.
    funcPassManager.addPass(amdgpu::createAmdgpuFoldMemRefOpsPass);
  }

  funcPassManager
      // Run checks on shared memory usage.
      .addPass([&] {
        auto getIndexBitwidth = [](mlir::FunctionOpInterface) { return 64; };
        return createGPUCheckResourceUsagePass(getIndexBitwidth);
      })
      // Handle complex operation conversion.
      .addPass(createConvertComplexToStandardPass)
      // Math dialect ops rewrites, approximations, casts.
      .addPass(createMathTransformPass)
      // SCF -> CF
      .addPass(createSCFToControlFlowPass)
      .addPass(createCanonicalizerPass)
      .addPass(createCSEPass)
      // Hoist allocations back into the entry block, as lowering to CF may have
      // split the block at a point before the allocation.
      .addPass(createHoistStaticallyBoundAllocationsPass)
      .addPass(createIREECodegenFoldMemRefAliasOpsPass)
      .addPass([]() {
        IREEExpandStridedMetadataPassOptions options;
        options.allowSubviewExpansion = true;
        return createIREEExpandStridedMetadataPass(options);
      })
      .addPass([&forROCDL]() {
        return forROCDL ? createAMDGPUEmulateNarrowTypePass()
                        : createEmulateNarrowTypePass();
      })
      .addPass(createIREECodegenAffineExpandIndexOpsPass)
      .addPass(createIREECodegenLowerAffinePass);

  if (!preserveDebugInfo) {
    modulePassManager.addPass(createStripDebugInfoPass());
  }
  // Cast address spaces of all function arguments to generic.
  modulePassManager.addPass(createLLVMGPUCastAddressSpaceFunctionPass());
  modulePassManager.addPass(IREE::Util::createDropCompilerHintsPass(
      IREE::Util::DropCompilerHintsPassOptions{/*keepAssumeInt=*/true}));

  if (forROCDL) {
    // convert to ROCDL.
    // Software emulation for small float types (fp4/fp8) is controlled by
    // --iree-llvmgpu-enable-small-float-emulation. When disabled (default),
    // ConvertToROCDL will error on unsupported types.
    funcPassManager.addPass([] {
      return createConvertUnsupportedFloatArithPass(
          ConvertUnsupportedFloatArithPassOptions{
              clLLVMGPUEnableSmallFloatEmulation});
    });
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

static void buildLLVMGPUCodegenCommonConfigurationPassPipelineImpl(
    OpPassManager &modulePassManager) {
  {
    FunctionLikeNest funcPassManager(modulePassManager);
    funcPassManager.addPass(createMaterializeDeviceEncodingPass);
    funcPassManager.addPass(createGPUGeneralizeNamedOpsPass);
    funcPassManager.addPass(createROCDLConfigureBufferInstructionsPass);
    addCommonTargetExecutablePreprocessingPasses(funcPassManager);
    if (clCombineLayoutTransformation) {
      funcPassManager.addPass(createBufferizeDispatchTensorLoadStorePass);
      funcPassManager.addPass(createGPUCombineLayoutTransformationPass);
      // GPUCombineLayoutTransformationPass specializes transpose ops, so they
      // need to be generalized again.
      // TODO(Max191): Re-generalize in the GPUCombineLayoutTransformationPass,
      // and remove the extra GPUGeneralizeNamedOpsPass invocation.
      funcPassManager.addPass(createGPUGeneralizeNamedOpsPass);
    }
    // This materializes into 'nop' in the absence of pad encoding layout
    // attributes.
    funcPassManager.addPass(createBlockDynamicDimensionsPass);
    funcPassManager.addPass(createConfigTrackingCanonicalizerPass);
    funcPassManager.addPass(createCSEPass);
  }
}

void buildLLVMGPUCodegenCommonConfigurationPassPipeline(
    OpPassManager &variantPassManager) {
  buildLLVMGPUCodegenCommonConfigurationPassPipelineImpl(
      variantPassManager.nest<ModuleOp>());
}

static void buildLLVMGPUCodegenConfigurationPassPipelineImpl(
    OpPassManager &modulePassManager) {
  buildLLVMGPUCodegenCommonConfigurationPassPipelineImpl(modulePassManager);
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
                                     bool useROCM, bool preserveDebugInfo) {
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
        .addPass(createVerifyWorkgroupDistributionPass)
        .addPass(createRemoveIndexHintsPass);
    if (clPatchFuncOps) {
      modulePassManager.addPass(createPatchFuncOpsPass());
    }
  }
  {
    ReconcileTranslationInfoPassOptions options;
    options.distributeAlong = clSetWorkgroupDistributionAlong;
    variantPassManager.addPass(createReconcileTranslationInfoPass(options));
    variantPassManager.addPass(createResolveWorkgroupCountHintsPass());
  }

  //===--------------------------------------------------------------------===//
  // Convert Linalg ops to LLVM+NVVM/ROCDL ops.
  //
  // Post-conditions:
  //   - All Linalg/Loops/GPU/Affine/Standard ops are converted away.
  //   - The module contains the final llvm.module ready to be serialized.
  //===--------------------------------------------------------------------===//
  addLowerToLLVMGPUPasses(variantPassManager.nest<ModuleOp>(), useROCM,
                          preserveDebugInfo);

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

void buildROCDLCodegenPassPipeline(OpPassManager &variantPassManager,
                                   bool preserveDebugInfo) {
  {
    OpPassManager &modulePassManager = variantPassManager.nest<ModuleOp>();
    modulePassManager.addPass(createLowerExecutableUsingTransformDialectPass());
    FunctionLikeNest(modulePassManager)
        .addPass(createROCDLLowerExecutableTargetPass)
        .addPass(createVerifyWorkgroupDistributionPass);
  }
  variantPassManager.addPass(createReconcileTranslationInfoPass());
  variantPassManager.addPass(createResolveWorkgroupCountHintsPass());
  variantPassManager.addPass(createIREECodegenLowerAffinePass());
  variantPassManager.addPass(IREE::Util::createDropCompilerHintsPass(
      IREE::Util::DropCompilerHintsPassOptions{/*keepAssumeInt=*/true}));

  addLowerToLLVMGPUPasses(variantPassManager.nest<ModuleOp>(),
                          /*forROCDL=*/true, preserveDebugInfo);

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

  struct LLVMGPUPipelineOptions final
      : PassPipelineOptions<LLVMGPUPipelineOptions> {
    Option<bool> preserveDebugInfo{
        *this, "preserve-debug-info",
        llvm::cl::desc("Preserve debug information (do not strip)")};
  };

  static PassPipelineRegistration<> LLVMGPUConfigPipeline(
      "iree-codegen-llvmgpu-configuration-pipeline",
      "Runs the translation strategy configuration pipeline on Linalg for GPU "
      "on all functions in a module",
      [](OpPassManager &modulePassManager) {
        buildLLVMGPUCodegenConfigurationPassPipelineImpl(modulePassManager);
      });

  static PassPipelineRegistration<LLVMGPUPipelineOptions> LinalgNVVMPipeline(
      "iree-codegen-linalg-to-nvvm-pipeline",
      "Runs the progressive lowering pipeline from Linalg to NVVM",
      [](OpPassManager &passManager, const LLVMGPUPipelineOptions &options) {
        buildLLVMGPUCodegenPassPipeline(passManager, false,
                                        options.preserveDebugInfo);
      });

  static PassPipelineRegistration<LLVMGPUPipelineOptions> LinalgROCDLPipeline(
      "iree-codegen-linalg-to-rocdl-pipeline",
      "Runs the progressive lowering pipeline from Linalg to ROCDL",
      [](OpPassManager &passManager, const LLVMGPUPipelineOptions &options) {
        buildLLVMGPUCodegenPassPipeline(passManager, true,
                                        options.preserveDebugInfo);
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

  struct ROCDLPipelineOptions final
      : PassPipelineOptions<ROCDLPipelineOptions> {
    Option<bool> preserveDebugInfo{
        *this, "preserve-debug-info",
        llvm::cl::desc("Preserve debug information (do not strip)")};
  };

  static PassPipelineRegistration<ROCDLPipelineOptions> LinalgROCDLPipeline(
      "iree-codegen-linalg-to-rocdl-pipeline2",
      "Runs pass pipeline to progressively lower Linalg to ROCDL",
      [](OpPassManager &passManager, const ROCDLPipelineOptions &options) {
        buildROCDLCodegenPassPipeline(passManager, options.preserveDebugInfo);
      });

  static PassPipelineRegistration<> LLVMGPUBufferizePipeline(
      "iree-codegen-llvmgpu-bufferization-pipeline",
      "Runs pass pipeline to bufferize for llvmgpu backends",
      [](OpPassManager &passManager) { addBufferizePasses(passManager); });

  static PassPipelineRegistration<ROCDLPipelineOptions>
      LowerToROCMLLVMGPUPasses(
          "iree-codegen-lower-to-rocm-gpu",
          "Runs pass pipeline to progressively lower Linalg to ROCDL",
          [](OpPassManager &passManager, const ROCDLPipelineOptions &options) {
            addLowerToLLVMGPUPasses(passManager, /*forROCDL=*/true,
                                    options.preserveDebugInfo);
          });
}

} // namespace mlir::iree_compiler

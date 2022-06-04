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

#include "iree/compiler/Codegen/Passes.h"

#include "iree-dialects/Dialect/LinalgExt/Passes/Passes.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Arithmetic/Transforms/Passes.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "iree-spirv-lowering-pass-pipeline"

namespace mlir {
namespace iree_compiler {

static Value allocateWorkgroupMemory(OpBuilder &builder, Location loc,
                                     ArrayRef<int64_t> staticShape,
                                     Type elementType,
                                     ArrayRef<Value> dynamicSizes) {
  auto storageClass = SPIRVTypeConverter::getMemorySpaceForStorageClass(
      spirv::StorageClass::Workgroup);
  MemRefType allocType =
      MemRefType::get(staticShape, elementType, {}, storageClass);
  return builder.create<memref::AllocOp>(loc, allocType, dynamicSizes);
}

static Value allocateFunctionMemory(OpBuilder &builder, Location loc,
                                    ArrayRef<int64_t> staticShape,
                                    Type elementType,
                                    ArrayRef<Value> dynamicSizes) {
  auto storageClass = SPIRVTypeConverter::getMemorySpaceForStorageClass(
      spirv::StorageClass::Function);
  MemRefType allocType =
      MemRefType::get(staticShape, elementType, {}, storageClass);
  return builder.create<memref::AllocaOp>(loc, allocType, dynamicSizes);
}

//===----------------------------------------------------------------------===//
// Common Pass Recipes
//===----------------------------------------------------------------------===//

static void addTileAndDistributeToWorkgroupsPasses(
    OpPassManager &passManager, bool useFuseTensorPadWithConsumerPass = false) {
  passManager.addPass(createInsertDistributionInfoPass());
  auto &nestedModulePM = passManager.nest<ModuleOp>();
  nestedModulePM.addNestedPass<func::FuncOp>(
      createTileAndDistributeToWorkgroupsPass());
  if (useFuseTensorPadWithConsumerPass) {
    nestedModulePM.addNestedPass<func::FuncOp>(
        createSPIRVFuseTensorPadWithConsumerPass());
  }
  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addPass(createCSEPass());
}

static void addSPIRVBufferizePasses(OpPassManager &passManager,
                                    WorkgroupMemoryAllocationFn allocationFn) {
  // Resolve dim ops first so that we don't have compute Linalg ops lingering on
  // becuase of dim op usage. This avoids bufferizing those compute ops just for
  // their shape dimensions.
  passManager.addPass(memref::createResolveShapedTypeResultDimsPass());
  passManager.addNestedPass<func::FuncOp>(
      createLinalgBufferizePass(allocationFn));
  // Distribute immediately after bufferization to avoid losing attribute
  // annotations in subsequent transformations. This is a bit fragile right now
  // but we expect upstream for loops to eventually recognize distribution as a
  // first-class attribute then we don't need this.
  passManager.addNestedPass<func::FuncOp>(createSPIRVDistributePass());
  passManager.addPass(memref::createResolveShapedTypeResultDimsPass());
  passManager.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  passManager.addNestedPass<func::FuncOp>(createCSEPass());
  passManager.addNestedPass<func::FuncOp>(createCleanupBufferAllocViewPass());
}

/// Adds passes to materialize structured ops as loops. This replaces structured
/// ops with loop nests containing payloads, so it should be invoked after
/// tiling and vectorization and before buffer transformations.
static void addLoopMaterializationPasses(OpPassManager &pm) {
  pm.addNestedPass<func::FuncOp>(IREE::LinalgExt::createLinalgExtToLoopsPass());
  pm.addNestedPass<func::FuncOp>(createMemrefCopyToLinalgPass());
  pm.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());
  pm.addNestedPass<func::FuncOp>(createRemoveSingleIterationLoopPass());
}

/// Adds passes to lowering MemRefs. This folds MemRef subviews, flattens n-D
/// MemRef into 1-D ones, vectorizes load/store when possible, and performs
/// cross loop nest optimizations. This should be invoked after structured op
/// lowering and before final SPIR-V conversion.
static void addMemRefLoweringPasses(OpPassManager &pm) {
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // math dialect elementry functions -> polynomial form.
  pm.addNestedPass<func::FuncOp>(createPolynomialApproximationPass());

  pm.addNestedPass<func::FuncOp>(createPadDynamicAlloc());

  // Fold load/store from/to subview ops into the original memref when possible.
  // In SPIR-V we don't use memref descriptor so it's not possible to handle
  // subview ops.
  pm.addPass(memref::createFoldSubViewOpsPass());
  pm.addNestedPass<func::FuncOp>(arith::createArithmeticExpandOpsPass());
  pm.addNestedPass<func::FuncOp>(memref::createExpandOpsPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Turn scalar load/store from memrefs into vectorized ones if possible. This
  // gives better memory access patterns, which is very important for perf.
  pm.addPass(createSPIRVVectorizeLoadStore());
  // Perform various vector-level cross-op optimizations like load-store
  // forwarding, shape casting and casting op cancelling.
  pm.addNestedPass<func::FuncOp>(createOptimizeVectorTransferPass());

  // Perform optimizations that need to across the scf.for region boundary.
  pm.addNestedPass<func::FuncOp>(createForOpCanonicalizationPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Turn multi-dimension memref into one-dimension. This is needed for SPIR-V
  // because we don't use upstream memref descriptors.
  pm.addPass(createFlattenMemRefSubspanPass());
}

/// Adds passes to perform the final SPIR-V conversion.
static void addSPIRVLoweringPasses(OpPassManager &pm) {
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addPass(createLowerAffinePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addPass(createConvertToSPIRVPass());

  OpPassManager &spirvPM = pm.nest<spirv::ModuleOp>();
  spirvPM.addPass(spirv::createUnifyAliasedResourcePass());
  spirvPM.addPass(spirv::createLowerABIAttributesPass());
  spirvPM.addPass(createCanonicalizerPass());
  spirvPM.addPass(createCSEPass());
  spirvPM.addPass(spirv::createCanonicalizeGLSLPass());
  spirvPM.addPass(spirv::createUpdateVersionCapabilityExtensionPass());
}

//===----------------------------------------------------------------------===//
// Pass Pipelines
//===----------------------------------------------------------------------===//

void addSPIRVTileAndVectorizePassPipeline(OpPassManager &pm) {
  addTileAndDistributeToWorkgroupsPasses(
      pm, /*useFuseTensorPadWithConsumerPass=*/true);

  auto &nestedModulePM = pm.nest<ModuleOp>();
  nestedModulePM.addNestedPass<func::FuncOp>(
      createFoldAffineMinInDistributedLoopsPass());
  nestedModulePM.addPass(memref::createResolveShapedTypeResultDimsPass());

  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addPass(createCSEPass());

  // Tile to GPU invocations and vectorize.
  nestedModulePM.addNestedPass<func::FuncOp>(
      createSPIRVCreateFastSlowPathPass());
  nestedModulePM.addNestedPass<func::FuncOp>(createSPIRVTilePass());
  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addPass(createCSEPass());
  nestedModulePM.addNestedPass<func::FuncOp>(createSPIRVVectorizePass());
  nestedModulePM.addNestedPass<func::FuncOp>(createForOpCanonicalizationPass());
  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addPass(createCSEPass());

  // Bufferize and distribute.
  addSPIRVBufferizePasses(nestedModulePM, allocateFunctionMemory);

  // Generate loop nests for all remaining ops and remove trivial loops.
  addLoopMaterializationPasses(nestedModulePM);

  // Perform various vector-level cross-op optimizations like load-store
  // forwarding, shape casting and casting op cancelling.
  nestedModulePM.addNestedPass<func::FuncOp>(
      createOptimizeVectorTransferPass());
}

void addSPIRVTileAndVectorizeToCooperativeOpsPassPipeline(OpPassManager &pm) {
  addTileAndDistributeToWorkgroupsPasses(pm);

  auto &nestedModulePM = pm.nest<ModuleOp>();

  addLinalgBufferizePasses(nestedModulePM, allocateWorkgroupMemory);

  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addPass(createCSEPass());

  // Tile and distribute to GPU subgroups and vectorize.
  nestedModulePM.addNestedPass<func::FuncOp>(
      createSPIRVTileAndVectorizeToCooperativeOpsPass());
  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addPass(createCSEPass());

  // Perform various vector-level cross-op optimizations like load-store
  // forwarding, shape casting and casting op cancelling.
  nestedModulePM.addNestedPass<func::FuncOp>(
      createOptimizeVectorTransferPass());

  // Fold subview ops is reqiured for converting vector transfer ops into SPIR-V
  // cooperative ops in the next step.
  nestedModulePM.addPass(memref::createFoldSubViewOpsPass());

  nestedModulePM.addNestedPass<func::FuncOp>(
      createSPIRVVectorToCooperativeOpsPass());
}

void addSPIRVTileAndVectorizeWithWorkgroupMemoryPassPipeline(
    OpPassManager &pm) {
  addTileAndDistributeToWorkgroupsPasses(pm);

  auto &nestedModulePM = pm.nest<ModuleOp>();
  addLinalgBufferizePasses(nestedModulePM, allocateWorkgroupMemory);

  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addPass(createCSEPass());

  // Tile and distribute to GPU invocations.
  nestedModulePM.addNestedPass<func::FuncOp>(createSPIRVTileAndPromotePass());
  nestedModulePM.addNestedPass<func::FuncOp>(createMemrefCopyToLinalgPass());
  nestedModulePM.addNestedPass<func::FuncOp>(
      createGPUDistributeSharedMemoryCopy());
  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addPass(createCSEPass());

  nestedModulePM.addNestedPass<func::FuncOp>(
      createRemoveSingleIterationLoopPass());

  nestedModulePM.addNestedPass<func::FuncOp>(createSPIRVVectorizePass());
  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addPass(createCSEPass());
  nestedModulePM.addNestedPass<func::FuncOp>(
      createOptimizeVectorTransferPass());

  // nestedModulePM.addNestedPass<func::FuncOp>(createGPUPipeliningPass());

  addLoopMaterializationPasses(nestedModulePM);
}

void addSPIRVTileAndDistributePassPipeline(OpPassManager &pm) {
  addTileAndDistributeToWorkgroupsPasses(pm);

  auto &nestedModulePM = pm.nest<ModuleOp>();

  addLinalgBufferizePasses(nestedModulePM, allocateWorkgroupMemory);

  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addPass(createCSEPass());

  // Tile and distribute to GPU invocations.
  nestedModulePM.addNestedPass<func::FuncOp>(
      createSPIRVTileAndDistributePass());
  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addPass(createCSEPass());

  addLoopMaterializationPasses(nestedModulePM);
}

//===----------------------------------------------------------------------===//
// Entry Point
//===----------------------------------------------------------------------===//

void buildSPIRVCodegenPassPipeline(OpPassManager &pm) {
  pm.nest<ModuleOp>().nest<func::FuncOp>().addPass(createTypePropagationPass());
  pm.nest<ModuleOp>().addPass(createBufferizeCopyOnlyDispatchesPass());
  pm.addPass(createSPIRVLowerExecutableTargetPass());

  addMemRefLoweringPasses(pm.nest<ModuleOp>());
  addSPIRVLoweringPasses(pm.nest<ModuleOp>());

  LLVM_DEBUG({
    llvm::dbgs() << "Using SPIR-V pass pipeline:\n";
    pm.printAsTextualPipeline(llvm::dbgs());
    llvm::dbgs() << "\n";
  });
}

}  // namespace iree_compiler
}  // namespace mlir

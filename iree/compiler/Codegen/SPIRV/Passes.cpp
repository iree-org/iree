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

#include "iree-dialects/Dialect/LinalgExt/Transforms/Passes.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/SPIRV/MemorySpace.h"
#include "llvm/Support/Debug.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Arithmetic/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "iree-spirv-lowering-pass-pipeline"

namespace mlir {
namespace iree_compiler {

static Value gpuAllocationFunction(OpBuilder &builder, Location loc,
                                   ArrayRef<int64_t> staticShape,
                                   Type elementType,
                                   ArrayRef<Value> dynamicSizes) {
  MemRefType allocType =
      MemRefType::get(staticShape, elementType, {}, getWorkgroupMemorySpace());
  return builder.create<memref::AllocOp>(loc, allocType, dynamicSizes);
}

//===----------------------------------------------------------------------===//
// Common Pass Recipes
//===----------------------------------------------------------------------===//

void addSPIRVBufferizePasses(OpPassManager &passManager,
                             WorkgroupMemoryAllocationFn allocationFn) {
  // Resolve dim ops first so that we don't have compute Linalg ops lingering on
  // becuase of dim op usage. This avoids bufferizing those compute ops just for
  // their shape dimensions.
  passManager.addPass(memref::createResolveShapedTypeResultDimsPass());
  passManager.addNestedPass<FuncOp>(createLinalgBufferizePass(allocationFn));
  // Distribute immediately after bufferization to avoid losing attribute
  // annotations in subsequent transformations. This is a bit fragile right now
  // but we expect upstream for loops to eventually recognize distribution as a
  // first-class attribute then we don't need this.
  passManager.addNestedPass<FuncOp>(createSPIRVDistributePass());
  passManager.addPass(memref::createResolveShapedTypeResultDimsPass());
  passManager.addNestedPass<FuncOp>(createCanonicalizerPass());
  passManager.addNestedPass<FuncOp>(createCSEPass());
  passManager.addNestedPass<FuncOp>(createCleanupBufferAllocViewPass());
}

/// Adds passes to materialize structured ops as loops. This replaces structured
/// ops with loop nests containing payloads, so it should be invoked after
/// tiling and vectorization and before buffer transformations.
static void addLoopMaterializationPasses(OpPassManager &pm) {
  pm.addNestedPass<FuncOp>(IREE::LinalgExt::createLinalgExtToLoopsPass());
  pm.addNestedPass<FuncOp>(createConvertLinalgToLoopsPass());
  pm.addNestedPass<FuncOp>(createRemoveSingleIterationLoopPass());
}

/// Adds passes to lowering MemRefs. This folds MemRef subviews, flattens n-D
/// MemRef into 1-D ones, vectorizes load/store when possible, and performs
/// cross loop nest optimizations. This should be invoked after structured op
/// lowering and before final SPIR-V conversion.
static void addMemRefLoweringPasses(OpPassManager &pm) {
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Fold load/store from/to subview ops into the original memref when possible.
  // In SPIR-V we don't use memref descriptor so it's not possible to handle
  // subview ops.
  pm.addPass(memref::createFoldSubViewOpsPass());
  pm.addNestedPass<FuncOp>(arith::createArithmeticExpandOpsPass());
  pm.addNestedPass<FuncOp>(createStdExpandOpsPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Turn scalar load/store from memrefs into vectorized ones if possible. This
  // gives better memory access patterns, which is very important for perf.
  pm.addPass(createSPIRVVectorizeLoadStore());

  // Perform optimizations that need to across the scf.for region boundary.
  pm.addNestedPass<FuncOp>(createForOpCanonicalizationPass());
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

  pm.addPass(createConvertToSPIRVPass());

  OpPassManager &spirvPM = pm.nest<spirv::ModuleOp>();
  spirvPM.addPass(spirv::createLowerABIAttributesPass());
  spirvPM.addPass(createCanonicalizerPass());
  spirvPM.addPass(createCSEPass());
  spirvPM.addPass(spirv::createUpdateVersionCapabilityExtensionPass());
}

//===----------------------------------------------------------------------===//
// Pass Pipelines
//===----------------------------------------------------------------------===//

void addSPIRVTileAndVectorizePassPipeline(OpPassManager &pm) {
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Tile to GPU invocations and vectorize.
  pm.addNestedPass<FuncOp>(createSPIRVTilePass());
  pm.addNestedPass<FuncOp>(createSPIRVVectorizePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Bufferize and distribute.
  addSPIRVBufferizePasses(pm, gpuAllocationFunction);

  // Generate loop nests for all remaining ops and remove trivial loops.
  addLoopMaterializationPasses(pm);

  // Perform various vector-level cross-op optimizations like load-store
  // forwarding, shape casting and casting op cancelling.
  pm.addNestedPass<FuncOp>(createOptimizeVectorTransferPass());
}

void addSPIRVTileAndVectorizeToCooperativeOpsPassPipeline(OpPassManager &pm) {
  addLinalgBufferizePasses(pm, gpuAllocationFunction);

  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Tile and distribute to GPU subgroups and vectorize.
  pm.addNestedPass<FuncOp>(createSPIRVTileAndVectorizeToCooperativeOpsPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Perform various vector-level cross-op optimizations like load-store
  // forwarding, shape casting and casting op cancelling.
  pm.addNestedPass<FuncOp>(createOptimizeVectorTransferPass());

  // Fold subview ops is reqiured for converting vector transfer ops into SPIR-V
  // cooperative ops in the next step.
  pm.addPass(memref::createFoldSubViewOpsPass());

  pm.addNestedPass<FuncOp>(createSPIRVVectorToCooperativeOpsPass());
}

void addSPIRVTileAndDistributePassPipeline(OpPassManager &pm) {
  addLinalgBufferizePasses(pm, gpuAllocationFunction);

  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Tile and distribute to GPU invocations.
  pm.addNestedPass<FuncOp>(createSPIRVTileAndDistributePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  addLoopMaterializationPasses(pm);
}

// An ad-hoc pipeline for tiling and distributing padding/copy ops. This is
// needed to migrate from a bufferization-first world to a vectorization-first
// world.
//
// In the former path for CodeGen, we perform bufferization first, which will
// turn padding/copy (via flow.dispatch.tensor.load/store pairs) into
// linalg.copy ops. Then we deduce CodeGen configuration from the linalg.copy op
// and use a `lowering.config` attribute on it to drive transformations.
//
// In the latter path for CodeGen, we will see linalg.pad_tensor directly.
// However, properly tiling and distributing it is an ongoing work. So for now
// still perform bufferization first to expose a linalg.copy op, from which we
// can deduce the configuration.
void addSPIRVTileAndDistributeCopyPassPipeline(OpPassManager &pm) {
  addLinalgBufferizePasses(pm.nest<ModuleOp>(), gpuAllocationFunction);

  // Rerun CodeGen configuration deduction after bufferization. This enables
  // us to find a better configuration for linalg.copy ops and attach the
  // `lowering.config` attribute properly to drive transformations.
  pm.addPass(createSPIRVInitConfigPass());
  pm.addPass(createSetNumWorkgroupsPass());

  OpPassManager &modulePM = pm.nest<ModuleOp>();

  modulePM.addPass(createCanonicalizerPass());
  modulePM.addPass(createCSEPass());

  // Tile and distribute to GPU invocations.
  modulePM.addNestedPass<FuncOp>(createSPIRVTileAndDistributePass());
  modulePM.addPass(createCanonicalizerPass());
  modulePM.addPass(createCSEPass());

  addLoopMaterializationPasses(modulePM);
}

//===----------------------------------------------------------------------===//
// Entry Point
//===----------------------------------------------------------------------===//

void buildSPIRVCodegenPassPipeline(OpPassManager &pm) {
  pm.nest<ModuleOp>().nest<FuncOp>().addPass(createTypePropagationPass());
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

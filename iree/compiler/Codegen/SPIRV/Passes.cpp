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
#include "iree/compiler/Dialect/Shape/Transforms/Passes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
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

/// Adds passes to materialize structured ops as loops. This replaces structured
/// ops with loop nests containing payloads, so it should be invoked after
/// tiling and vectorization and before buffer transformations.
static void addLoopMaterializationPasses(OpPassManager &pm) {
  pm.addNestedPass<FuncOp>(IREE::LinalgExt::createLinalgExtToLoopsPass());
  pm.addNestedPass<FuncOp>(createConvertLinalgToLoopsPass());
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
  pm.addNestedPass<FuncOp>(Shape::createFoldDimOverShapeCarryingOpPass());
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

void addSPIRVTileAndVectorizePassPipeline(OpPassManager &pm) {
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Tile and distribute to GPU invocations and vectorize.
  pm.addNestedPass<FuncOp>(createSPIRVTileAndDistributePass());
  pm.addNestedPass<FuncOp>(createRemoveSingleIterationLoopPass());
  pm.addNestedPass<FuncOp>(createSPIRVVectorizePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Perform various vector-level cross-op optimizations like load-store
  // forwarding, shape casting and casting op cancelling.
  pm.addNestedPass<FuncOp>(createOptimizeVectorTransferPass());

  addLoopMaterializationPasses(pm);
}

void addSPIRVTileAndVectorizeToCooperativeOpsPassPipeline(OpPassManager &pm) {
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
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Tile and distribute to GPU invocations.
  pm.addNestedPass<FuncOp>(createSPIRVTileAndDistributePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  addLoopMaterializationPasses(pm);
}

void buildSPIRVCodegenPassPipeline(OpPassManager &pm) {
  addLinalgBufferizePasses(pm.nest<ModuleOp>(), gpuAllocationFunction);
  pm.addPass(createSPIRVLowerExecutableTargetPass());
  addMemRefLoweringPasses(pm.nest<ModuleOp>());
  addSPIRVLoweringPasses(pm.nest<ModuleOp>());

  LLVM_DEBUG({
    llvm::dbgs() << "Using SPIRV Pass pipeline:\n";
    pm.printAsTextualPipeline(llvm::dbgs());
    llvm::dbgs() << "\n";
  });
}

}  // namespace iree_compiler
}  // namespace mlir

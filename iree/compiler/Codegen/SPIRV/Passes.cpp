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

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/SPIRV/MemorySpace.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"
#include "iree/compiler/Dialect/Shape/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/GPUToSPIRV/GPUToSPIRV.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"
#include "mlir/Conversion/StandardToSPIRV/StandardToSPIRV.h"
#include "mlir/Conversion/StandardToSPIRV/StandardToSPIRVPass.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/FoldUtils.h"
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

void addSPIRVTileAndVectorizePassPipeline(OpPassManager &pm) {
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addNestedPass<FuncOp>(createSPIRVRemoveOneTripTiledLoopPass());
  //  Tile and distribute to GPU subgroups/invocations and vectorize.
  pm.addNestedPass<FuncOp>(createSPIRVTileAndDistributePass());
  pm.addNestedPass<FuncOp>(createSPIRVVectorizePass());
  pm.addPass(createCanonicalizerPass());

  pm.addNestedPass<FuncOp>(createSPIRVCopyToWorkgroupMemoryPass());
  pm.addNestedPass<FuncOp>(linalg_ext::createLinalgExtToLoopsPass());
  pm.addNestedPass<FuncOp>(createConvertLinalgToLoopsPass());
  pm.addPass(createLowerAffinePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Perform various vector-level cross-op optimizations like load-store
  // forwarding, shape casting and casting op cancelling.
  pm.addNestedPass<FuncOp>(createOptimizeVectorTransferPass());
}

void addSPIRVTileAndDistributePassPipeline(OpPassManager &pm) {
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Tile and distribute to GPU subgroups/invocations.
  pm.addNestedPass<FuncOp>(createSPIRVTileAndDistributePass());
  pm.addPass(createCanonicalizerPass());

  pm.addNestedPass<FuncOp>(createSPIRVCopyToWorkgroupMemoryPass());
  pm.addNestedPass<FuncOp>(linalg_ext::createLinalgExtToLoopsPass());
  pm.addNestedPass<FuncOp>(createConvertLinalgToLoopsPass());
  pm.addPass(createLowerAffinePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Perform various vector-level cross-op optimizations like load-store
  // forwarding, shape casting and casting op cancelling.
  pm.addNestedPass<FuncOp>(createOptimizeVectorTransferPass());
}

void addSPIRVDistributeToGlobalIDPassPipeline(OpPassManager &pm) {
  pm.addNestedPass<FuncOp>(createSPIRVDistributeToGlobalIDPass());
  pm.addPass(createLowerAffinePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Perform various vector-level cross-op optimizations like load-store
  // forwarding, shape casting and casting op cancelling.
  pm.addNestedPass<FuncOp>(createOptimizeVectorTransferPass());
}

static void addLowerToSPIRVPasses(OpPassManager &pm) {
  // Fold load/store from/to subview ops into the original memref when possible.
  // In SPIR-V we don't use memref descriptor so it's not possible to handle
  // subview ops.
  pm.addPass(memref::createFoldSubViewOpsPass());
  pm.addNestedPass<FuncOp>(Shape::createFoldDimOverShapeCarryingOpPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Turn scalar load/store from memrefs into vectorized ones if possible. This
  // gives better memory access patterns, which is very important for perf.
  pm.addPass(createSPIRVVectorizeLoadStore());
  // Lower vector ops to SPIR-V cooperative matrix ops. This needs to be done
  // before flattening memref because we still need the multi-dimension
  // structure.
  pm.addNestedPass<FuncOp>(createSPIRVVectorToCooperativeMatrixPass());

  // Perform optimizations that need to across the scf.for region boundary.
  pm.addNestedPass<FuncOp>(createForOpCanonicalizationPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Turn multi-dimension memref into one-dimension. This is needed for SPIR-V
  // because we don't use upstream memref descriptors.
  pm.addPass(createFlattenMemRefSubspanPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createLowerAffinePass());

  // Finally convert everything to SPIR-V.
  pm.addPass(createConvertToSPIRVPass());

  OpPassManager &spirvModulePM = pm.nest<spirv::ModuleOp>();
  spirvModulePM.addPass(spirv::createLowerABIAttributesPass());
  spirvModulePM.addPass(createCanonicalizerPass());
  spirvModulePM.addPass(createCSEPass());
  spirvModulePM.addPass(spirv::createUpdateVersionCapabilityExtensionPass());
}

void buildSPIRVCodegenPassPipeline(OpPassManager &pm) {
  {
    OpPassManager &nestedModulePM = pm.nest<ModuleOp>();
    addLinalgBufferizePasses(nestedModulePM, gpuAllocationFunction);
  }
  pm.addPass(createSPIRVLowerExecutableTargetPass());
  OpPassManager &nestedModulePM = pm.nest<ModuleOp>();
  addLowerToSPIRVPasses(nestedModulePM);

  LLVM_DEBUG({
    llvm::dbgs() << "Using SPIRV Pass pipeline :\n";
    pm.printAsTextualPipeline(llvm::dbgs());
    llvm::dbgs() << "\n";
  });
}

}  // namespace iree_compiler
}  // namespace mlir

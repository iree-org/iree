// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Modules/VMVX/Transforms/Passes.h"

#include <memory>

#include "iree/compiler/Conversion/Common/Passes.h"
#include "iree/compiler/Conversion/LinalgToLLVM/Passes.h"
#include "iree/compiler/Conversion/Passes.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "iree/compiler/Dialect/Shape/Transforms/Passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace VMVX {

// NOTE:
// NOTE:    THIS IS ALL JUST A HACK
// NOTE:
// NOTE:    this entire pipeline needs to be reworked - it's been randomly
// NOTE:    constructed to "work" for a few samples by someone who does not
// NOTE:    understand the codegen system :)
// NOTE:

static void buildVectorVMVXTransformPassPipeline(OpPassManager &passManager) {
  OpPassManager &nestedModulePM = passManager.nest<ModuleOp>();

  // ---------------------------------------------------------------------------
  // Configuration
  // ---------------------------------------------------------------------------

  // TODO(#5925): This can also be modified to just use the dynamic pass
  // pipeline like the CPU side.
  // passManager.addPass(createMaterializeCPULaunchConfigurationPass());
  passManager.addPass(createSetNumWorkgroupsPass());

  // ---------------------------------------------------------------------------
  // Linalg -> Vectors
  // ---------------------------------------------------------------------------

  nestedModulePM.addNestedPass<FuncOp>(createLinalgVectorizePass());

  // Use stack allocation for transient buffers.
  WorkgroupMemoryAllocationFn allocationFn =
      [](OpBuilder &builder, Location loc, ArrayRef<int64_t> staticShape,
         Type elementType, ArrayRef<Value> dynamicSizes) {
        MemRefType allocType = MemRefType::get(staticShape, elementType);
        return builder.create<memref::AllocaOp>(loc, allocType, dynamicSizes);
      };
  addLinalgBufferizePasses(nestedModulePM, allocationFn);
  nestedModulePM.addPass(createPromoteBuffersToStackPass(
      /*maxAllocSizeInBytes=*/1 << 10, /*bitwidthOfIndexType=*/32,
      /*maxRankOfAllocatedMemRef=*/10));

  nestedModulePM.addNestedPass<FuncOp>(createResolveShapeOpsPass());
  nestedModulePM.addNestedPass<FuncOp>(
      Shape::createCleanupShapePlaceholdersPass());

  // Tiling and distribution.
  nestedModulePM.addNestedPass<FuncOp>(createCanonicalizerPass());
  // TODO(#5925): This can also be modified to just use the dynamic pass
  // pipeline like the CPU side.
  // nestedModulePM.addNestedPass<FuncOp>(
  //     createLinalgTileAndVectorizeWorkgroupsPass());

  // Linalg -> SCF.
  nestedModulePM.addNestedPass<FuncOp>(createConvertLinalgToLoopsPass());
  nestedModulePM.addNestedPass<FuncOp>(createCanonicalizerPass());
  nestedModulePM.addNestedPass<FuncOp>(createCSEPass());
  nestedModulePM.addNestedPass<FuncOp>(createConvertVectorToSCFPass());
  nestedModulePM.addNestedPass<FuncOp>(createCanonicalizerPass());

  // Handle tensor-type constants.
  nestedModulePM.addPass(createTensorConstantBufferizePass());
  nestedModulePM.addPass(createFoldTensorExtractOpPass());

  // Flatten and cleanup memrefs.
  nestedModulePM.addNestedPass<FuncOp>(memref::createFoldSubViewOpsPass());
  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addPass(createCSEPass());
  nestedModulePM.addPass(createFlattenMemRefSubspanPass());
  nestedModulePM.addPass(createNormalizeMemRefsPass());
  nestedModulePM.addNestedPass<FuncOp>(createMemRefDataFlowOptPass());
}

static void buildLoopOptimizationVMVXTransformPassPipeline(
    OpPassManager &passManager) {
  OpPassManager &nestedModulePM = passManager.nest<ModuleOp>();

  nestedModulePM.addNestedPass<FuncOp>(createLowerAffinePass());
  nestedModulePM.addNestedPass<FuncOp>(createForOpCanonicalizationPass());
  nestedModulePM.addNestedPass<FuncOp>(createLoopInvariantCodeMotionPass());
}

void buildVMVXTransformPassPipeline(OpPassManager &passManager) {
  // ---------------------------------------------------------------------------
  // Linalg -> Scalars/Vectors
  // ---------------------------------------------------------------------------

  buildVectorVMVXTransformPassPipeline(passManager);
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());

  // ---------------------------------------------------------------------------
  // Standard/Vector/HAL/etc -> VMVX conversion
  // ---------------------------------------------------------------------------

  passManager.addPass(createConversionPass());
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());

  // ---------------------------------------------------------------------------
  // Cleanup and canonicalization
  // ---------------------------------------------------------------------------

  buildLoopOptimizationVMVXTransformPassPipeline(passManager);
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());
}

void createVMVXTransformPassPipeline() {
  PassPipelineRegistration<> transformPassPipeline(
      "iree-vmvx-transformation-pipeline",
      "Runs the full IREE VMVX dialect transformation pipeline",
      [](OpPassManager &passManager) {
        buildVMVXTransformPassPipeline(passManager);
      });
}

}  // namespace VMVX
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

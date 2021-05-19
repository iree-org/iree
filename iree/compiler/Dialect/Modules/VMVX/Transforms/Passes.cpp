// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/compiler/Dialect/Modules/VMVX/Transforms/Passes.h"

#include <memory>

#include "iree/compiler/Conversion/Common/Passes.h"
#include "iree/compiler/Conversion/HLOToLinalg/Passes.h"
#include "iree/compiler/Conversion/LinalgToLLVM/Passes.h"
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

  passManager.addPass(createMaterializeCPULaunchConfigurationPass());

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

// Copyright 2020 Google LLC
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

#include "iree/compiler/Conversion/Common/Passes.h"

#include "iree/compiler/Conversion/HLOToHLO/Passes.h"
#include "iree/compiler/Conversion/LinalgToLLVM/Passes.h"
#include "iree/compiler/Dialect/Shape/Transforms/Passes.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {

void addLinalgToLLVMPasses(OpPassManager &passManager,
                           LLVMCodegenOptions options) {
  OpPassManager &nestedModulePM = passManager.nest<ModuleOp>();

  // Tile and vectorize linalg ops.
  nestedModulePM.addNestedPass<FuncOp>(createCanonicalizerPass());
  nestedModulePM.addNestedPass<FuncOp>(
      createLinalgTileAndVectorizeWorkgroupsPass());
  nestedModulePM.addNestedPass<FuncOp>(createCanonicalizerPass());
  nestedModulePM.addNestedPass<FuncOp>(createForOpCanonicalizationPass());

  nestedModulePM.addNestedPass<FuncOp>(createPlanConvLoopOrderPass());

  // Linalg -> SCF
  nestedModulePM.addNestedPass<FuncOp>(createConvertLinalgToLoopsPass());
  nestedModulePM.addNestedPass<FuncOp>(createCanonicalizerPass());
  nestedModulePM.addNestedPass<FuncOp>(createCSEPass());

  // SCF -> STD
  nestedModulePM.addNestedPass<FuncOp>(createLowerToCFGPass());
  nestedModulePM.addNestedPass<FuncOp>(createCanonicalizerPass());
  nestedModulePM.addNestedPass<FuncOp>(createCSEPass());

  // Handled tensor-type constants.
  nestedModulePM.addPass(createTensorConstantBufferizePass());
  nestedModulePM.addPass(createFoldTensorExtractOpPass());

  // (HAL, IREE, Linalg, STD) -> LLVM
  nestedModulePM.addPass(createConvertToLLVMPass(options));

  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addPass(createCSEPass());
}

void buildLLVMTransformPassPipeline(OpPassManager &passManager,
                                    LLVMCodegenOptions options) {
  passManager.addPass(createMaterializeCPULaunchConfigurationPass());
  OpPassManager &nestedModulePM = passManager.nest<ModuleOp>();
  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addNestedPass<FuncOp>(createPadLinalgWorkgroupTilesPass());
  // TODO(ataei): We want to enable when tensor -> vector pass is fully
  // supported which requires first moving vector-tiling before this step.
  if (options.useLinalgOnTensorsToVectors) {
    nestedModulePM.addNestedPass<FuncOp>(createLinalgVectorizePass());
  }
  // Use stack allocation on CPU side.
  WorkgroupMemoryAllocationFn allocationFn =
      [](OpBuilder &builder, Location loc, ArrayRef<int64_t> staticShape,
         Type elementType, ArrayRef<Value> dynamicSizes) {
        MemRefType allocType = MemRefType::get(staticShape, elementType);
        return builder.create<memref::AllocaOp>(loc, allocType, dynamicSizes);
      };
  addLinalgBufferizePasses(nestedModulePM, allocationFn);
  nestedModulePM.addPass(createPromoteBuffersToStackPass(1 << 10, 64, 10));

  // Linalg -> LLVM passes.
  addLinalgToLLVMPasses(passManager, options);
}

static PassPipelineRegistration<> linalgLLVMVPipeline(
    "iree-codegen-linalg-to-llvm-pipeline",
    "Runs the progressive lowering pipeline from Linalg to LLVM",
    [](OpPassManager &passManager) {
      buildLLVMTransformPassPipeline(passManager,
                                     getLLVMCodegenOptionsFromClOptions());
    });

static PassPipelineRegistration<> hloToLinalgLLVMVPipeline(
    "iree-codegen-hlo-to-llvm-pipeline",
    "Runs the progressive lowering pipeline from XLA HLO to Linalg to LLVM",
    [](OpPassManager &passManager) {
      buildLLVMTransformPassPipeline(passManager,
                                     getLLVMCodegenOptionsFromClOptions());
    });

}  // namespace iree_compiler
}  // namespace mlir

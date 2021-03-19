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

#include "iree/compiler/Conversion/HLOToLinalg/Passes.h"

#include "iree/compiler/Conversion/Common/Attributes.h"
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
  // Distribute linalg op among a 3d grid of parallel threads. Tile each
  // workgroup thread memory then vectorize the linalg op.
  if (options.usingLinalgOnTensors) {
    passManager.addPass(createMaterializeCPULaunchConfigurationPass());
  } else {
    passManager.addPass(createLinalgTileAndDistributePass());
  }

  OpPassManager &nestedModulePM = passManager.nest<ModuleOp>();
  if (options.useConvImg2Col) {
    // linalg::ConvInputNHWCFilterHWCFOp -> (Img2Col packing + matmul).
    // After convolution is tiled and distributed among workgroups its converted
    // before vectorize workgroup workload.
    nestedModulePM.addNestedPass<FuncOp>(
        createConvImg2ColMatmulConversionPass());
  }

  nestedModulePM.addNestedPass<FuncOp>(
      createLinalgTileAndVectorizeWorkgroupsPass());
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
  OpPassManager &nestedModulePM = passManager.nest<ModuleOp>();

  nestedModulePM.addPass(createInlinerPass());

  // HLO -> Linalg on buffers.
  if (options.usingLinalgOnTensors) {
    nestedModulePM.addPass(createLinalgVectorizePass());
    // Use stack allocation on CPU side.
    WorkgroupMemoryAllocationFn allocationFn =
        [](OpBuilder &builder, Location loc, ArrayRef<int64_t> staticShape,
           Type elementType, ArrayRef<Value> dynamicSizes) {
          MemRefType allocType = MemRefType::get(staticShape, elementType);
          return builder.create<memref::AllocaOp>(loc, allocType, dynamicSizes);
        };
    addLinalgBufferizePasses(nestedModulePM, allocationFn);
    nestedModulePM.addPass(createPromoteBuffersToStackPass(1 << 10, 64, 10));
  } else {
    // Propagates dynamic shapes computation on tensors.
    nestedModulePM.addNestedPass<FuncOp>(Shape::createTieDynamicShapesPass());
    nestedModulePM.addNestedPass<FuncOp>(
        Shape::createMaterializeShapeCalculationsPass());
    nestedModulePM.addNestedPass<FuncOp>(
        Shape::createHoistShapeCalculationsPass());
    nestedModulePM.addNestedPass<FuncOp>(createConvert1x1ConvToDotPass());
    nestedModulePM.addNestedPass<FuncOp>(createDecomposeHLOClampPass());
    addHLOToLinalgOnBuffersPasses(nestedModulePM);
  }
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

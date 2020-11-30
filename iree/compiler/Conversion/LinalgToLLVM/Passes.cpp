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
#include "iree/compiler/Conversion/LLVMToLLVM/Passes.h"
#include "iree/compiler/Conversion/LinalgToLLVM/Passes.h"
#include "iree/compiler/Dialect/Shape/Transforms/Passes.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {

static llvm::cl::opt<bool> convImg2ColConversion(
    "iree-codegen-linalg-to-llvm-conv-img2col-conversion",
    llvm::cl::desc("Enable rewriting linalg.conv linalg.generic that does "
                   "img2col buffer packing + "
                   "linag.matmul"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> fastExpConversion(
    "iree-codegen-linalg-to-llvm-fast-exp",
    llvm::cl::desc("If true convert llvm.intr.exp into its range reduced "
                   "polynomial approximation."),
    llvm::cl::init(false));

void addLinalgToLLVMPasses(OpPassManager &passManager) {
  // Distribute linalg op among a 3d grid of parallel threads. Tile each
  // workgroup thread memory then vectorize the linalg op.

  passManager.addPass(createLinalgTileAndDistributePass());
  passManager.addPass(createLegalizeNumWorkgroupsFnPass());

  // Linalg.ConvOp -> (Img2Col packing + matmul).
  // After convolution is tiled and distributed among workgroups its converted
  // before vectorize workgroup workload.
  if (convImg2ColConversion) {
    passManager.addNestedPass<FuncOp>(createConvImg2ColMatmulConversionPass());
  }
  passManager.addNestedPass<FuncOp>(createPlanConvLoopOrderPass());

  passManager.addNestedPass<FuncOp>(
      createLinalgTileAndVectorizeWorkgroupsPass());

  // Linalg -> SCF
  passManager.addNestedPass<FuncOp>(createConvertLinalgToLoopsPass());
  passManager.addNestedPass<FuncOp>(createCanonicalizerPass());
  passManager.addNestedPass<FuncOp>(createCSEPass());

  // SCF -> STD
  passManager.addNestedPass<FuncOp>(createLowerToCFGPass());
  passManager.addNestedPass<FuncOp>(createCanonicalizerPass());
  passManager.addNestedPass<FuncOp>(createCSEPass());

  // (HAL, IREE, Linalg, STD) -> LLVM
  // OpPassManager& llvmPassManager = passManager.nest<ModuleOp>();
  passManager.addPass(createConvertToLLVMPass());
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());

  // Approximate llvm.intr.exp with a 4-th order ploynmial in range[0, ln2].
  if (fastExpConversion) {
    passManager.addPass(createFastExpApproximationConversionPass());
  }
}

void buildLLVMTransformPassPipeline(OpPassManager &passManager) {
  passManager.addPass(createDeclareNumWorkgroupsFnPass());

  passManager.addPass(createInlinerPass());

  // Propagates dynamic shapes computation on tensors.
  passManager.addNestedPass<FuncOp>(Shape::createTieDynamicShapesPass());
  passManager.addNestedPass<FuncOp>(
      Shape::createMaterializeShapeCalculationsPass());
  passManager.addNestedPass<FuncOp>(Shape::createHoistShapeCalculationsPass());

  // HLO -> Linalg on buffers.
  passManager.addNestedPass<FuncOp>(createDecomposeHLOClampPass());
  addHLOToLinalgOnBuffersPasses(passManager);

  // Linalg -> LLVM passes.
  addLinalgToLLVMPasses(passManager);
}

static PassPipelineRegistration<> linalgLLVMVPipeline(
    "iree-codegen-linalg-to-llvm-pipeline",
    "Runs the progressive lowering pipeline from Linalg to LLVM",
    [](OpPassManager &passManager) {
      buildLLVMTransformPassPipeline(passManager);
    });

static PassPipelineRegistration<> hloToLinalgLLVMVPipeline(
    "iree-codegen-hlo-to-llvm-pipeline",
    "Runs the progressive lowering pipeline from XLA HLO to Linalg to LLVM",
    [](OpPassManager &passManager) {
      buildLLVMTransformPassPipeline(passManager);
    });

}  // namespace iree_compiler
}  // namespace mlir

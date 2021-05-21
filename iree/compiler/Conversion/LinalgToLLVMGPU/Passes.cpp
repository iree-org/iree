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

#include "iree/compiler/Conversion/LinalgToLLVMGPU/Passes.h"

#include "iree/compiler/Conversion/Common/Passes.h"
#include "iree/compiler/Dialect/Shape/Transforms/Passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToSPIRV/StandardToSPIRVPass.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {

static void addLinalgToLLVMGPUPasses(OpPassManager &pm, bool useROCM) {
  //===--------------------------------------------------------------------===//
  // Initial clean up.
  //===--------------------------------------------------------------------===//
  pm.addNestedPass<ModuleOp>(createCanonicalizerPass());
  pm.addNestedPass<ModuleOp>(createCSEPass());

  // Distribute linalg onto threads within the workgroup.
  pm.addPass(createTileAndDistributeToThreads());
  pm.addNestedPass<ModuleOp>(createCanonicalizerPass());
  pm.addNestedPass<ModuleOp>(createCSEPass());

  pm.nest<ModuleOp>().addNestedPass<FuncOp>(
      createRemoveSingleIterationLoopPass());

  // Linalg -> vector
  pm.nest<ModuleOp>().addNestedPass<FuncOp>(createVectorizationPass());
  pm.nest<ModuleOp>().addNestedPass<FuncOp>(createCanonicalizerPass());
  pm.nest<ModuleOp>().addNestedPass<FuncOp>(createCSEPass());

  pm.addNestedPass<ModuleOp>(createLowerAffinePass());
  pm.addNestedPass<ModuleOp>(createCanonicalizerPass());
  pm.addNestedPass<ModuleOp>(createCSEPass());

  // TODO: This currently maps to a single thread. We should share Tile and
  // distribute with other GPU backends.
  // Linalg -> SCF
  pm.nest<ModuleOp>().addNestedPass<FuncOp>(createConvertLinalgToLoopsPass());
  pm.nest<ModuleOp>().addNestedPass<FuncOp>(createCanonicalizerPass());
  pm.nest<ModuleOp>().addNestedPass<FuncOp>(createCSEPass());

  // Handled tensor-type constants.
  pm.addNestedPass<ModuleOp>(createTensorConstantBufferizePass());
  pm.addNestedPass<ModuleOp>(createFoldTensorExtractOpPass());

  // SCF -> STD
  pm.nest<ModuleOp>().addNestedPass<FuncOp>(createLowerToCFGPass());
  pm.nest<ModuleOp>().addNestedPass<FuncOp>(createCanonicalizerPass());
  pm.nest<ModuleOp>().addNestedPass<FuncOp>(createCSEPass());

  pm.addNestedPass<ModuleOp>(createLowerAffinePass());

  // Strip out the debug info for the kernel as CUDA driver doesn't diggest PTX
  // debug info well.
  pm.addNestedPass<ModuleOp>(createStripDebugInfoPass());
  if (useROCM) {
    // convert to ROCDL.
    pm.addNestedPass<ModuleOp>(createConvertToROCDLPass());
  } else {
    // convert to NVVM.
    pm.addNestedPass<ModuleOp>(createConvertToNVVMPass());
  }
}

void buildLLVMGPUTransformPassPipeline(OpPassManager &pm, bool useROCM) {
  OpPassManager &nestedModulePM = pm.nest<ModuleOp>();
  nestedModulePM.addPass(createInlinerPass());

  WorkgroupMemoryAllocationFn allocationFn =
      [](OpBuilder &builder, Location loc, ArrayRef<int64_t> staticShape,
         Type elementType, ArrayRef<Value> dynamicSizes) {
        MemRefType allocType = MemRefType::get(staticShape, elementType, {}, 3);
        return builder.create<memref::AllocOp>(loc, allocType, dynamicSizes);
      };
  addLinalgBufferizePasses(nestedModulePM, allocationFn);

  //===--------------------------------------------------------------------===//
  // Convert Linalg ops to LLVM+NVVM/ROCDL ops.
  //
  // Post-conditions:
  //   - All Linalg/Loops/GPU/Affine/Standard ops are converted away.
  //   - The module contains the final llvm.module ready to be serialized.
  //===--------------------------------------------------------------------===//
  addLinalgToLLVMGPUPasses(pm, useROCM);
}

static PassPipelineRegistration<> linalgToNVVMPipeline(
    "iree-codegen-linalg-to-nvvm-pipeline",
    "Runs the progressive lowering pipeline from Linalg to NVVM",
    [](OpPassManager &passManager) {
      addLinalgToLLVMGPUPasses(passManager, false);
    });

static PassPipelineRegistration<> linalgToROCDLPipeline(
    "iree-codegen-linalg-to-rocdl-pipeline",
    "Runs the progressive lowering pipeline from Linalg to ROCDL",
    [](OpPassManager &passManager) {
      addLinalgToLLVMGPUPasses(passManager, true);
    });

static PassPipelineRegistration<> hloToLinalgNVVMPipeline(
    "iree-codegen-hlo-to-nvvm-pipeline",
    "Runs the progressive lowering pipeline from XLA HLO to Linalg to "
    "NVVM",
    [](OpPassManager &passManager) {
      buildLLVMGPUTransformPassPipeline(passManager, false);
    });

static PassPipelineRegistration<> hloToLinalgROCDLPipeline(
    "iree-codegen-hlo-to-rocdl-pipeline",
    "Runs the progressive lowering pipeline from XLA HLO to Linalg to "
    "ROCDL",
    [](OpPassManager &passManager) {
      buildLLVMGPUTransformPassPipeline(passManager, true);
    });

}  // namespace iree_compiler
}  // namespace mlir

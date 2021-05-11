// Copyright 2021 Nod Labs
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

#include "experimental/rocm/Conversion/LinalgToROCDL/Passes.h"

#include "iree/compiler/Conversion/Common/Passes.h"
#include "iree/compiler/Conversion/HLOToHLO/Passes.h"
#include "iree/compiler/Dialect/Shape/Transforms/Passes.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToSPIRV/StandardToSPIRVPass.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {

static void addLinalgToROCDLPasses(OpPassManager &pm) {
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

  // SCF -> STD
  pm.nest<ModuleOp>().addNestedPass<FuncOp>(createLowerToCFGPass());
  pm.nest<ModuleOp>().addNestedPass<FuncOp>(createCanonicalizerPass());
  pm.nest<ModuleOp>().addNestedPass<FuncOp>(createCSEPass());

  // TODO: Is this also necessary for AMD?
  // Strip out the debug info
  pm.addNestedPass<ModuleOp>(createStripDebugInfoPass());
  // convert to ROCDL.
  pm.addNestedPass<ModuleOp>(createConvertToROCDLPass());
}

void buildROCDLTransformPassPipeline(OpPassManager &pm) {
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
  // Convert Linalg ops to LLVM+ROCDL ops.
  //
  // Post-conditions:
  //   - All Linalg/Loops/GPU/Affine/Standard ops are converted away.
  //   - The module contains the final llvm.module ready to be serialized.
  //===--------------------------------------------------------------------===//
  addLinalgToROCDLPasses(pm);
}

void registerLinalgToROCDLPipeline() {
  PassPipelineRegistration<> linalgToROCDLPipeline(
    "iree-codegen-linalg-to-rocdl-pipeline",
    "Runs the progressive lowering pipeline from Linalg to ROCDL",
    [](OpPassManager &passManager) { addLinalgToROCDLPasses(passManager); });
}

void registerHLOToROCDLPipeline() {
  PassPipelineRegistration<> hloToLinalgROCDLPipeline(
    "iree-codegen-hlo-to-rocdl-pipeline",
    "Runs the progressive lowering pipeline from XLA HLO to Linalg to "
    "ROCDL",
    [](OpPassManager &passManager) {
      buildROCDLTransformPassPipeline(passManager);
    });
}

}  // namespace iree_compiler
}  // namespace mlir

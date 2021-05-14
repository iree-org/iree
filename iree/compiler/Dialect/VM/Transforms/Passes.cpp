// Copyright 2019 Google LLC
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

#include "iree/compiler/Dialect/VM/Transforms/Passes.h"

#include <memory>

#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace VM {

void buildVMTransformPassPipeline(OpPassManager &passManager,
                                  TargetOptions targetOptions) {
  passManager.addNestedPass<mlir::FuncOp>(createLoopCoalescingPass());
  passManager.addNestedPass<mlir::FuncOp>(createLoopFusionPass());
  passManager.addNestedPass<mlir::FuncOp>(createLoopInvariantCodeMotionPass());
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());
  passManager.addNestedPass<mlir::FuncOp>(createLowerToCFGPass());
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());

  passManager.addPass(createConversionPass(targetOptions));

  passManager.addNestedPass<VM::ModuleOp>(createHoistInlinedRodataPass());
  passManager.addNestedPass<VM::ModuleOp>(createGlobalInitializationPass());

  passManager.addPass(createInlinerPass());
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());

  passManager.addPass(createSymbolDCEPass());
  if (targetOptions.optimizeForStackSize) {
    passManager.addNestedPass<VM::ModuleOp>(createSinkDefiningOpsPass());
  }
}

void registerVMTransformPassPipeline() {
  PassPipelineRegistration<> transformPassPipeline(
      "iree-vm-transformation-pipeline",
      "Runs the full IREE VM dialect transformation pipeline",
      [](OpPassManager &passManager) {
        buildVMTransformPassPipeline(passManager, getTargetOptionsFromFlags());
      });
}

}  // namespace VM
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

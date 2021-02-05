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

#include "iree/compiler/Bindings/TFLite/Transforms/Passes.h"

#include <memory>

#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace TFLite {

void buildTransformPassPipeline(OpPassManager &passManager) {
  // Wraps the entry points in a "_tflite_xx" function.
  passManager.addPass(createWrapEntryPointsPass());

  // Materialize the functions required by the runtime bindings to manipulate
  // the program state.
  passManager.addPass(createMaterializeShapeSupportPass());

  // Cleanup the IR after manipulating it.
  passManager.addPass(createInlinerPass());
  passManager.addNestedPass<FuncOp>(createCanonicalizerPass());
  passManager.addNestedPass<FuncOp>(createCSEPass());
  passManager.addPass(createSymbolDCEPass());
}

void registerTransformPassPipeline() {
  PassPipelineRegistration<> transformPassPipeline(
      "iree-tflite-transform-pipeline",
      "Runs the TFLite bindings support pipeline",
      [](OpPassManager &passManager) {
        buildTransformPassPipeline(passManager);
      });
}

}  // namespace TFLite
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

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

#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"

#include <memory>

#include "iree/compiler/Dialect/HAL/Conversion/FlowToHAL/ConvertFlowToHAL.h"
#include "iree/compiler/Dialect/Shape/Transforms/Passes.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

void buildHALTransformPassPipeline(OpPassManager &passManager,
                                   ExecutableTargetOptions executableOptions) {
  passManager.addPass(createCanonicalizerPass());

  passManager.addPass(createMaterializeInterfacesPass(executableOptions));
  passManager.addPass(createTranslateExecutablesPass(executableOptions));
  passManager.addPass(createConvertFlowToHALPass());

  // Phase ordering note: Before this pass, functions signatures will be based
  // on explicit shape types (such as ranked_shape). After this pass, these
  // composite types will be expanded to primitives (i.e. one 'index' for each
  // dynamic dim in the case of ranked_shape).
  passManager.addPass(Shape::createExpandFunctionRankedShapeDimsPass());

  // For each exported function, processes the reflection metadata and
  // generates public ABI wrappers for various calling conventions.
  // Phase ordering note: This operates on functions whose signatures have
  // been expanded to primitives.
  passManager.addPass(createPublicABIGenerationPass());

  passManager.addPass(createMaterializeResourceCachesPass(executableOptions));

  passManager.addNestedPass<FuncOp>(createCanonicalizerPass());
  passManager.addNestedPass<FuncOp>(createCSEPass());

  passManager.addPass(createOutlineDeviceSwitchesPass());
  passManager.addPass(createMemoizeDeviceQueriesPass());
  // TODO(benvanik): function deduplication to remove outlined functions.

  // TODO(benvanik): run symbol DCE when all symbols have visibility defined.
  // Right now the global value initializers don't have proper tracking and if
  // we do this we lose initializers that have side effects we care about.
  // passManager.addPass(createSymbolDCEPass());
}

static PassPipelineRegistration<> transformPassPipeline(
    "iree-hal-transformation-pipeline",
    "Runs the full IREE HAL dialect transformation pipeline",
    [](OpPassManager &passManager) {
      buildHALTransformPassPipeline(passManager,
                                    getExecutableTargetOptionsFromFlags());
    });

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

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

#include "iree/compiler/Bindings/SIP/Transforms/Passes.h"

#include <memory>

#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace SIP {

void buildTransformPassPipeline(OpPassManager &passManager) {
  // Run inlining to avoid issues with flatten tuples in cfg and tf.while & co.
  // This should not be required if we are going to linalg in the importer.
  passManager.addPass(createInlinerPass());

  // Currently SIP can't handle tuples; we probably want to make it handle them
  // if we expect them to be in the inputs.
  passManager.addPass(IREE::Flow::createFlattenTuplesInCFGPass());

  // Materialize default arg/result reflection metadata.
  // This pass must come before any 1:N type expansion that will not be retained
  // in the public ABI (i.e. loose shape dims, etc).
  passManager.addNestedPass<FuncOp>(
      IREE::SIP::createMaterializeReflectionAttrsPass());

  // Cleanup the IR after manipulating it.
  passManager.addNestedPass<FuncOp>(createCanonicalizerPass());
  passManager.addNestedPass<FuncOp>(createCSEPass());
  passManager.addPass(createSymbolDCEPass());
}

void registerTransformPassPipeline() {
  PassPipelineRegistration<> transformPassPipeline(
      "iree-sip-transform-pipeline",
      "Runs the SIP-compatible binding support pipeline",
      [](OpPassManager &passManager) {
        buildTransformPassPipeline(passManager);
      });
}

}  // namespace SIP
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

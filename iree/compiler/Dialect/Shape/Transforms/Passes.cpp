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

#include "iree/compiler/Dialect/Shape/Transforms/Passes.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir {
namespace iree_compiler {

void populateMaterializeDynamicShapesPipeline(OpPassManager &pm) {
  // Expands function signatures to accept/return explicit dynamic dimensions.
  pm.addPass(createExpandFunctionDynamicDimsPass());
  // Inserts tie_shape ops for any dynamic tensors in functions.
  pm.addPass(createTieDynamicShapesPass());
  // Materializes shape calculations for any get_ranked_shape ops.
  pm.addPass(createMaterializeShapeCalculationsPass());
}

static mlir::PassPipelineRegistration<> pipeline(
    "iree-shape-materialize-dynamic",
    "Run IREE-specific passes for augmenting dynamic functions and "
    "materializing shapes.",
    populateMaterializeDynamicShapesPipeline);

}  // namespace iree_compiler
}  // namespace mlir

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

#include "iree/compiler/Dialect/VMLA/Transforms/Passes.h"

#include <memory>

#include "iree/compiler/Dialect/IREE/Transforms/Passes.h"
#include "iree/compiler/Dialect/Shape/Transforms/Passes.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/transforms/passes.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace VMLA {

void buildVMLATransformPassPipeline(OpPassManager &passManager) {
  passManager.addPass(createCanonicalizerPass());

  // ---------------------------------------------------------------------------
  // Inline and flatten structured control flow to our CFG.
  // ---------------------------------------------------------------------------
  passManager.addNestedPass<FuncOp>(mhlo::createLegalizeControlFlowPass());

  // Perform inlining and cleanup after CFG manipulation.
  passManager.addPass(createInlinerPass());
  passManager.addPass(createSymbolDCEPass());
  passManager.addNestedPass<FuncOp>(createCanonicalizerPass());
  passManager.addNestedPass<FuncOp>(createCSEPass());

  // ---------------------------------------------------------------------------
  // Tensor-level rewrites.
  // At this point, the computation is in tensor-level CFG form.
  // There are no specific requirements on shape-related calculations at this
  // point yet, so general tensor->tensor transformations in preparation
  // for later conversion steps should go here.
  // ---------------------------------------------------------------------------
  // Legalize input types.
  // TODO(benvanik): legalize input.
  // passManager.addPass(IREE::VMLA::createLegalizeInputTypesPass());

  // TODO(benvanik): preserve these hints during conversion.
  passManager.addNestedPass<FuncOp>(createDropCompilerHintsPass());

  // Unroll multi-dimensional reductions to one reduction per dimension.
  passManager.addNestedPass<FuncOp>(createUnrollReductionsPass());

  // Tensor-level pattern-based lowerings. Thrown into one pass for simplicity.
  passManager.addNestedPass<FuncOp>(createPreConversionLoweringPass());

  // Clean up the IR before going into shape-materialized IR.
  passManager.addNestedPass<FuncOp>(createCanonicalizerPass());

  // ---------------------------------------------------------------------------
  // Shape calculation.
  // Pre-conditions:
  //   - All transformations altering the tensor-level shapes have been done.
  //   - "Root" dynamic tensors all pass through a single shapex.tie_shape
  //     use which associates them to their shape.
  //   - Loose, non-associated shapex.get_ranked_shape ops can exist anywhere
  //     and will be resolved.
  // Post-conditions:
  //   - All dynamic tensors bridge through a shapex.tie_shape op with the
  //     appropriate shape.
  //   - No shapex.get_ranked_shape ops exist.
  //   - Shape folding and canonicalization has been done.
  // ---------------------------------------------------------------------------
  passManager.addNestedPass<FuncOp>(Shape::createTieDynamicShapesPass());
  passManager.addNestedPass<FuncOp>(
      Shape::createMaterializeShapeCalculationsPass());
  passManager.addNestedPass<FuncOp>(Shape::createHoistShapeCalculationsPass());

  // ---------------------------------------------------------------------------
  // VMLA conversion.
  // Performs lowering from tensor-level to VMLA-level ops/types and on to the
  // VM dialect.
  // Pre-conditions:
  //   - All tensors with dynamic dimensions must have a tie_shape use which
  //     associates them with the SSA values providing the missing dims.
  //   - Functions must be in CFG form.
  //   - Any non-trivial tensor-level transformations have already been done.
  //   - No shapex.get_ranked_shape ops can exist (or be introduced).
  // Post-conditions:
  //   - All ops and types have been fully lowered to the VM dialect.
  // ---------------------------------------------------------------------------
  passManager.addNestedPass<FuncOp>(createCSEPass());
  passManager.addPass(createConversionPass());

  // ---------------------------------------------------------------------------
  // Cleanup identity ops that clutter up the IR and canonicalize.
  // ---------------------------------------------------------------------------
  passManager.addNestedPass<FuncOp>(createCSEPass());
  passManager.addNestedPass<FuncOp>(createCanonicalizerPass());

  // TODO(benvanik): run symbol DCE pass.
}

void createVMLATransformPassPipeline() {
  PassPipelineRegistration<> transformPassPipeline(
      "iree-vmla-transformation-pipeline",
      "Runs the full IREE VMLA dialect transformation pipeline",
      [](OpPassManager &passManager) {
        buildVMLATransformPassPipeline(passManager);
      });
}

}  // namespace VMLA
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

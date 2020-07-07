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

#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"

#include <memory>

#include "iree/compiler/Dialect/Shape/Conversion/Passes.h"
#include "iree/compiler/Dialect/Shape/Transforms/Passes.h"
#include "mlir/Dialect/Shape/Transforms/Passes.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/transforms/passes.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

void buildFlowTransformPassPipeline(OpPassManager &passManager) {
  //----------------------------------------------------------------------------
  // Input dialect sanitization and type legalization.
  // On completion:
  //   - All ops remain at the input-dialect tensor-level.
  //   - Loose shapex.get_ranked_shape ops can exist at points where dynamic
  //     dims are required.
  //----------------------------------------------------------------------------
  passManager.addPass(createCanonicalizerPass());

  // Flatten structured control flow to our CFG.
  passManager.addNestedPass<FuncOp>(mhlo::createLegalizeControlFlowPass());
  passManager.addPass(createHLOPreprocessingPass());

  // Run passes to remove shape constraints. HLO lowering inserts them, but they
  // are not desired here.
  //
  // TODO(GH-2277): Lower HLO shape constraints instead of eliding them here.
  passManager.addNestedPass<FuncOp>(createRemoveShapeConstraintsPass());
  passManager.addNestedPass<FuncOp>(createCanonicalizerPass());

  // Convert `shape` dialect to `shapex` dialect.
  passManager.addPass(Shape::createConvertShapeToShapexPass());

  // Flatten tuples (like tuple<tensor<...>, tensor<...>>) so we can do
  // fine-grained tensor tracking.
  passManager.addPass(IREE::Flow::createFlattenTuplesInCFGPass());

  // Perform inlining and cleanup after CFG manipulation.
  passManager.addPass(createInlinerPass());
  passManager.addNestedPass<FuncOp>(createCanonicalizerPass());
  passManager.addNestedPass<FuncOp>(createCSEPass());

  // Legalize input types. We do this after flattening tuples so that we don't
  // have to deal with them.
  passManager.addPass(IREE::Flow::createLegalizeInputTypesPass());

  //----------------------------------------------------------------------------
  // Shape and reflection ABI materialization.
  // Must happen after:
  //   - Type conversion and sanitization of public function signatures
  //   - Conversion to CFG
  // Must happen before:
  //   - Dependencies on shape metadata ops
  //   - Dependencies on reflection attributes
  //----------------------------------------------------------------------------
  // Materialize default arg/result reflection metadata.
  // This pass must come before any 1:N type expansion that will not be retained
  // in the public ABI (i.e. loose shape dims, etc).
  passManager.addPass(IREE::Flow::createMaterializeExportedReflection());

  // Materialize dynamic shapes in the IR, also expanding function signatures
  // such that:
  //   - Dynamic ranked tensors: (tensor<?x?xf32>) expands to
  //     (tensor<?x?xf32>, ranked_shape<[?,?]>), and ultimately expands to
  //     (tensor<?x?xf32>, i32, i32)
  //   - Unranked tensors: TODO
  // The generated ABI wrappers assume such an expansion and will generate code
  // to produce it from the original reflection metadata captured in the
  // previous pass.
  passManager.addPass(Shape::createExpandFunctionDynamicDimsPass());

  // Merge arg/result reflection metadata.
  // NOTE(laurenzo): This will eventually not be the right place for this as
  // it should happen after the HAL has further annotated the exported
  // functions (such as with synthetic barrier arguments).
  passManager.addPass(IREE::Flow::createMergeExportedReflection());

  //----------------------------------------------------------------------------
  // Shape materialization for buffer assignment and stream formation.
  //
  // Phase ordering constraints:
  //   - All tensor-level transformations which alter shapes must be complete
  //     prior to this phase.
  //
  // Pre-conditions:
  //   - "Root" dynamic tensors all pass through a single shapex.tie_shape
  //     use which associates them to their shape.
  //   - Loose, non-associated shapex.get_ranked_shape ops can exist anywhere
  //     and will be resolved.
  // Post-conditions:
  //   - All dynamic tensors bridge through a shapex.tie_shape op with the
  //     appropriate shape.
  //   - No shapex.get_ranked_shape ops exist (they have been converted to
  //     concrete IR which materializes the shapes, either statically or
  //     dynamically).
  //   - Shape folding and canonicalization has been done.
  // TODO(laurenzo): Investigate whether this can be done more incrementally
  // during dispatch/stream formation versus having such a large phase
  // ordering constraint.
  //----------------------------------------------------------------------------
  passManager.addPass(Shape::createTieDynamicShapesPass());
  passManager.addPass(Shape::createMaterializeShapeCalculationsPass());
  passManager.addPass(Shape::createHoistShapeCalculationsPass());

  //----------------------------------------------------------------------------
  // Partitioning and dispatch region formation
  //
  // Phase ordering constraints:
  //   - Must precede dependencies on fully formed flow.dispatch and
  //     flow.dispatch_region ops
  // Pre-conditions:
  //   - Conversion to CFG
  //   - Materialization of shape metadata ops
  // Post-conditions:
  //   - Dispatch functions have been outlined such that only their dynamic
  //     root tensors are tied via shapex.tie_shape
  //   - Non-dispatchable ops have either been converted to flow ops or deemed
  //     legal.
  //   - shapex.tie_shape ops exist at any dispatch operands/results that are
  //     dynamic, preserving the shape association.
  //     TODO(laurenzo): determine if this is needed versus only preserving
  //     for non-dispatchable ops.
  //----------------------------------------------------------------------------
  // Convert into our expected input and (hopefully) some flow ops.
  passManager.addNestedPass<FuncOp>(
      IREE::Flow::createPrePartitioningConversionPass());

  // First perform module-level analysis that following passes will use to query
  // per-function dispatchability information. We run this first so that it only
  // needs to run once and will be cached for all of the following passes.
  passManager.addPass(IREE::Flow::createDispatchabilityAnalysisPass());

  // Create all of the dispatch regions, CSE their workloads, and fold.
  passManager.addPass(IREE::Flow::createIdentifyDispatchRegions2Pass());
  passManager.addNestedPass<FuncOp>(createCSEPass());
  passManager.addPass(IREE::Flow::createFoldCompatibleDispatchRegionsPass());

  // Note that as we are rematerializing things here it's critical we do not run
  // the canonicalizer/CSE between now and when we outline - otherwise it'll
  // undo all of our work!
  passManager.addPass(IREE::Flow::createRematerializeDispatchConstantsPass());

  // Outline the dispatch regions into their own functions. This separates the
  // sequencer functions performing dispatches from the dispatchees.
  passManager.addPass(IREE::Flow::createOutlineDispatchRegionsPass());

  // Cleanup identity ops that clutter up the IR and canonicalize.
  passManager.addNestedPass<FuncOp>(createCanonicalizerPass());

  // Convert any leftover ops outside of dispatch regions to flow ops.
  passManager.addNestedPass<FuncOp>(createPostPartitioningConversionPass());

  // Assign attributes and negotiate each executable's ABI signature.
  // passManager.addPass(IREE::Flow::createAssignExecutableWorkloadsPass());

  //----------------------------------------------------------------------------
  // Stream formation.
  // Pre-conditions:
  //   - Full formation of dispatch regions
  //----------------------------------------------------------------------------
  // Form streams.
  // Cleanup the IR before we try to form streams.
  passManager.addNestedPass<FuncOp>(createCanonicalizerPass());
  passManager.addNestedPass<FuncOp>(createCSEPass());

  // Reorder blocks to increase the grouping of streamable ops.
  passManager.addNestedPass<FuncOp>(createHoistUnstreamableOpsPass());
  // The hoisting pass does some reordering. Canonicalize to avoid unnecessary
  // arbitrary ordering.
  passManager.addNestedPass<FuncOp>(createCanonicalizerPass());

  passManager.addPass(IREE::Flow::createFormStreamsPass());
  // Forming streams involves a fair amount of subgraph stitching, which can
  // cause duplication. Run CSE to collapse.
  passManager.addNestedPass<FuncOp>(createCanonicalizerPass());
  passManager.addNestedPass<FuncOp>(createCSEPass());

  // Symbol DCE any remaining variables/functions that are now no longer
  // required.
  passManager.addPass(createSymbolDCEPass());
}

void registerFlowTransformPassPipeline() {
  PassPipelineRegistration<> transformPassPipeline(
      "iree-flow-transformation-pipeline",
      "Runs the full IREE flow dialect transformation pipeline",
      [](OpPassManager &passManager) {
        buildFlowTransformPassPipeline(passManager);
      });
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

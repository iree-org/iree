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

#include "iree/compiler/Dialect/Shape/Transforms/Passes.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"
#include "tensorflow/compiler/mlir/xla/transforms/passes.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

void buildFlowTransformPassPipeline(OpPassManager &passManager) {
  //----------------------------------------------------------------------------
  // Input dialect sanitization and type legalization.
  //----------------------------------------------------------------------------
  passManager.addPass(createCanonicalizerPass());

  // Flatten structured control flow to our CFG.
  passManager.addNestedPass<FuncOp>(xla_hlo::createLegalizeControlFlowPass());

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
  Shape::populateMaterializeDynamicShapesPipeline(passManager);

  // Merge arg/result reflection metadata.
  // NOTE(laurenzo): This will eventually not be the right place for this as
  // it should happen after the HAL has further annotated the exported
  // functions (such as with synthetic barrier arguments).
  passManager.addPass(IREE::Flow::createMergeExportedReflection());

  // Rewrite some hlo ops to a form that is compatible for all
  // backends.
  passManager.addPass(createHLOPreprocessingPass());

  //----------------------------------------------------------------------------
  // Partitioning and dispatch region formation
  // Must happen after:
  //   - Conversion to CFG
  //   - Materialization of shape metadata ops
  // Must happen before:
  //   - Dependencies on fully formed flow.dispatch and flow.dispatch_region ops
  //   - Dependencies on extra-dispatch-region ops being converted to flow
  //     ops
  //----------------------------------------------------------------------------
  // Convert into our expected input and (hopefully) some flow ops.
  passManager.addNestedPass<FuncOp>(
      IREE::Flow::createPrePartitioningConversionPass());

  // First perform module-level analysis that following passes will use to query
  // per-function dispatchability information. We run this first so that it only
  // needs to run once and will be cached for all of the following passes.
  passManager.addPass(IREE::Flow::createDispatchabilityAnalysisPass());

  // Create all of the dispatch regions, CSE their workloads, and fold.
  passManager.addPass(IREE::Flow::createIdentifyDispatchRegionsPass());
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
  // Must happen after:
  //   - Full formation of dispatch regions
  //----------------------------------------------------------------------------
  // Form streams.
  passManager.addPass(IREE::Flow::createFormStreamsPass());
  // Forming streams involves a fair amount of subgraph stitching, which can
  // cause duplication. Run CSE to collapse.
  passManager.addNestedPass<FuncOp>(createCSEPass());

  // TODO(benvanik): run symbol DCE pass.
}

static PassPipelineRegistration<> transformPassPipeline(
    "iree-flow-transformation-pipeline",
    "Runs the full IREE flow dialect transformation pipeline",
    [](OpPassManager &passManager) {
      buildFlowTransformPassPipeline(passManager);
    });

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

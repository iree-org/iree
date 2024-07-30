// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"

#include <memory>

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "iree/compiler/Utils/PassUtils.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "iree-flow-transforms-passes"

// TODO(ravishankarm): Change to a pipeline option.
static llvm::cl::opt<bool> clExportBenchmarkFuncs(
    "iree-flow-export-benchmark-funcs",
    llvm::cl::desc(
        "Exports one function per original module entry point and "
        "unique flow.executable that dispatches with dummy arguments."),
    llvm::cl::init(false));

// TODO(ravishankarm): Change to a pipeline option.
static llvm::cl::opt<bool> clTraceDispatchTensors(
    "iree-flow-trace-dispatch-tensors",
    llvm::cl::desc(
        "Trace runtime input/output tensors for each dispatch function."),
    llvm::cl::init(false));
static llvm::cl::opt<std::string> clBreakOnDispatch(
    "iree-flow-break-dispatch",
    llvm::cl::desc(
        "Enables inserting a break after a specified dispatch. Supports two "
        "modes; breaking on the dispatch ordinal before deduplication "
        "(@function_name:<index>) and breaking on the dispatch symbol."),
    llvm::cl::init(""));
static llvm::cl::opt<std::string> clTraceDispatch(
    "iree-flow-trace-dispatch",
    llvm::cl::desc("Enables tracing tensors at specified dispatches. Supports "
                   "two modes; tracing the dispatch by ordinal before "
                   "deduplication (@function_name:<index>) and tracing all "
                   "occurrences of the dispatch symbol."),
    llvm::cl::init(""));

static llvm::cl::opt<bool> clDetensoring(
    "iree-flow-enable-detensoring",
    llvm::cl::desc(
        "Enable changing of tensor operations into scalar operations."),
    llvm::cl::init(false));

static llvm::cl::opt<bool> clEnablePadHandling(
    "iree-flow-enable-pad-handling",
    llvm::cl::desc("Enable native handling of tensor.pad operations."),
    llvm::cl::init(false));

static llvm::cl::opt<bool> clEnableFusePaddingIntoLinalgConsumerOps(
    "iree-flow-enable-fuse-padding-into-linalg-consumer-ops",
    llvm::cl::desc("Enable fusing tensor.pad ops into Linalg consumer ops."),
    llvm::cl::init(false));

static llvm::cl::opt<bool> clEnableFusePaddingIntoLinalgProducerOps(
    "iree-flow-enable-fuse-padding-into-linalg-producer-ops",
    llvm::cl::desc("Enable fusing tensor.pad ops into Linalg consumer ops."),
    llvm::cl::init(false));

static llvm::cl::opt<bool> clCollapseReductionDims(
    "iree-flow-collapse-reduction-dims",
    llvm::cl::desc("Enable collapsing of reduction dims"),
    llvm::cl::init(false));

static llvm::cl::opt<bool>
    clEnableFuseMultiUse("iree-flow-fuse-multi-use",
                         llvm::cl::desc("Fuse multi-use ops."),
                         llvm::cl::init(false));

static llvm::cl::opt<bool> clEnableElementWiseFuseMultiReduction(
    "iree-flow-element-wise-fuse-multi-reduction",
    llvm::cl::desc("Enable element-wise fusion of multi-reduction loop ops."),
    llvm::cl::init(true));

static llvm::cl::opt<bool> clEnableAggressiveFusion(
    "iree-flow-enable-aggressive-fusion",
    llvm::cl::desc("Aggressive fusion opportunities that are behind a flag "
                   "since all backends dont support it yet"),
    llvm::cl::init(false));

static llvm::cl::opt<bool>
    clDumpDispatchGraph("iree-flow-dump-dispatch-graph",
                        llvm::cl::desc("Dump a dot graph for dispatches."),
                        llvm::cl::init(false));

static llvm::cl::opt<std::string> clDumpDispatchGraphOutputFile(
    "iree-flow-dump-dispatch-graph-output-file",
    llvm::cl::desc("Output file name for a dispatch graph dump."),
    llvm::cl::init("dispatch.dot"));

static llvm::cl::opt<std::string> clDispatchTransformFileName(
    "iree-flow-dispatch-use-transform-dialect",
    llvm::cl::desc("MLIR file containing a top-level module that specifies "
                   "the transformations to apply to form dispatch regions."),
    llvm::cl::init(""));

static llvm::cl::opt<bool> clZeroFillEmptyTensors(
    "iree-flow-zero-fill-empty-tensors",
    llvm::cl::desc(
        "Zero fill empty tensors instead of leaving them uninitialized."),
    llvm::cl::init(false));

static llvm::cl::opt<bool> clEnableDataTiling(
    "iree-flow-enable-data-tiling",
    llvm::cl::desc("Enable data-tiling at flow level, i.e., it set encodings "
                   "in dispatch regions, hoist them out of region, and enable "
                   "fusion for the set_encodings."),
    llvm::cl::init(false));

static llvm::cl::opt<int> clPadFactor(
    "iree-flow-pad-factor",
    llvm::cl::desc("provides padding size hints that will be attached to "
                   "encodings."),
    llvm::cl::init(32));

namespace mlir::iree_compiler::IREE::Flow {

using FunctionLikeNest =
    MultiOpNest<func::FuncOp, IREE::Util::InitializerOp, IREE::Util::FuncOp>;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

static void addCleanupPatterns(OpPassManager &passManager) {
  FunctionLikeNest(passManager)
      // Standard MLIR cleanup.
      .addPass(IREE::Flow::createCanonicalizerPass)
      .addPass(mlir::createCSEPass)

      // Simplify util.global accesses; this can help with data flow tracking as
      // redundant store-loads are removed.
      .addPass(IREE::Util::createSimplifyGlobalAccessesPass);

  // Cleanup and canonicalization of util.global (and other util ops).
  passManager.addPass(IREE::Util::createApplyPatternsPass());
  passManager.addPass(IREE::Util::createFoldGlobalsPass());
  passManager.addPass(IREE::Util::createFuseGlobalsPass());

  // Large IPO pass. Note that this can introduce a significant amount of
  // duplication/inlined constants and we'll want to ensure we're running
  // cleanup again after (this entire set of patterns is run in a fixed-point
  // iteration to do that).
  passManager.addPass(IREE::Util::createIPOPass());
}

//===----------------------------------------------------------------------===//
// Pipelines
//===----------------------------------------------------------------------===//

void addDispatchRegionCreationPreprocessingPasses(OpPassManager &passManager) {
  // 1. Do some simple elementwise op fusion. This could be skipped,
  //    but could reduce the surface area of ops to handle later.
  FunctionLikeNest(passManager)
      .addPass([]() {
        return IREE::Flow::createElementwiseOpFusionPass(
            ElementwiseOpFusionPassOptions{
                clEnableElementWiseFuseMultiReduction});
      })
      .addPass(IREE::Flow::createCanonicalizerPass)
      .addPass(mlir::createCSEPass)

      // 2. Bubble up expand_shape ops (or sink collapse_shape ops) to get
      //    elementwise operation into higher dimensions for more fusion
      //    opportunities.
      .addPass(IREE::Flow::createBubbleUpExpandShapesPass)
      .addPass(IREE::Flow::createCanonicalizerPass)
      .addPass(mlir::createCSEPass)

      // 3. Perform elementwise operation fusion again (now with higher
      //    dimensionality).
      .addPass([]() {
        return IREE::Flow::createElementwiseOpFusionPass(
            ElementwiseOpFusionPassOptions{
                clEnableElementWiseFuseMultiReduction});
      })
      .addPass(IREE::Flow::createCanonicalizerPass)
      .addPass(mlir::createCSEPass)

      // 4. After elementwise operation fusion sink reshapes that block
      //    producer-consumer fusion.
      .addPass(IREE::Flow::createSinkReshapesPass)
      .addPass(IREE::Flow::createCanonicalizerPass)
      .addPass(mlir::createCSEPass)

      // 5. After all the reshape propagations, fuse elementwise operations
      //    even if the producer has multiple uses.
      .addPass(IREE::Flow::createFuseMultiUseElementwiseProducerPass)

      // 6. Some more "post elementwise fusion passes".
      //    a. Detensorize.
      //       TODO: This is probably not in the right place.
      .addPredicatedPass(clDetensoring,
                         [&]() { return mlir::createLinalgDetensorizePass(); })
      .addPass(IREE::Flow::createCanonicalizerPass)
      .addPass(mlir::createCSEPass)

      //    b. For ops with multiple reduction dimensions, collapse the
      //       reduction dimension.
      //       TODO: This pass is only needed till all backends can handle
      //       multiple reduction dimensions.
      .addPredicatedPass(clCollapseReductionDims,
                         IREE::Flow::createCollapseReductionDimensionsPass)

      //     c. Split reduction operations into parallel and reduction, i.e
      //        .
      .addPass(IREE::Flow::createSplitReductionPass)

      //     d. Transpose generic ops to
      //        - help with dispatch region formation.
      //        - move reduction iterators to be innermost.
      .addPass(IREE::Flow::createTransposeGenericOpsPass);
}

// Pipeline to first create `flow.dispatch.region` ops and then lower to
// `flow.dispatch.workgroup` ops.
static void addDispatchRegionCreationPasses(OpPassManager &passManager) {
  FunctionLikeNest(passManager)
      // Only want use the transform dialect for some dispatch regions and let
      // the FormDispatchRegions handle the rest. This only moves the root
      // compute op into the dispatch region, so that we can run additional
      // transformations afterwards with a simple region and without bothering
      // producers.
      .addPredicatedPass(
          !clDispatchTransformFileName.empty(),
          [&]() {
            DispatchWithTransformDialectPassOptions options;
            options.transformSpecPath = clDispatchTransformFileName;
            return createDispatchWithTransformDialectPass(options);
          })
      // Create dispatches for scalar operations as roots
      .addPass(IREE::Flow::createFormScalarDispatchesPass)
      // Create `flow.dispatch.region` centered around a root and fuse with
      // producers and consumers.
      .addPass([&]() {
        return IREE::Flow::createFormDispatchRegionsPass(
            FormDispatchRegionsPassOptions{
                clEnableAggressiveFusion,
                clEnableFusePaddingIntoLinalgConsumerOps,
                clEnableFusePaddingIntoLinalgProducerOps});
      })
      // Clone all producers into the dispatch region to perpare for being
      // isolated from above. This enables running additional transformations
      // afterwards that would need the full dispatch content but don't want to
      // handle explicit captures as materialized as dispatch workgroup operands
      // and block arguments.
      .addPass(IREE::Flow::createCloneProducersIntoDispatchRegionsPass)
      .addPredicatedPass(clEnableDataTiling,
                         [&]() {
                           return createSetEncodingPass(
                               SetEncodingPassOptions{clPadFactor});
                         })
      // Collapse dimensions of linalg Ops.
      .addPass(IREE::Flow::createCollapseDimensionsPass)
      // Convert dispatch regions into dispatch workgroups by capturing values
      // and making the new workgroups isolated from above.
      .addPass(IREE::Flow::createConvertDispatchRegionsToWorkgroupsPass)
      // Convert tensor operations to flow.tensor ops.
      // - Convert extract/insert slice to flow update ops when the tensor op
      // acts as a contiguous view of the tensor
      // - Apply tensor -> flow patterns
      .addPass(IREE::Flow::createConvertTensorToFlowPass)
      .addPass(IREE::Flow::createCanonicalizerPass)
      /// Creates the workgroup count region where the materialized computation
      /// is derived as a program slice of the body of the dispatch. This method
      /// - Computes the `workload` to use for the `workgroupsOp`, which are
      ///   derived from the values captured by the `workgroupsOp`.
      /// - Populates the workgroup count region for this with the placeholder
      ///   op `flow.dispatch.workgroups_count_from_body_slice`. This op is
      ///   resolved in the backends into the actual workgroup count
      ///   computation.
      /// - To correlate back to the captured workload,
      /// `flow.dispatch.workload.ordinal`
      ///   to map the captured operand to the position in the workload list.
      .addPass(IREE::Flow::createMaterializeDefaultWorkgroupCountRegionPass);
}

// Apply preprocessing and form dispatch regions
void addDispatchRegionCreationPasses(OpPassManager &passManager,
                                     const TransformOptions &transformOptions) {
  FunctionLikeNest(passManager)
      // Preprocess the input to a form more amenable for fusion.
      .addPass(IREE::Flow::createFusionPreprocessingPass)
      .addPass(IREE::Flow::createCanonicalizerPass)
      .addPass(mlir::createCSEPass);

  addDispatchRegionCreationPreprocessingPasses(passManager);
  addDispatchRegionCreationPasses(passManager);
}

void buildFlowTransformPassPipeline(OpPassManager &passManager,
                                    const TransformOptions &transformOptions) {
  // Start of Flow pipeline, verify input legality.
  passManager.addPass(IREE::Flow::createVerifyInputLegalityPass());

  // Inject tensor tracing early as we need to have the tracers in the IR
  // prior to dispatch region formation where we may lose access to them.
  FunctionLikeNest(passManager)
      .addPass(IREE::Flow::createInjectTensorTracingPass);

  // Transform pad operations into linalg.fill + tensor.insert_slice.
  // This is a WAR for not having native pad handling.
  if (!clEnablePadHandling && !clEnableFusePaddingIntoLinalgProducerOps) {
    passManager.addPass(IREE::Flow::createTensorPadToTensorInsertSlicePass(
        TensorPadToTensorInsertSlicePassOptions{
            /*skipSingleLinalgOpUses=*/
            clEnableFusePaddingIntoLinalgConsumerOps}));
  }

  {
    // We run these under a fixed-point iteration such that we can perform
    // inter-procedural, intra-procedural, and canonicalization as separably
    // verifiable/reusable passes. IPO will fold duplicate arguments/results
    // and inline constants to allow the local optimizations to work more
    // effectively.
    OpPassManager ipoPipeline(mlir::ModuleOp::getOperationName());

    // IPO and other cleanups.
    addCleanupPatterns(ipoPipeline);

    // Run fixed-point iteration on the IPO pipeline.
    passManager.addPass(
        IREE::Util::createFixedPointIteratorPass(std::move(ipoPipeline)));
  }

  addDispatchRegionCreationPasses(passManager, transformOptions);

  FunctionLikeNest(passManager)
      .addPass(IREE::Flow::createCaptureDynamicDimsPass)
      .addPass(IREE::Flow::createCanonicalizerPass)
      .addPass(mlir::createCSEPass)
      .addPass([&]() {
        return IREE::Flow::createInitializeEmptyTensorsPass(
            InitializeEmptyTensorsPassOptions{clZeroFillEmptyTensors});
      });

  // Module pass to outline dispatch regions (and similar ops) into their own
  // functions wrapped in executables.
  passManager.addPass(IREE::Flow::createOutlineDispatchExternsPass());
  passManager.addPass(IREE::Flow::createOutlineDispatchRegionsPass());

  // Annotate executables based on their contents.
  // This is optional but can provide useful information during compilation and
  // runtime profiling/tracing.
  passManager.addPass(IREE::Flow::createAnnotateDispatchesPass());

  // Trace/break dispatches by ordinal in the specified region. There is a
  // similar version of the pass run both before and after deduplication
  // depending on if the target is specified by ordinal or by symbol.
  std::string dispatchBreakOrdinalStr =
      !clBreakOnDispatch.empty() && clBreakOnDispatch[0] == '@'
          ? clBreakOnDispatch
          : std::string("");
  std::string dispatchTraceOrdinalStr =
      !clTraceDispatch.empty() && clTraceDispatch[0] == '@' ? clTraceDispatch
                                                            : std::string("");
  if (!dispatchBreakOrdinalStr.empty() || !dispatchTraceOrdinalStr.empty()) {
    passManager.addPass(IREE::Flow::createInsertDebugTargetAtOrdinalPass(
        InsertDebugTargetAtOrdinalPassOptions{dispatchBreakOrdinalStr,
                                              dispatchTraceOrdinalStr}));
  }

  // Strip assertions from executables. We could support them with a bunch of
  // work but our generated executables are designed to be safe in the face of
  // invalid values and it'd only be useful for debugging.
  passManager.addNestedPass<IREE::Flow::ExecutableOp>(
      IREE::Util::createStripDebugOpsPass());

  // Cleanup identity ops that clutter up the IR and canonicalize.
  FunctionLikeNest(passManager).addPass(IREE::Flow::createCanonicalizerPass);

  // Deduplicate executables created from dispatch regions.
  // Note: this only deduplicates equivalent executables. We could in addition
  // generalize executables to prune further (e.g. by promoting a dimension to
  // an argument if two executables differ only in that one dimension).
  passManager.addPass(IREE::Flow::createDeduplicateExecutablesPass());

  // Create one function per exported program entry point that can be used with
  // iree-benchmark-module to benchmark each function individually. Whether
  // a model supports execution like this (handles zero/null args, has state
  // resets, etc) is up to the author.
  if (clExportBenchmarkFuncs) {
    passManager.addPass(IREE::Flow::createExportBenchmarkFuncsPass());
  }

  // Trace/break dispatches by symbol. Symbols are partially matched against
  // the exact string specified in the cli option.
  std::string dispatchBreakSymbolStr =
      !clBreakOnDispatch.empty() && clBreakOnDispatch[0] != '@'
          ? clBreakOnDispatch
          : std::string("");
  std::string dispatchTraceSymbolStr =
      !clTraceDispatch.empty() && clTraceDispatch[0] != '@' ? clTraceDispatch
                                                            : std::string("");
  if (!dispatchBreakSymbolStr.empty() || !dispatchTraceSymbolStr.empty()) {
    passManager.addPass(IREE::Flow::createInsertDebugTargetAtSymbolPass(
        InsertDebugTargetAtSymbolPassOptions{dispatchBreakSymbolStr,
                                             dispatchTraceSymbolStr}));
  }

  FunctionLikeNest(passManager)
      // Inject tracing that logs both input and output tensors from all
      // dispatches. We do this after deduping so that the executable names
      // match later stages.
      .addPredicatedPass(clTraceDispatchTensors,
                         IREE::Flow::createInjectDispatchTracingPass)
      // Inject tensor tracing late for any attributes that were added by the
      // passes above after we've formed dispatch regions.
      .addPass(IREE::Flow::createInjectTensorTracingPass)
      // Cleanup the IR after we are done.
      .addPass(IREE::Flow::createCleanupTensorShapesPass);

  {
    // We run these under a fixed-point iteration such that we can perform
    // inter-procedural, intra-procedural, and canonicalization as separably
    // verifiable/reusable passes. IPO will fold duplicate arguments/results
    // and inline constants to allow the local optimizations to work more
    // effectively.
    OpPassManager ipoPipeline(mlir::ModuleOp::getOperationName());

    // Turn all constant ops into global variables and fix up the IR.
    // As many locations change and constants are deduplicated we'll end up with
    // a lot of extraneous IR (mostly global loads) and clean those up here.
    ipoPipeline.addPass(IREE::Flow::createOutlineConstantsPass());

    // IPO and other cleanups.
    addCleanupPatterns(ipoPipeline);

    // Run fixed-point iteration on the IPO pipeline.
    passManager.addPass(
        IREE::Util::createFixedPointIteratorPass(std::move(ipoPipeline)));
  }

  // Cleanup executable contents.
  {
    auto executablePassManager = passManager.nest<IREE::Flow::ExecutableOp>();
    executablePassManager.addPass(IREE::Flow::createCanonicalizerPass());
    executablePassManager.addPass(mlir::createCSEPass());
  }

  // Symbol DCE any remaining variables/functions that are now no longer
  // required.
  passManager.addPass(mlir::createSymbolDCEPass());

  /// Print the dispatch graph in the Graphviz format.
  if (clDumpDispatchGraph) {
    DumpDispatchGraphPassOptions options;
    options.outputFile = clDumpDispatchGraphOutputFile;
    passManager.addPass(IREE::Flow::createDumpDispatchGraphPass(options));
  }
}

void registerFlowTransformPassPipeline() {
  PassPipelineRegistration<TransformOptions> transformPassPipeline(
      "iree-flow-transformation-pipeline",
      "Runs the full IREE flow dialect transformation pipeline",
      [](OpPassManager &passManager, const TransformOptions &transformOptions) {
        buildFlowTransformPassPipeline(passManager, transformOptions);
      });

  PassPipelineRegistration<> flowDispatchRegionFormationPreprocessingPipeline(
      "iree-flow-dispatch-region-formation-preprocessing-pipeline",
      "Flag used to run preprocessing passes that run passes before dispatch "
      "region formation. Used only for testing",
      [](OpPassManager &passManager) {
        addDispatchRegionCreationPreprocessingPasses(passManager);
      });

  PassPipelineRegistration<> flowDispatchRegionCreationPipeline(
      "iree-flow-dispatch-region-creation-pipeline",
      "Flag used to run passes that form dispatch regions",
      [](OpPassManager &passManager) {
        addDispatchRegionCreationPasses(passManager);
      });
}

namespace {
#define GEN_PASS_REGISTRATION
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h.inc" // IWYU pragma: export
} // namespace

void registerFlowPasses() {
  // Generated.
  registerPasses();

  // Pipelines.
  registerFlowTransformPassPipeline();
}

} // namespace mlir::iree_compiler::IREE::Flow

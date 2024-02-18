// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"

#include <memory>

#include "iree-dialects/Dialect/LinalgExt/Passes/Passes.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "iree/compiler/Utils/PassUtils.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/Passes.h"

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

static llvm::cl::opt<bool> clDispatchGenerateWorkloadRegion(
    "iree-flow-dispatch-generate-workload-region",
    llvm::cl::desc("Generate the workload region."), llvm::cl::init(true));

static llvm::cl::opt<bool> clNormalizeInputIndexingMap(
    "iree-flow-normalize-input-indexing-map",
    llvm::cl::desc("Enable normalizing input indexing map to identity."),
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

static llvm::cl::opt<std::string> clDumpFlowExecutableFunctions(
    "iree-flow-dump-executable-functions-to",
    llvm::cl::desc("directory path to dump individual mlir files for all flow "
                   "executables."),
    llvm::cl::init(""));

namespace mlir::iree_compiler::IREE::Flow {

using FunctionLikeNest =
    MultiOpNest<func::FuncOp, IREE::Util::InitializerOp, IREE::Util::FuncOp>;

void buildFlowTransformPassPipeline(OpPassManager &passManager,
                                    const TransformOptions &transformOptions) {
  // Start of Flow pipeline, verify input legality.
  passManager.addPass(IREE::Flow::createVerifyInputLegalityPass());

  // Transform pad operations into linalg.fill + tensor.insert_slice.
  // This is a WAR for not having native pad handling.
  if (!clEnablePadHandling && !clEnableFusePaddingIntoLinalgProducerOps) {
    passManager.addPass(IREE::Flow::createTensorPadToTensorInsertSlicePass(
        /*skipSingleLinalgOpUses=*/clEnableFusePaddingIntoLinalgConsumerOps));
  }

  FunctionLikeNest(passManager)
      // Preprocess the input to a form more amenable for fusion
      .addPass(createInterchangeGenericOpsPass)
      .addPass(memref::createResolveShapedTypeResultDimsPass)
      .addPass(mlir::createCanonicalizerPass)
      .addPass(mlir::createCSEPass)
      // Elementwise fusion.
      .addPass(
          []() { return createFusionOfTensorOpsPass(clEnableFuseMultiUse); })
      .addPredicatedPass(clDetensoring, mlir::createLinalgDetensorizePass)
      .addPass(mlir::createCanonicalizerPass)
      .addPass(mlir::createCSEPass)
      .addPredicatedPass(clCollapseReductionDims, createCollapseDimsPass)
      // Split reduction operations into parallel and reduction.
      .addPass(createSplitReductionPass)
      // SplitReductionPass may create reduction dimension that are not the last
      // dimension.
      .addPass(createInterchangeGenericOpsPass)
      // Normalize the input indexing map to make the input indexing map
      // identity. This helps fusing named linalg op with a generic op with
      // transpose.
      .addPredicatedPass(clNormalizeInputIndexingMap,
                         createInterchangeTransposeGenericOpsPass)
      ////////////////////////////////////////////////////////////////////////
      // Dispatch region formation.
      .addPredicatedPass(!clDispatchTransformFileName.empty(),
                         [&]() {
                           return createDispatchWithTransformDialect(
                               clDispatchTransformFileName);
                         })

      .addPass(createFormScalarDispatchesPass)
      // Only want use the transform dialect for some dispatch regions and let
      // the FormDispatchRegions handle the rest. This only moves the root
      // compute op into the dispatch region, so that we can run additional
      // transformations afterwards with a simple region and without bothering
      // producers.
      .addPass([&]() {
        return createFormDispatchRegionsPass(FormDispatchRegionsOptions{
            clEnableFuseMultiUse, clDispatchGenerateWorkloadRegion,
            clEnableFusePaddingIntoLinalgConsumerOps,
            clEnableFusePaddingIntoLinalgProducerOps});
      })
      // Clone all producers into the dispatch region to perpare for being
      // isolated from above. This enables running additional transformations
      // afterwards that would need the full dispatch content but don't want to
      // handle explicit captures as materialized as dispatch workgroup operands
      // and block arguments.
      .addPass(createCloneProducersIntoDispatchRegionsPass)
      // Collapse dimensions of linalg Ops.
      .addPass(createCollapseDimensionsPass)
      // Form dispatch region into dispatch workgroups
      .addPass([&]() {
        return createFormDispatchWorkgroupsPass(
            clDispatchGenerateWorkloadRegion);
      })
      ////////////////////////////////////////////////////////////////////////
      .addPass(createCaptureDispatchDynamicDimsPass)
      .addPass(mlir::createCanonicalizerPass)
      .addPass(createCSEPass)

      // Initialize any empty tensors to zero.
      .addPass([&]() {
        return createInitializeEmptyTensorsPass(clZeroFillEmptyTensors);
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
        dispatchBreakOrdinalStr, dispatchTraceOrdinalStr));
  }

  // Strip assertions from executables. We could support them with a bunch of
  // work but our generated executables are designed to be safe in the face of
  // invalid values and it'd only be useful for debugging.
  passManager.addNestedPass<IREE::Flow::ExecutableOp>(
      IREE::Util::createStripDebugOpsPass());

  // Cleanup identity ops that clutter up the IR and canonicalize.
  FunctionLikeNest(passManager).addPass(mlir::createCanonicalizerPass);

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
        dispatchBreakSymbolStr, dispatchTraceSymbolStr));
  }

  FunctionLikeNest(passManager)
      // Inject tracing that logs both input and output tensors from all
      // dispatches. We do this after deduping so that the executable names
      // match later stages.
      .addPredicatedPass(clTraceDispatchTensors,
                         IREE::Flow::createInjectDispatchTracingPass)
      // Cleanup the IR after we are done.
      .addPass(IREE::Flow::createCleanupTensorShapesPass)
      .addPass(mlir::createCanonicalizerPass)
      .addPass(mlir::createCSEPass);

  passManager.addNestedPass<IREE::Flow::ExecutableOp>(
      mlir::createCanonicalizerPass());
  passManager.addNestedPass<IREE::Flow::ExecutableOp>(mlir::createCSEPass());

  // Symbol DCE any remaining variables/functions that are now no longer
  // required.
  passManager.addPass(mlir::createSymbolDCEPass());

  /// Print the dispatch graph in the Graphviz format.
  if (clDumpDispatchGraph) {
    std::string errorMessage;
    static auto dotFile =
        openOutputFile(clDumpDispatchGraphOutputFile, &errorMessage);
    if (!dotFile) {
      llvm::errs() << errorMessage << "\n";
    } else {
      passManager.addPass(
          IREE::Flow::createDumpDispatchGraphPass(dotFile->os()));
      dotFile->keep();
    }
  }

  if (!clDumpFlowExecutableFunctions.empty()) {
    std::string path = clDumpFlowExecutableFunctions;
    passManager.addPass(IREE::Flow::createDumpExecutableFunctionsPass(path));
  }
}

void registerFlowTransformPassPipeline() {
  PassPipelineRegistration<TransformOptions> transformPassPipeline(
      "iree-flow-transformation-pipeline",
      "Runs the full IREE flow dialect transformation pipeline",
      [](OpPassManager &passManager, const TransformOptions &transformOptions) {
        buildFlowTransformPassPipeline(passManager, transformOptions);
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

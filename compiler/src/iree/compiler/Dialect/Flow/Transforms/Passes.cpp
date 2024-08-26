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

static llvm::cl::opt<bool>
    clDumpDispatchGraph("iree-flow-dump-dispatch-graph",
                        llvm::cl::desc("Dump a dot graph for dispatches."),
                        llvm::cl::init(false));

static llvm::cl::opt<std::string> clDumpDispatchGraphOutputFile(
    "iree-flow-dump-dispatch-graph-output-file",
    llvm::cl::desc("Output file name for a dispatch graph dump."),
    llvm::cl::init("dispatch.dot"));

static llvm::cl::opt<bool> clZeroFillEmptyTensors(
    "iree-flow-zero-fill-empty-tensors",
    llvm::cl::desc(
        "Zero fill empty tensors instead of leaving them uninitialized."),
    llvm::cl::init(false));

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

void buildFlowTransformPassPipeline(OpPassManager &passManager,
                                    const TransformOptions &transformOptions) {
  // Start of Flow pipeline, verify input legality.
  passManager.addPass(IREE::Flow::createVerifyInputLegalityPass());

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

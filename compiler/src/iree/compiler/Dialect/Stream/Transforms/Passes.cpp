// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"

#include <memory>

#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "iree/compiler/Utils/PassUtils.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

static llvm::cl::opt<bool> clAnnotateInputAffinities(
    "iree-stream-annotate-input-affinities",
    llvm::cl::desc("Annotates all tensor/resource affinities on the input to "
                   "the pipeline for debugging."),
    llvm::cl::init(false));

// TODO(hanchung): Enable the pass by default once the implementation is done.
static llvm::cl::opt<bool> clSpecializeEncodings(
    "iree-stream-experimental-specialize-encodings",
    llvm::cl::desc(
        "Enables SpecializeEncodingPass in Stream pass pipeline. This pass is "
        "currently under development, so it is not enabled by default. It can "
        "only handle limited cases at this moment."),
    llvm::cl::init(false));

namespace mlir::iree_compiler::IREE::Stream {

using FunctionLikeNest =
    MultiOpNest<func::FuncOp, IREE::Util::InitializerOp, IREE::Util::FuncOp>;

//===----------------------------------------------------------------------===//
// --iree-stream-cleanup-pipeline
//===----------------------------------------------------------------------===//

static void buildStreamCleanupPassPipeline(
    OpPassManager &passManager,
    const IREE::Stream::TransformOptions &transformOptions) {
  FunctionLikeNest(passManager)
      // Standard MLIR cleanup.
      .addPass(mlir::createCanonicalizerPass)
      .addPass(mlir::createCSEPass)

      // Integer optimizations. These operate best on a canonical form both
      // for performance (post-simplifications cause less analysis) and
      // simplified pattern matching.
      .addPass(IREE::Util::createOptimizeIntArithmeticPass)

      // Simplify util.global accesses; this can help with data flow tracking as
      // redundant store-loads are removed.
      .addPass(IREE::Util::createSimplifyGlobalAccessesPass)

      // Aggressive cleanup.
      .addPass(IREE::Util::createApplyPatternsPass);

  // Cleanup and canonicalization of util.global (and other util ops).
  passManager.addPass(IREE::Util::createFoldGlobalsPass());
  passManager.addPass(IREE::Util::createFuseGlobalsPass());

  // Large IPO pass. Note that this can introduce a significant amount of
  // duplication/inlined constants and we'll want to ensure we're running
  // cleanup again after (this entire set of patterns is run in a fixed-point
  // iteration to do that).
  passManager.addPass(IREE::Util::createIPOPass());
}

//===----------------------------------------------------------------------===//
// --iree-stream-tensor-transformation-pipeline
//===----------------------------------------------------------------------===//

void buildStreamTensorPassPipeline(OpPassManager &passManager,
                                   const TransformOptions &transformOptions) {
  //----------------------------------------------------------------------------
  // Input cleanup and simplification
  //----------------------------------------------------------------------------

  // Verify we support the program.
  passManager.addPass(IREE::Stream::createVerifyInputPass());

  // Cleanup the program prior to outlining constants in case there is
  // propagation or fusion that needs to happen first.
  buildStreamCleanupPassPipeline(passManager, transformOptions);

  //----------------------------------------------------------------------------
  // Conversion
  //----------------------------------------------------------------------------

  // Annotate all ops/resources with the analyzed affinities.
  // This should have no behavioral changes during conversion but allows for
  // debugging of analysis errors in end-user tooling.
  if (clAnnotateInputAffinities) {
    passManager.addPass(IREE::Stream::createAnnotateAffinitiesPass());
  }

  // Converts from all input dialects into various levels of the stream dialect.
  // Tensor-like things go to stream.tensor.* ops while lower level buffer-like
  // things will go to stream.async.* ops.
  passManager.addPass(IREE::Stream::createConvertToStreamPass());

  // No more tensor.*/etc ops are allowed. This is conservative - there may be
  // a lot of ops we convert but this will catch the majority of stragglers.
  passManager.addPass(IREE::Stream::createVerifyLoweringToTensorsPass());

  //----------------------------------------------------------------------------
  // Constant/variable optimization
  //----------------------------------------------------------------------------

  // Run inlining after having baked out affinities.
  passManager.addPass(mlir::createInlinerPass());

  // Cleanup globals that were created during conversion.
  buildStreamCleanupPassPipeline(passManager, transformOptions);

  // Bring all initializers together so that we can schedule them.
  passManager.addPass(IREE::Util::createCombineInitializersPass());

  //----------------------------------------------------------------------------
  // Stream affinity/assignment
  //----------------------------------------------------------------------------

  // TODO(benvanik): pin based on target backends here.
  // TODO(benvanik): compute affinities for executables.
  // TODO(benvanik): annotate all dispatches with preferred executable affinity.
  // TODO(benvanik): DFA to specify all value affinities and pin dispatches.

  // TODO(multi-device): it's really nice to be able to verify here but it
  // prevents compiling to stream without devices specified or continuation at
  // various phases. It'd be nice to find a way to enable this when the user
  // expects it to work and otherwise not.
  //
  // Verify that all ops that may require affinities have them assigned or
  // available (on a parent scope, etc). This allows subsequent passes to trust
  // that an affinity lookup will always return a valid affinity.
  // passManager.addPass(IREE::Stream::createVerifyAffinitiesPass());
}

//===----------------------------------------------------------------------===//
// --iree-stream-async-transformation-pipeline
//===----------------------------------------------------------------------===//

void buildStreamAsyncPassPipeline(OpPassManager &passManager,
                                  const TransformOptions &transformOptions) {
  //----------------------------------------------------------------------------
  // Tensor lowering and resource management
  //----------------------------------------------------------------------------

  if (clSpecializeEncodings) {
    passManager.addPass(IREE::Stream::createSpecializeEncodingsPass());
  }

  // Lower stream.tensor.* ops to stream.async.* ops based on
  // affinity/configuration assigned during placement.
  FunctionLikeNest(passManager)
      .addPass(IREE::Stream::createEncodeHostTensorsPass);
  passManager.addNestedPass<IREE::Stream::ExecutableOp>(
      IREE::Stream::createEncodeDeviceTensorsPass());

  buildStreamCleanupPassPipeline(passManager, transformOptions);

  // Everything must now be in stream.async.* form but we don't yet have
  // lifetime assigned.
  passManager.addPass(IREE::Stream::createVerifyLoweringToAsyncResourcesPass());

  // Materialize copy-on-write behavior with explicit stream.async.* ops.
  // This will insert a lot of copies, so follow it up with a pass that elides
  // ones that aren't needed. This is easier to verify than if there was one
  // pass attempting to do both. Note that copy-on-write materialization is
  // required for correct execution while copy elision is for performance only
  // (though it's critical enough that it is not optional).
  FunctionLikeNest(passManager)
      .addPass(IREE::Stream::createMaterializeCopyOnWritePass)
      .addPass(mlir::createCanonicalizerPass);
  passManager.addPass(IREE::Stream::createElideAsyncCopiesPass());
  FunctionLikeNest(passManager)
      .addPass(mlir::createCanonicalizerPass)
      .addPass(IREE::Stream::createEmplaceAllocationsPass);

  // Refine lifetime of all resources across the module.
  // We do this after scheduling execution so that we know how the resources
  // move across devices. We do it before scheduling waves as lifetime doesn't
  // change and it makes the IR cleaner.
  passManager.addPass(IREE::Stream::createRefineUsagePass());

  buildStreamCleanupPassPipeline(passManager, transformOptions);

  // Verify all stream.async.* op access ranges that we can by taking advantage
  // of statically available information or that which we can infer from data
  // flow analysis. Because this may require a global analysis it's best done in
  // a pass instead of individual op verifiers. We could run the pass more
  // frequently above or move some of the simpler checks to op verifiers if we
  // wanted to catch errors earlier but this is mostly a guard before we go into
  // the stream.cmd.* layer.
  passManager.addPass(IREE::Stream::createVerifyAsyncAccessRangesPass());

  //----------------------------------------------------------------------------
  // Stream formation and scheduling
  //----------------------------------------------------------------------------

  FunctionLikeNest(passManager)
      // Combine async work into execution regions.
      .addPass(IREE::Stream::createScheduleExecutionPass)
      // Group concurrently executable work into waves.
      .addPass(IREE::Stream::createScheduleConcurrencyPass);

  // When synchronous initialization is requested we need to separate any work
  // behind a timepoint in the initializer from the consumers of that timepoint.
  if (transformOptions.initializationMode ==
      IREE::Stream::InitializationMode::Synchronous) {
    passManager.addPass(IREE::Stream::createSyncInitializersPass());
  }

  // Materialize timepoints across the entire module. This simplifies scheduling
  // of the timeline as we can shake the IR and see what timepoints we still
  // have left.
  passManager.addPass(IREE::Stream::createPropagateTimepointsPass());

  // Expand builtins to dispatches. This may introduce new executables.
  // We do this after scheduling so that we preserve the semantics of the ops
  // for partitioning/placement before turning them into opaque dispatches.
  passManager.addPass(IREE::Stream::createMaterializeBuiltinsPass());

  buildStreamCleanupPassPipeline(passManager, transformOptions);

  // Everything must now be in stream.async.* form.
  passManager.addPass(IREE::Stream::createVerifyLoweringToAsyncPass());
}

//===----------------------------------------------------------------------===//
// --iree-stream-cmd-transformation-pipeline
//===----------------------------------------------------------------------===//

void buildStreamCmdPassPipeline(OpPassManager &passManager,
                                const TransformOptions &transformOptions) {
  // Schedule fine-grained allocations and insert placeholders for larger/longer
  // lifetime allocations.
  passManager.addPass(IREE::Stream::createScheduleAllocationPass());
  FunctionLikeNest(passManager)
      // TODO(benvanik): passes to convert alloc to alloca and thread through
      // streams. Ideally all transient allocs become stream-ordered allocas.
      // createPropagateTransientsPass()

      // Allocate backing storage for fused constant resources.
      // This expands packed constants into explicit forms with partitioned
      // storage buffers and upload logic.
      .addPass(IREE::Stream::createPackConstantsPass)

      // Layout packed slices to emit the arithmetic required for all resource
      // offsets. This enables us to propagate the subviews across the program
      // below.
      .addPass(IREE::Stream::createLayoutSlicesPass)

      // Apply canonicalization patterns to clean up subview ops prior to
      // propagating subranges.
      .addPass(mlir::createCanonicalizerPass);

  // Propagate subviews throughout the program to unify resource storage access.
  // After propagation many resource SSA values can be deduped or folded by the
  // cleanup patterns.
  passManager.addPass(IREE::Util::createPropagateSubrangesPass());
  buildStreamCleanupPassPipeline(passManager, transformOptions);

  // TODO(benvanik): outline streams (ala dispatch regions). Note that we may
  // want to do this earlier to enable better deduplication but that makes the
  // above passes trickier. Outlining may be more like "find chunks of streams
  // useful to move into secondary command buffers."

  // Everything must now be in explicit stream.cmd.* form.
  passManager.addPass(IREE::Stream::createVerifyLoweringToCmdPass());
}

//===----------------------------------------------------------------------===//
// --iree-stream-optimization-pipeline
//===----------------------------------------------------------------------===//

void buildStreamOptimizationPassPipeline(
    OpPassManager &passManager, const TransformOptions &transformOptions) {
  // Forming streams involves a fair amount of subgraph stitching, which can
  // cause duplication. Run CSE to collapse.
  buildStreamCleanupPassPipeline(passManager, transformOptions);

  // If any scf ops crept in we get rid of them here. We should be able to
  // support them all the way through the stream dialect but some passes are not
  // currently set up to handle them (such as elide timepoints).
  FunctionLikeNest(passManager).addPass(mlir::createConvertSCFToCFPass);

  //----------------------------------------------------------------------------
  // Whole-program scheduling optimization
  //----------------------------------------------------------------------------

  {
    // We run these under a fixed-point iteration such that we can perform
    // inter-procedural, intra-procedural, and canonicalization as separably
    // verifiable/reusable passes alongside the custom stream ones. IPO will
    // fold duplicate arguments/results and inline constants to allow the local
    // optimizations to work more effectively.
    OpPassManager ipoPipeline(mlir::ModuleOp::getOperationName());

    // IPO and other cleanups.
    buildStreamCleanupPassPipeline(ipoPipeline, transformOptions);

    // TODO(#9747): elide timepoints that are know-reached due to host
    // synchronization via stream.timepoint.await.

    // Elide timepoints in dependency chains where one is known to have been
    // reached by the time another is (A -> B -> A|C).
    ipoPipeline.addPass(IREE::Stream::createElideTimepointsPass());

    // Run fixed-point iteration on the IPO pipeline.
    passManager.addPass(
        IREE::Util::createFixedPointIteratorPass(std::move(ipoPipeline)));
  }

  //----------------------------------------------------------------------------
  // Binding optimization
  //----------------------------------------------------------------------------

  if (transformOptions.optimizeBindings) {
    // Fuse bindings together and add operands for their subview ranges.
    passManager.addPass(IREE::Stream::createFuseDispatchBindingsPass());

    // TODO(benvanik): canonicalize bindings: we should sort the bindings by
    // the block argument order of the parent stream.cmd.execute. This will get
    // us more regular descriptor set layouts. We could also use some other
    // heuristics (all constant bindings -> transients -> external etc) to
    // make partitioning the bindings easier. Note we need to update both the
    // dispatches and the dispatch function argument order.
  }

  // Annotate dispatch region arguments based on the operands passed at dispatch
  // sites. This allows codegen to see the potential values for the operands
  // when operating locally on executables.
  passManager.addPass(IREE::Stream::createAnnotateDispatchArgumentsPass());
  passManager.addPass(IREE::Stream::createAnnotateDispatchAssumptionsPass());

  // Pack dispatch operands on stream.executable into i32 values.
  // We do this prior to exiting the pipeline as here we can still easily
  // add/remove operands.
  passManager.addPass(IREE::Stream::createPackDispatchOperandsPass());

  // Folding operands requires that canonicalization/CSE folds the inputs that
  // we check for.
  buildStreamCleanupPassPipeline(passManager, transformOptions);
  passManager.addPass(IREE::Stream::createFoldUniformOperandsPass());

  // Only want to specialize after we've added all the operands we need above.
  // TODO(benvanik): make codegen more efficient with the specialized
  // constants. The lookup tables inserted are currently extremely slow on
  // some backends.
  // passManager.addPass(IREE::Stream::createSpecializeDispatchesPass());

  // TODO(benvanik): when we spill push constants spill to staging buffers.
  // Need to know push constant limit but that could be specified as a stream
  // option (max operand count).
}

//===----------------------------------------------------------------------===//
// --iree-stream-transformation-pipeline
//===----------------------------------------------------------------------===//

void buildStreamTransformPassPipeline(
    OpPassManager &passManager, const TransformOptions &transformOptions) {
  //----------------------------------------------------------------------------
  // Primary pipeline stages (required)
  //----------------------------------------------------------------------------

  buildStreamTensorPassPipeline(passManager, transformOptions);
  buildStreamAsyncPassPipeline(passManager, transformOptions);
  buildStreamCmdPassPipeline(passManager, transformOptions);

  // Dump statistics before the deeper optimizations happen.
  // Optimizations such as dispatch operand fusion remove information we can use
  // to determine memory usage by dispatches.
  if (transformOptions.dumpStatisticsFormat != DumpOutputFormat::None) {
    DumpStatisticsPassOptions dumpStatisticsOptions;
    dumpStatisticsOptions.outputFormat = transformOptions.dumpStatisticsFormat;
    dumpStatisticsOptions.outputFile = transformOptions.dumpStatisticsFile;
    passManager.addPass(
        IREE::Stream::createDumpStatisticsPass(dumpStatisticsOptions));
  }

  //----------------------------------------------------------------------------
  // Optimizations (may be required by some targets)
  //----------------------------------------------------------------------------

  buildStreamOptimizationPassPipeline(passManager, transformOptions);

  //----------------------------------------------------------------------------
  // Post-pipeline cleanup
  //----------------------------------------------------------------------------

  // Final cleanup after we optimize dispatches and fuse operands and bindings.
  buildStreamCleanupPassPipeline(passManager, transformOptions);

  // Symbol DCE any remaining variables/functions that are now no longer
  // required.
  passManager.addPass(mlir::createSymbolDCEPass());
}

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

namespace {
#define GEN_PASS_REGISTRATION
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h.inc" // IWYU pragma: export
} // namespace

void registerStreamPasses() {
  // Generated.
  registerPasses();

  // Pipelines.
  PassPipelineRegistration<TransformOptions> cleanupPassPipeline(
      "iree-stream-cleanup-pipeline",
      "Runs the cleanup passes that are performed between stages of the full "
      "stream pipeline.",
      [](OpPassManager &passManager, const TransformOptions &transformOptions) {
        buildStreamCleanupPassPipeline(passManager, transformOptions);
      });
  PassPipelineRegistration<TransformOptions> tensorPassPipeline(
      "iree-stream-tensor-transformation-pipeline",
      "Lowers source dialects into stream.tensor.* IR.",
      [](OpPassManager &passManager, const TransformOptions &transformOptions) {
        buildStreamTensorPassPipeline(passManager, transformOptions);
      });
  PassPipelineRegistration<TransformOptions> asyncPassPipeline(
      "iree-stream-async-transformation-pipeline",
      "Lowers stream.tensor.* to stream.async.* IR.",
      [](OpPassManager &passManager, const TransformOptions &transformOptions) {
        buildStreamAsyncPassPipeline(passManager, transformOptions);
      });
  PassPipelineRegistration<TransformOptions> cmdPassPipeline(
      "iree-stream-cmd-transformation-pipeline",
      "Lowers stream.async.* to stream.cmd.* IR.",
      [](OpPassManager &passManager, const TransformOptions &transformOptions) {
        buildStreamCmdPassPipeline(passManager, transformOptions);
      });
  PassPipelineRegistration<TransformOptions> optimizationPassPipeline(
      "iree-stream-optimization-pipeline",
      "Optimizes stream commands and resources (may be required for some "
      "targets).",
      [](OpPassManager &passManager, const TransformOptions &transformOptions) {
        buildStreamOptimizationPassPipeline(passManager, transformOptions);
      });
  PassPipelineRegistration<TransformOptions> transformPassPipeline(
      "iree-stream-transformation-pipeline",
      "Runs the full IREE stream dialect transformation pipeline.",
      [](OpPassManager &passManager, const TransformOptions &transformOptions) {
        buildStreamTransformPassPipeline(passManager, transformOptions);
      });
}

} // namespace mlir::iree_compiler::IREE::Stream

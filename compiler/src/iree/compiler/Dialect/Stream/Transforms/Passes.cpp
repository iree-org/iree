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
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Stream {

using FunctionLikeNest = MultiOpNest<func::FuncOp, IREE::Util::InitializerOp>;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

static void addCleanupPatterns(OpPassManager &passManager) {
  FunctionLikeNest(passManager)
      // Standard MLIR cleanup.
      .addPass(mlir::createCanonicalizerPass)
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
// -iree-stream-tensor-transformation-pipeline
//===----------------------------------------------------------------------===//

void buildStreamTensorPassPipeline(OpPassManager &passManager,
                                   const TransformOptions &transformOptions) {
  //----------------------------------------------------------------------------
  // Input cleanup and simplification
  //----------------------------------------------------------------------------

  // Verify we support the program.
  passManager.addPass(IREE::Stream::createVerifyInputPass());

  // Turn all constant ops into global variables and fix up the IR.
  // As many locations change and constants are deduplicated we'll end up with
  // a lot of extraneous IR (mostly global loads) and clean those up here.
  passManager.addPass(IREE::Stream::createOutlineConstantsPass());

  // Perform cleanup after constnat simplification as more canonicalizers may be
  // able to kick in.
  addCleanupPatterns(passManager);

  //----------------------------------------------------------------------------
  // Conversion
  //----------------------------------------------------------------------------

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

  // Cleanup globals that were created during conversion.
  addCleanupPatterns(passManager);

  // Bring all initializers together so that we can schedule them.
  passManager.addPass(IREE::Util::createCombineInitializersPass());

  //----------------------------------------------------------------------------
  // Stream affinity/assignment
  //----------------------------------------------------------------------------

  // TODO(benvanik): pin based on target backends here.
  // TODO(benvanik): compute affinities for executables.
  // TODO(benvanik): annotate all dispatches with preferred executable affinity.
  // TODO(benvanik): DFA to specify all value affinities and pin dispatches.
}

//===----------------------------------------------------------------------===//
// -iree-stream-async-transformation-pipeline
//===----------------------------------------------------------------------===//

void buildStreamAsyncPassPipeline(OpPassManager &passManager,
                                  const TransformOptions &transformOptions) {
  //----------------------------------------------------------------------------
  // Tensor lowering and resource management
  //----------------------------------------------------------------------------

  // Lower stream.tensor.* ops to stream.async.* ops based on
  // affinity/configuration assigned during placement.
  FunctionLikeNest(passManager)
      .addPass(IREE::Stream::createEncodeHostTensorsPass);
  passManager.addNestedPass<IREE::Stream::ExecutableOp>(
      IREE::Stream::createEncodeDeviceTensorsPass());

  // Expand builtins to dispatches. This may introduce new executables.
  passManager.addPass(IREE::Stream::createMaterializeBuiltinsPass());

  addCleanupPatterns(passManager);

  // Materialize copy-on-write behavior with explicit stream.async.* ops.
  // This will insert a lot of copies, so follow it up with a pass that elides
  // ones that aren't needed. This is easier to verify than if there was one
  // pass attempting to do both. Note that copy-on-write materialization is
  // required for correct execution while copy elision is for performance only
  // (though it's critical enough that it is not optional).
  FunctionLikeNest(passManager)
      .addPass(IREE::Stream::createMaterializeCopyOnWritePass);
  passManager.addPass(IREE::Stream::createElideAsyncCopiesPass());
  FunctionLikeNest(passManager)
      .addPass(mlir::createCanonicalizerPass)
      .addPass(IREE::Stream::createEmplaceAllocationsPass);

  // Refine lifetime of all resources across the module.
  // We do this after scheduling execution so that we know how the resources
  // move across devices. We do it before scheduling waves as lifetime doesn't
  // change and it makes the IR cleaner.
  passManager.addPass(IREE::Stream::createRefineUsagePass());
  addCleanupPatterns(passManager);

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

  // Materialize timepoints across the entire module. This simplifies scheduling
  // of the timeline as we can shake the IR and see what timepoints we still
  // have left.
  passManager.addPass(IREE::Stream::createPropagateTimepointsPass());
  addCleanupPatterns(passManager);

  // Everything must now be in stream.async.* form.
  passManager.addPass(IREE::Stream::createVerifyLoweringToAsyncPass());
}

//===----------------------------------------------------------------------===//
// -iree-stream-cmd-transformation-pipeline
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

      // Pack fused allocations based on lifetime.
      .addPass(IREE::Stream::createPackAllocationsPass)

      // Layout packed slices to emit the arithmetic required for all resource
      // offsets. This enables us to propagate the subviews across the program
      // below.
      .addPass(IREE::Stream::createLayoutSlicesPass);

  // Propagate subviews throughout the program to unify resource storage access.
  // After propagation many resource SSA values can be deduped or folded by the
  // cleanup patterns.
  passManager.addPass(IREE::Util::createPropagateSubrangesPass());
  addCleanupPatterns(passManager);

  // TODO(benvanik): outline streams (ala dispatch regions). Note that we may
  // want to do this earlier to enable better deduplication but that makes the
  // above passes trickier. Outlining may be more like "find chunks of streams
  // useful to move into secondary command buffers."

  // Everything must now be in explicit stream.cmd.* form.
  passManager.addPass(IREE::Stream::createVerifyLoweringToCmdPass());
}

//===----------------------------------------------------------------------===//
// -iree-stream-optimization-pipeline
//===----------------------------------------------------------------------===//

void buildStreamOptimizationPassPipeline(
    OpPassManager &passManager, const TransformOptions &transformOptions) {
  // Forming streams involves a fair amount of subgraph stitching, which can
  // cause duplication. Run CSE to collapse.
  addCleanupPatterns(passManager);

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
    addCleanupPatterns(ipoPipeline);

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

  // Pack dispatch operands on stream.executable into i32 values.
  // We do this prior to exiting the pipeline as here we can still easily
  // add/remove operands.
  passManager.addPass(IREE::Stream::createPackDispatchOperandsPass());

  // Folding operands requires that CSE folds the inputs that we check for.
  passManager.addPass(mlir::createCSEPass());
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
// -iree-stream-transformation-pipeline
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
    passManager.addPass(IREE::Stream::createDumpStatisticsPass(
        transformOptions.dumpStatisticsFormat,
        transformOptions.dumpStatisticsFile));
  }

  //----------------------------------------------------------------------------
  // Optimizations (may be required by some targets)
  //----------------------------------------------------------------------------

  buildStreamOptimizationPassPipeline(passManager, transformOptions);

  //----------------------------------------------------------------------------
  // Post-pipeline cleanup
  //----------------------------------------------------------------------------

  // Final cleanup after we optimize dispatches and fuse operands and bindings.
  addCleanupPatterns(passManager);

  // Symbol DCE any remaining variables/functions that are now no longer
  // required.
  passManager.addPass(mlir::createSymbolDCEPass());
}

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

void registerStreamTransformPassPipelines() {
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

namespace {
#define GEN_PASS_REGISTRATION
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h.inc"  // IWYU pragma: export
}  // namespace

void registerStreamPasses() {
  // Generated.
  registerPasses();

  // Pipelines.
  registerStreamTransformPassPipelines();
}

}  // namespace Stream
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

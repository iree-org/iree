// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"

#include <memory>

#include "iree/compiler/Dialect/Shape/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Stream {

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

static void addCleanupPatterns(OpPassManager &passManager) {
  // Standard MLIR cleanup.
  passManager.addPass(mlir::createCanonicalizerPass());
  passManager.addPass(mlir::createCSEPass());

  // Cleanup and canonicalization of util.global (and other util ops).
  passManager.addPass(IREE::Util::createApplyPatternsPass());
  passManager.addPass(IREE::Util::createFoldGlobalsPass());
  passManager.addPass(IREE::Util::createFuseGlobalsPass());

  // Simplify util.global accesses; this can help with data flow tracking as
  // redundant store-loads are removed.
  passManager.addNestedPass<IREE::Util::InitializerOp>(
      IREE::Util::createSimplifyGlobalAccessesPass());
  passManager.addNestedPass<mlir::FuncOp>(
      IREE::Util::createSimplifyGlobalAccessesPass());
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
                                  const TransformOptions &transformOptions) {}

//===----------------------------------------------------------------------===//
// -iree-stream-cmd-transformation-pipeline
//===----------------------------------------------------------------------===//

void buildStreamCmdPassPipeline(OpPassManager &passManager,
                                const TransformOptions &transformOptions) {}

//===----------------------------------------------------------------------===//
// -iree-stream-optimization-pipeline
//===----------------------------------------------------------------------===//

void buildStreamOptimizationPassPipeline(
    OpPassManager &passManager, const TransformOptions &transformOptions) {}

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

  //----------------------------------------------------------------------------
  // Optimizations (may be required by some targets)
  //----------------------------------------------------------------------------

  buildStreamOptimizationPassPipeline(passManager, transformOptions);

  //----------------------------------------------------------------------------
  // Post-pipeline cleanup
  //----------------------------------------------------------------------------

  // Forming streams involves a fair amount of subgraph stitching, which can
  // cause duplication. Run CSE to collapse.
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

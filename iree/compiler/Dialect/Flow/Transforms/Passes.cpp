// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"

#include <memory>

#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "iree/compiler/Utils/PassUtils.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
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
    "iree-flow-trace-dispatch-tensors2",
    llvm::cl::desc(
        "Trace runtime input/output tensors for each dispatch function."),
    llvm::cl::init(false));

static llvm::cl::opt<bool> clDemoteF32ToF16(
    "iree-flow-demote-f32-to-f16",
    llvm::cl::desc("Convert all f32 ops and values into f16 counterparts "
                   "unconditionally before main flow conversions"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> clEnableConvToImg2Col(
    "iree-flow-enable-conv-img2col-transform",
    llvm::cl::desc("Enable converting convolution ops to img2col form."),
    llvm::cl::init(false));

static llvm::cl::opt<bool> clEnablePaddingLinalgOps(
    "iree-flow-enable-padding-linalg-ops",
    llvm::cl::desc("Enable padding linalg ops to an integer multiple of "
                   "flow-padding-size"),
    llvm::cl::init(false));

static llvm::cl::opt<int> clLinalgOpsPaddingSize(
    "iree-flow-linalg-ops-padding-size",
    llvm::cl::desc("Enable padding linalg ops to an integer multiple of "
                   "flow-padding-size"),
    llvm::cl::init(4));

// TODO(#1159): enable by default or remove this option once it works on
//              a broader set of programs
static llvm::cl::opt<bool> clEnableLinalgDetensorize(
    "iree-flow-enable-linalg-detensorize",
    llvm::cl::desc("Enable detensorizing linalg ops to operate on primitives"),
    llvm::cl::init(false));

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

namespace {

using FunctionLikeNest = MultiOpNest<FuncOp, IREE::Util::InitializerOp>;

// Subset of the overall pass pipeline for optimizing globals and numerics.
// We may ultimately break this out separately so creating a syntactic
// distinction to keep that as an option.
void buildGlobalOptimizationPassPipeline(
    OpPassManager &mainPassManager, const TransformOptions &transformOptions) {
  OpPassManager pipeline(ModuleOp::getOperationName());

  FunctionLikeNest(pipeline)
      // Simplify util.global accesses early on; this can help with dispatch
      // region formation as redundant store-loads are removed.
      .addPass(IREE::Util::createSimplifyGlobalAccessesPass);

  // Module level cleanup and canonicalization of util.global (and other util
  // ops).
  pipeline.addPass(IREE::Util::createApplyPatternsPass());
  pipeline.addPass(IREE::Util::createFoldGlobalsPass());

  if (transformOptions.constExprHoisting) {
    pipeline.addPass(IREE::Util::createHoistIntoGlobalsPass());
  }

  if (transformOptions.buildConstEvalPassPipeline) {
    transformOptions.buildConstEvalPassPipeline(pipeline);
  }

  if (transformOptions.numericPrecisionReduction) {
    pipeline.addPass(createInferNumericNarrowingPass());
    pipeline.addPass(createOptimizeNumericsPass());
    pipeline.addPass(createCleanupNumericNarrowingPass());
  }

  FunctionLikeNest(pipeline)
      .addPass(mlir::createCanonicalizerPass)
      .addPass(mlir::createCSEPass);

  // Add the whole fixed point iterator.
  mainPassManager.addPass(
      IREE::Util::createFixedPointIteratorPass(std::move(pipeline)));
}

}  // namespace

void buildFlowTransformPassPipeline(OpPassManager &passManager,
                                    const TransformOptions &transformOptions) {
  // Special case peephole optimizations.
  FunctionLikeNest(passManager)
      .addPass(createConvertConv2D1x1ToMatmulPass)
      .addPredicatedPass(clEnableConvToImg2Col,
                         createConvertConv2DToImg2ColPass)
      // Pad linalg op
      .addPredicatedPass(clEnablePaddingLinalgOps,
                         []() {
                           return createPadLinalgOpsToIntegerMultiplePass(
                               clLinalgOpsPaddingSize);
                         })

      // Input should now be legal.
      .addPass(createVerifyInputLegalityPass);

  passManager.addPass(mlir::createLinalgNamedOpConversionPass());
  buildGlobalOptimizationPassPipeline(passManager, transformOptions);

  // Perform cleanup after variable simplification as more canonicalizers may be
  // able to kick in.
  FunctionLikeNest(passManager)
      // Pad tensors.
      .addPass(createPadTensorToSubTensorInsertPass)

      // Elementwise, fusion, tiling and distribution.
      .addPass(mlir::createConvertElementwiseToLinalgPass)
      .addPass(mlir::createLinalgFoldUnitExtentDimsPass)
      .addPass(createInterchangeGenericOpsPass)
      .addPass(mlir::createCanonicalizerPass)
      .addPass(memref::createResolveShapedTypeResultDimsPass)

      // Fusion.
      .addPass(createFusionOfTensorOpsPass)
      .addPass(mlir::createCSEPass)
      .addPredicatedPass(clEnableLinalgDetensorize,
                         mlir::createLinalgDetensorizePass)
      // Dispatch region formation.
      .addPass(createConvertToFlowBeforeDispatchFormation)
      .addPass(mlir::createCanonicalizerPass)
      .addPass(createDispatchLinalgOnTensorsPass)
      .addPass(memref::createResolveShapedTypeResultDimsPass)
      .addPass(createCaptureDispatchDynamicDimsPass)
      .addPass(createConvertToFlowAfterDispatchFormation)
      .addPass(mlir::createCanonicalizerPass)
      .addPass(memref::createResolveShapedTypeResultDimsPass)

      // Cleanup again?
      .addPass(createConvertToFlowAfterDispatchFormation)
      // NOTE: required because the current dispatch-linalg-on-tensors pass
      // creates a lot of dead IR that needs to be cleaned up.
      .addPass(mlir::createCanonicalizerPass);

  // Module pass to outline the dispatch regions into their own functions
  // wrapped in executables.
  passManager.addPass(createOutlineDispatchRegionsPass());

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
  passManager.addPass(createDeduplicateExecutablesPass());

  // Create one function per remaining flow.executable that can be used with
  // iree-benchmark-module to benchmark each dispatch individually, as well as
  // exporting all original model entry points.
  if (clExportBenchmarkFuncs) {
    passManager.addPass(IREE::Flow::createExportBenchmarkFuncsPass());
  }

  FunctionLikeNest(passManager)
      // Inject tracing that logs both input and output tensors from all
      // dispatches. We do this after deduping so that the executable names
      // match later stages.
      .addPredicatedPass(clTraceDispatchTensors,
                         IREE::Flow::createInjectDispatchTracingPass)
      // Cleanup the IR after we are done.
      .addPass(mlir::createCanonicalizerPass)
      .addPass(mlir::createCSEPass);

  passManager.addNestedPass<IREE::Flow::ExecutableOp>(
      mlir::createCanonicalizerPass());
  passManager.addNestedPass<IREE::Flow::ExecutableOp>(mlir::createCSEPass());

  // Symbol DCE any remaining variables/functions that are now no longer
  // required.
  passManager.addPass(mlir::createSymbolDCEPass());
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
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h.inc"  // IWYU pragma: export
}  // namespace

/// Test passes.
std::unique_ptr<OperationPass<void>>
createTestPartitionableLoopsInterfacePass();

/// Register test passes.
inline void registerTestPasses() {
  createTestPartitionableLoopsInterfacePass();
}

void registerFlowPasses() {
  // Generated.
  registerPasses();

  // Test passes.
  registerTestPasses();

  // Pipelines.
  registerFlowTransformPassPipeline();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"

#include <memory>

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Shape/Transforms/Passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

namespace {

struct TransformOptions : public PassPipelineOptions<TransformOptions> {
  Option<bool> serializeExecutables{
      *this, "serialize-executables",
      llvm::cl::desc("Whether to serialize hal.executable.variant ops to "
                     "hal.executable.binary ops."),
      llvm::cl::init(true)};
  Option<bool> linkExecutables{
      *this, "link-executables",
      llvm::cl::desc("Whether to link hal.executable ops together."),
      llvm::cl::init(true)};
};

static llvm::cl::opt<unsigned> benchmarkDispatchRepeatCount{
    "iree-hal-benchmark-dispatch-repeat-count",
    llvm::cl::desc(
        "The number of times to repeat each hal.command_buffer.dispatch op. "
        "(Not that this simply duplicates the dispatch op and inserts "
        "barriers. It's meant for command buffers having linear dispatch "
        "structures.)"),
    llvm::cl::init(1)};

}  // namespace

void buildHALTransformPassPipeline(OpPassManager &passManager,
                                   const TargetOptions &targetOptions,
                                   const TransformOptions &transformOptions) {
  passManager.addPass(createCanonicalizerPass());

  // Handle large constants (weights/params/etc) first so that we can use the
  // resulting constant pools to determine the interfaces.
  passManager.addPass(createIdentifyConstantPoolsPass(targetOptions));
  passManager.addNestedPass<ConstantPoolOp>(
      createPackConstantPoolStoragePass());
  passManager.addPass(createMaterializeConstantPoolBuffersPass());
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createSymbolDCEPass());

  // Each executable needs a hal.interface to specify how the host and device
  // comminucate across the ABI boundary.
  passManager.addPass(createMaterializeInterfacesPass(targetOptions));

  passManager.nest<ExecutableOp>().addNestedPass<ExecutableVariantOp>(
      createPropagateConstantWorkgroupInfoPass());
  passManager.nest<ExecutableOp>().addNestedPass<ExecutableVariantOp>(
      createTranslateExecutablesPass(targetOptions));

  // Convert supported input dialects (std, flow, etc) into the HAL dialect.
  passManager.addPass(createConvertToHALPass());

  // Phase ordering note: Before this pass, functions signatures will be based
  // on explicit shape types (such as ranked_shape). After this pass, these
  // composite types will be expanded to primitives (i.e. one 'index' for each
  // dynamic dim in the case of ranked_shape).
  passManager.addNestedPass<FuncOp>(
      Shape::createExpandFunctionRankedShapeDimsPass());

  passManager.addNestedPass<FuncOp>(createCanonicalizerPass());
  passManager.addNestedPass<FuncOp>(createCSEPass());

  // Pack transient allocations in the program that were materialized during
  // stream conversion.
  //
  // NOTE: this works best if canonicalization/CSE has run such that the packed
  // sizes are as much as possible available as constants.
  passManager.addNestedPass<FuncOp>(createPackAllocationsPass(targetOptions));

  // After all executables are translated and before resolving entry point
  // ordinals, we allow the backends to link executables together. For example,
  // the LLVM AOT backend may combine all executable targets for the same
  // architecture into a single executable and link it as a shared library.
  // TODO(scotttodd): Move after createTranslateExecutablesPass
  //   * ConvertStreamOps under ConvertFlowToHALPass assumes one entry point.
  //     Adjust it to handle multiple entry points then this can move up.
  if (transformOptions.linkExecutables) {
    passManager.addPass(createLinkExecutablesPass(targetOptions));
  }

  // Resolve entry point ordinals from nested symbol references prior to
  // serialization. As this pass creates lookup ops it should run before
  // MaterializeResourceCachesPass.
  passManager.addPass(createResolveEntryPointOrdinalsPass());
  passManager.addNestedPass<FuncOp>(createCanonicalizerPass());
  passManager.addNestedPass<FuncOp>(createCSEPass());

  // Gather cachable resources such as executables and descriptor sets and
  // cache them at initialization-time.
  passManager.addPass(createMaterializeResourceCachesPass(targetOptions));

  // Inline hal.device.switch ops and memoize their queries such that we can
  // better CSE/fold dispatch logic.
  passManager.addNestedPass<FuncOp>(createInlineDeviceSwitchesPass());
  if (benchmarkDispatchRepeatCount != 1) {
    passManager.addNestedPass<FuncOp>(
        createBenchmarkBatchDispatchesPass(benchmarkDispatchRepeatCount));
  }
  passManager.addPass(createLowerAffinePass());
  passManager.addPass(createMemoizeDeviceQueriesPass());
  passManager.addNestedPass<FuncOp>(createCanonicalizerPass());
  passManager.addNestedPass<FuncOp>(createCSEPass());

  // Run our own CSE on variable loads before moving on.
  // When specifying side effects can help MLIR's core CSE pass eliminate
  // redundant loads we can remove this.
  passManager.addNestedPass<FuncOp>(createCSEVariableLoadsPass());

  if (transformOptions.serializeExecutables) {
    passManager.addNestedPass<ExecutableOp>(
        createSerializeExecutablesPass(targetOptions));
    // NOTE: symbol DCE will destroy executable target contents, so only run it
    // if we serialized things.
    passManager.addPass(createSymbolDCEPass());
  }

  // Final cleanup of IR; cleans up things left behind by CSE/DCE above.
  passManager.addNestedPass<FuncOp>(createCanonicalizerPass());
}

void buildHALTransformPassPipeline(OpPassManager &passManager,
                                   const TargetOptions &targetOptions) {
  TransformOptions transformOptions;
  buildHALTransformPassPipeline(passManager, targetOptions, transformOptions);
}

void registerHALTransformPassPipeline() {
  PassPipelineRegistration<TransformOptions>(
      "iree-hal-transformation-pipeline",
      "Runs the full IREE HAL dialect transformation pipeline",
      [](OpPassManager &passManager, const TransformOptions &transformOptions) {
        buildHALTransformPassPipeline(passManager, getTargetOptionsFromFlags(),
                                      transformOptions);
      });
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

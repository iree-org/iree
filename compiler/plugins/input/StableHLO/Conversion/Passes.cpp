// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/input/StableHLO/Conversion/Passes.h"

#include "compiler/plugins/input/StableHLO/Conversion/Preprocessing/Passes.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "iree/compiler/InputConversion/Common/Passes.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ShapeToStandard/ShapeToStandard.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/Shape/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"
#include "stablehlo/transforms/Passes.h"

namespace mlir::iree_compiler::stablehlo {
namespace {
#define GEN_PASS_REGISTRATION
#include "compiler/plugins/input/StableHLO/Conversion/Passes.h.inc" // IWYU pragma: export
} // namespace

namespace {

void registerStableHLOConversionPassPipeline() {
  PassPipelineRegistration<StableHloOptions> stablehlo(
      "iree-stablehlo-input-transformation-pipeline",
      "Runs the StableHLO IREE flow dialect transformation pipeline",
      [](OpPassManager &passManager, const StableHloOptions &options) {
        buildStableHLOInputConversionPassPipeline(passManager, options);
      });
}

// Prepare HLO for use as an input to the Flow dialect.
void buildStableHLOInputConversionPassPipelineImpl(
    OpPassManager &passManager, const StableHloOptions &options, bool detuple) {
  // Having both StableHLO and VHLO in the same module is not supported.
  // If the input is VHLO, then it is automatically converted to StableHLO.
  // If the input is StableHLO, this pass is considered a NOP.
  passManager.addPass(stablehlo::createCheckVHLOStableHloMixUsage());
  ::mlir::stablehlo::createStablehloDeserializePipeline(passManager);
  passManager.addNestedPass<func::FuncOp>(mlir::createCanonicalizerPass());
  passManager.addNestedPass<func::FuncOp>(createStableHLOCanonicalize());
  passManager.addNestedPass<func::FuncOp>(mlir::createCSEPass());
  passManager.addNestedPass<func::FuncOp>(createLegalizeStableHLOCustomCalls());
  passManager.addNestedPass<func::FuncOp>(
      stablehlo::createLegalizeControlFlow());

  passManager.addPass(createFlattenTuplesInSCF());
  if (detuple) {
    passManager.addPass(createFlattenTuplesInCFG());
  }

  passManager.addPass(createStableHLOToStableHLOPreprocessing());
  passManager.addNestedPass<func::FuncOp>(createStableHLOCanonicalize());

  // Various shape functions may have been materialized in the `shape.shape_of`
  // style of treating shapes as tensors. We prefer to legalize these to
  // scalar ops as early as possible to avoid having them persist as tensor
  // computations.
  passManager.addNestedPass<func::FuncOp>(createShapeToShapeLowering());
  passManager.addPass(createConvertShapeToStandardPass());
  passManager.addNestedPass<func::FuncOp>(mlir::createCanonicalizerPass());
  passManager.addNestedPass<func::FuncOp>(createStableHLOCanonicalize());

  // We also don't handle calls well on the old codepath; until we remove the
  // use of the CFG we can continue inlining.
  passManager.addPass(mlir::createInlinerPass());

  // Perform initial cleanup. createLegalizeInputTypes could rewrite types. In
  // this context, some operations could be folded away.
  passManager.addNestedPass<func::FuncOp>(mlir::createCanonicalizerPass());
  passManager.addNestedPass<func::FuncOp>(createStableHLOCanonicalize());
  passManager.addNestedPass<func::FuncOp>(mlir::createCSEPass());

  // Convert to Linalg. After this point, StableHLO will be eliminated.
  passManager.addNestedPass<func::FuncOp>(
      stablehlo::createLegalizeShapeComputations());
  passManager.addNestedPass<func::FuncOp>(
      stablehlo::createConvertStableHloToLinalgExt());
  passManager.addNestedPass<func::FuncOp>(stablehlo::createLegalizeChlo());
  passManager.addPass(createConvertStableHloToIreeInputDialects());
  passManager.addPass(createReconcileUnrealizedCastsPass());

  // Note that some StableHLO ops are left by the above and must resolve via
  // canonicalization. See comments in the above pass and find a better way.
  passManager.addNestedPass<func::FuncOp>(mlir::createCanonicalizerPass());
  passManager.addNestedPass<func::FuncOp>(createStableHLOCanonicalize());

  passManager.addPass(stablehlo::createVerifyCompilerStableHloInputLegality());
}
} // namespace

void buildStableHLOInputConversionPassPipeline(
    OpPassManager &passManager, const StableHloOptions &options) {
  buildStableHLOInputConversionPassPipelineImpl(passManager, options,
                                                /*detuple=*/false);
}

void buildStableHLOXLAInputConversionPassPipeline(
    OpPassManager &passManager, const StableHloOptions &options) {
  buildStableHLOInputConversionPassPipelineImpl(passManager, options,
                                                /*detuple=*/true);
}

void registerStableHLOConversionPasses() {
  // Generated.
  registerPasses();

  registerStableHLOPreprocessingPasses();
  registerStableHLOConversionPassPipeline();
}

} // namespace mlir::iree_compiler::stablehlo

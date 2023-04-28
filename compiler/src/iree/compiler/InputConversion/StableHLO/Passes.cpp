// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/InputConversion/StableHLO/Passes.h"

#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "iree/compiler/InputConversion/Common/Passes.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ShapeToStandard/ShapeToStandard.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/Shape/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::iree_compiler::stablehlo {
namespace {
#define GEN_PASS_REGISTRATION
#include "iree/compiler/InputConversion/StableHLO/Passes.h.inc"  // IWYU pragma: export
}  // namespace

namespace {
// TODO(#8745): remove these flags when the -iree-flow-demote-* flags can be
// used without tripping upstream verifier issues.
llvm::cl::opt<bool> clDemoteI64ToI32(
    "iree-stablehlo-demote-i64-to-i32",
    llvm::cl::desc(
        "Converts all StableHLO i64 ops and values into i32 counterparts."),
    llvm::cl::init(true));
llvm::cl::opt<bool> clDemoteF64ToF32(
    "iree-stablehlo-demote-f64-to-f32",
    llvm::cl::desc(
        "Converts all StableHLO f64 ops and values into f32 counterparts."),
    llvm::cl::init(true));
llvm::cl::opt<bool> clPromoteBF16ToF32(
    "iree-stablehlo-promote-bf16-to-f32",
    llvm::cl::desc(
        "Converts all StableHLO bf16 ops and values into f32 counterparts."),
    llvm::cl::init(true));

void registerStableHLOConversionPassPipeline() {
  PassPipelineRegistration<> stablehlo(
      "iree-stablehlo-input-transformation-pipeline",
      "Runs the StableHLO IREE flow dialect transformation pipeline",
      [](OpPassManager &passManager) {
        buildStableHLOInputConversionPassPipeline(passManager);
      });
}

// Prepare HLO for use as an input to the Flow dialect.
void buildStableHLOInputConversionPassPipelineImpl(OpPassManager &passManager) {
  passManager.addNestedPass<func::FuncOp>(mlir::createCanonicalizerPass());
  passManager.addNestedPass<func::FuncOp>(
      stablehlo::createLegalizeControlFlow());

  // Currently we don't handle SCF ops well and have to convert them all to CFG.
  // In the future it would be nice if we could have all of flow be both scf
  // and cfg compatible.
  passManager.addNestedPass<func::FuncOp>(createTopLevelSCFToCFGPass());
  // TODO(#12678): Port StableHLO detuple pass.

  // TODO(#12678): Port StableHLO-StableHLO preprocessing.
  passManager.addNestedPass<func::FuncOp>(mlir::createCanonicalizerPass());

  // Various shape functions may have been materialized in the `shape.shape_of`
  // style of treating shapes as tensors. We prefer to legalize these to
  // scalar ops as early as possible to avoid having them persist as tensor
  // computations.
  passManager.addNestedPass<func::FuncOp>(createShapeToShapeLowering());
  passManager.addPass(createConvertShapeToStandardPass());
  passManager.addNestedPass<func::FuncOp>(mlir::createCanonicalizerPass());

  // We also don't handle calls well on the old codepath; until we remove the
  // use of the CFG we can continue inlining.
  passManager.addPass(mlir::createInlinerPass());

  // Hacky type conversion to work around lack of type support lower in the
  // stack. This is often required because of implicit i64 insertion by JAX/HLO
  // that we don't want forcing 32-bit embedded devices to support.
  // TODO(#8745): remove these and prefer the flow pipeline options instead.
  if (clDemoteI64ToI32) {
    passManager.addPass(IREE::Util::createDemoteI64ToI32Pass());
  }
  if (clDemoteF64ToF32) {
    passManager.addPass(IREE::Util::createDemoteF64ToF32Pass());
  }
  if (clPromoteBF16ToF32) {
    passManager.addPass(IREE::Util::createPromoteBF16ToF32Pass());
  }

  // Perform initial cleanup. createLegalizeInputTypes could rewrite types. In
  // this context, some operations could be folded away.
  passManager.addNestedPass<func::FuncOp>(mlir::createCanonicalizerPass());
  passManager.addNestedPass<func::FuncOp>(mlir::createCSEPass());

  // Convert to Linalg. After this point, StableHLO will be eliminated.
  passManager.addNestedPass<func::FuncOp>(
      stablehlo::createLegalizeShapeComputations());
  passManager.addNestedPass<func::FuncOp>(
      stablehlo::createConvertStableHloToLinalgExt());
  passManager.addPass(stablehlo::createConvertStableHloToLinalg());
  // Ensure conversion completed.
  passManager.addPass(createReconcileUnrealizedCastsPass());

  // Note that some StableHLO ops are left by the above and must resolve via
  // canonicalization. See comments in the above pass and find a better way.
  passManager.addNestedPass<func::FuncOp>(mlir::createCanonicalizerPass());

  // TODO(#12678): Port StableHLO input legality verification pass.
}
}  // namespace

void buildStableHLOInputConversionPassPipeline(OpPassManager &passManager) {
  buildStableHLOInputConversionPassPipelineImpl(passManager);
}

void registerStableHLOConversionPasses() {
  // Generated.
  registerPasses();

  registerStableHLOConversionPassPipeline();
}

}  // namespace mlir::iree_compiler::stablehlo

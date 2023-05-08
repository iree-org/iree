// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/InputConversion/MHLO/Passes.h"

#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "iree/compiler/InputConversion/Common/Passes.h"
#include "mhlo/transforms/passes.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ShapeToStandard/ShapeToStandard.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/Shape/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {
namespace MHLO {

// TODO(#8745): remove these flags when the -iree-flow-demote-* flags can be
// used without tripping upstream verifier issues.
static llvm::cl::opt<bool> clDemoteI64ToI32(
    "iree-mhlo-demote-i64-to-i32",
    llvm::cl::desc(
        "Converts all MHLO i64 ops and values into i32 counterparts."),
    llvm::cl::init(true));
static llvm::cl::opt<bool> clDemoteF64ToF32(
    "iree-mhlo-demote-f64-to-f32",
    llvm::cl::desc(
        "Converts all MHLO f64 ops and values into f32 counterparts."),
    llvm::cl::init(true));
static llvm::cl::opt<bool> clPromoteBF16ToF32(
    "iree-mhlo-promote-bf16-to-f32",
    llvm::cl::desc(
        "Converts all MHLO bf16 ops and values into f32 counterparts."),
    llvm::cl::init(false));

void registerMHLOConversionPassPipeline() {
  PassPipelineRegistration<> mhlo(
      "iree-mhlo-input-transformation-pipeline",
      "Runs the MHLO IREE flow dialect transformation pipeline",
      [](OpPassManager &passManager) {
        buildMHLOInputConversionPassPipeline(passManager);
      });
  PassPipelineRegistration<> xla(
      "iree-xla-input-transformation-pipeline",
      "Runs the XLA IREE flow dialect transformation pipeline",
      [](OpPassManager &passManager) {
        buildXLAInputConversionPassPipeline(passManager);
      });
}

// Prepare HLO for use as an input to the Flow dialect.
static void buildMHLOInputConversionPassPipelineImpl(OpPassManager &passManager,
                                                     bool detuple) {
  passManager.addPass(mlir::mhlo::createStablehloLegalizeToHloPass());
  passManager.addNestedPass<func::FuncOp>(mlir::createCanonicalizerPass());
  passManager.addNestedPass<func::FuncOp>(
      mhlo::createLegalizeControlFlowPass());

  // Currently we don't handle SCF ops well and have to convert them all to CFG.
  // In the future it would be nice if we could have all of flow be both scf
  // and cfg compatible.
  passManager.addNestedPass<func::FuncOp>(createTopLevelSCFToCFGPass());
  if (detuple) passManager.addPass(createFlattenTuplesInCFGPass());

  passManager.addNestedPass<func::FuncOp>(createMHLOToMHLOPreprocessingPass());
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

  // Convert to Linalg. After this point, MHLO will be eliminated.
  passManager.addNestedPass<func::FuncOp>(
      mhlo::createLegalizeShapeComputationsPass());
  passManager.addNestedPass<func::FuncOp>(createConvertMHLOToLinalgExtPass());
  passManager.addPass(createMHLOToLinalgOnTensorsPass());
  // Ensure conversion completed.
  passManager.addPass(createReconcileUnrealizedCastsPass());

  // Note that some MHLO ops are left by the above and must resolve via
  // canonicalization. See comments in the above pass and find a better way.
  passManager.addNestedPass<func::FuncOp>(mlir::createCanonicalizerPass());

  //----------------------------------------------------------------------------
  // Entry dialect cleanup
  //----------------------------------------------------------------------------
  passManager.addPass(createVerifyCompilerMHLOInputLegality());
}

void buildMHLOInputConversionPassPipeline(OpPassManager &passManager) {
  buildMHLOInputConversionPassPipelineImpl(passManager, /*detuple=*/false);
}

void buildXLAInputConversionPassPipeline(OpPassManager &passManager) {
  buildMHLOInputConversionPassPipelineImpl(passManager, /*detuple=*/true);
}

namespace {
#define GEN_PASS_REGISTRATION
#include "iree/compiler/InputConversion/MHLO/Passes.h.inc"  // IWYU pragma: export
}  // namespace

void registerMHLOConversionPasses() {
  // Generated.
  registerPasses();

  // Pipelines.
  registerMHLOConversionPassPipeline();
}

}  // namespace MHLO
}  // namespace iree_compiler
}  // namespace mlir

// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/InputConversion/MHLO/Passes.h"

#include "iree/compiler/InputConversion/Common/Passes.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/ShapeToStandard/ShapeToStandard.h"
#include "mlir/Dialect/Shape/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {

void registerMHLOConversionPassPipeline() {
  PassPipelineRegistration<> mhlo(
      "iree-mhlo-input-transformation-pipeline",
      "Runs the MHLO IREE flow dialect transformation pipeline",
      [](OpPassManager &passManager) {
        buildMHLOInputConversionPassPipeline(passManager);
      });
}

// Prepare HLO for use as an input to the Flow dialect.
void buildMHLOInputConversionPassPipeline(OpPassManager &passManager) {
  // Currently we don't handle SCF ops well and have to convert them all to CFG.
  // In the future it would be nice if we could have all of flow be both scf
  // and cfg compatible.
  passManager.addNestedPass<FuncOp>(createTopLevelSCFToCFGPass());

  // Various shape functions may have been materialized in the `shape.shape_of`
  // style of treating shapes as tensors. We prefer to legalize these to
  // scalar ops as early as possible to avoid having them persist as tensor
  // computations.
  passManager.addNestedPass<FuncOp>(createShapeToShapeLowering());
  passManager.addPass(createConvertShapeToStandardPass());
  passManager.addNestedPass<FuncOp>(mlir::createCanonicalizerPass());

  // We also don't handle calls well on the old codepath; until we remove the
  // use of the CFG we can continue inlining.
  passManager.addPass(mlir::createInlinerPass());

  passManager.addNestedPass<FuncOp>(createMHLOToMHLOPreprocessingPass());

  // Legalize input types. We do this after flattening tuples so that we don't
  // have to deal with them.
  // TODO(nicolasvasilache): createLegalizeInputTypesPass is old and does not
  // handle region conversion properly (parent cloned before children). Revisit
  // when using ops with regions such as scf.for and linalg.generic.
  passManager.addPass(mlir::iree_compiler::createLegalizeInputTypesPass());

  // Perform initial cleanup. createLegalizeInputTypes could rewrite types. In
  // this context, some operations could be folded away.
  passManager.addNestedPass<FuncOp>(mlir::createCanonicalizerPass());
  passManager.addNestedPass<FuncOp>(mlir::createCSEPass());

  // Convert to Linalg. After this point, MHLO will be eliminated.
  passManager.addNestedPass<FuncOp>(
      mlir::iree_compiler::createConvertMHLOToLinalgExtPass());
  passManager.addNestedPass<FuncOp>(
      mlir::iree_compiler::createMHLOToLinalgOnTensorsPass());
  // Ensure conversion completed.
  passManager.addPass(createReconcileUnrealizedCastsPass());

  // Note that some MHLO ops are left by the above and must resolve via
  // canonicalization. See comments in the above pass and find a better way.
  passManager.addNestedPass<FuncOp>(mlir::createCanonicalizerPass());

  //----------------------------------------------------------------------------
  // Entry dialect cleanup
  //----------------------------------------------------------------------------
  passManager.addPass(createVerifyCompilerMHLOInputLegality());
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

}  // namespace iree_compiler
}  // namespace mlir

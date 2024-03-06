// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/input/TOSA/tosa-iree/InputConversion/Passes.h"

#include "iree/compiler/InputConversion/Common/Passes.h"
#include "mlir/Conversion/TosaToArith/TosaToArith.h"
#include "mlir/Conversion/TosaToLinalg/TosaToLinalg.h"
#include "mlir/Conversion/TosaToMLProgram/TosaToMLProgram.h"
#include "mlir/Conversion/TosaToSCF/TosaToSCF.h"
#include "mlir/Conversion/TosaToTensor/TosaToTensor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::iree_compiler {

void registerTOSAConversionPassPipeline() {
  PassPipelineRegistration<> tosa(
      "iree-tosa-input-transformation-pipeline",
      "Runs the TOSA IREE flow dialect transformation pipeline",
      [](OpPassManager &passManager) {
        buildTOSAInputConversionPassPipeline(passManager);
      });
}

// Prepare TOSA for use as an input to the Flow dialect.
void buildTOSAInputConversionPassPipeline(OpPassManager &passManager) {
  passManager.addPass(mlir::createTosaToMLProgram());
  // Currently we don't handle SCF ops well and have to convert them all to CFG.
  // In the future it would be nice if we could have all of flow be both scf
  // and cfg compatible.
  passManager.addNestedPass<func::FuncOp>(tosa::createTosaToSCF());

  // We also don't handle calls well on the old codepath; until we remove the
  // use of the CFG we can continue inlining.
  passManager.addPass(mlir::createInlinerPass());

  passManager.addNestedPass<func::FuncOp>(
      tosa::createTosaMakeBroadcastablePass());
  passManager.addNestedPass<func::FuncOp>(tosa::createTosaToArith());
  passManager.addNestedPass<func::FuncOp>(tosa::createTosaToTensor());
  passManager.addNestedPass<func::FuncOp>(
      iree_compiler::createTosaToLinalgExt());
  passManager.addNestedPass<func::FuncOp>(mlir::createCanonicalizerPass());

  TosaToLinalgNamedOptions tosaToLinalgNamedOptions;
  tosaToLinalgNamedOptions.preferConv2DKernelLayoutHWCF = true;
  tosa::addTosaToLinalgPasses(passManager, TosaToLinalgOptions(),
                              tosaToLinalgNamedOptions);
  passManager.addNestedPass<func::FuncOp>(
      iree_compiler::createConverti48Toi64());

  // Sometimes we generate more TOSA operations during the lowering to linalg.
  passManager.addNestedPass<func::FuncOp>(tosa::createTosaToArith());
  passManager.addNestedPass<func::FuncOp>(tosa::createTosaToTensor());

  passManager.addNestedPass<func::FuncOp>(
      iree_compiler::createStripSignednessPass());
  passManager.addNestedPass<func::FuncOp>(mlir::createCanonicalizerPass());

  passManager.addNestedPass<func::FuncOp>(
      createLinalgQuantizedMatmulToMatmulPass());
  passManager.addNestedPass<func::FuncOp>(
      createLinalgQuantizedConvToConvPass());
  passManager.addNestedPass<func::FuncOp>(mlir::createCanonicalizerPass());

  //----------------------------------------------------------------------------
  // Entry dialect cleanup
  //----------------------------------------------------------------------------
  passManager.addPass(createVerifyCompilerTOSAInputLegality());
}

namespace {
#define GEN_PASS_REGISTRATION
#include "compiler/plugins/input/TOSA/tosa-iree/InputConversion/Passes.h.inc" // IWYU pragma: export
} // namespace

void registerTOSAConversionPasses() {
  // Generated.
  registerPasses();

  // Pipelines.
  registerTOSAConversionPassPipeline();
}

} // namespace mlir::iree_compiler

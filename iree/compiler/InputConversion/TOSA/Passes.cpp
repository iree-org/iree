// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/InputConversion/TOSA/Passes.h"

#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/InputConversion/Common/Passes.h"
#include "mlir/Conversion/TosaToLinalg/TosaToLinalg.h"
#include "mlir/Conversion/TosaToSCF/TosaToSCF.h"
#include "mlir/Conversion/TosaToStandard/TosaToStandard.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {

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
  // Currently we don't handle SCF ops well and have to convert them all to CFG.
  // In the future it would be nice if we could have all of flow be both scf
  // and cfg compatible.
  passManager.addNestedPass<FuncOp>(tosa::createTosaToSCF());
  passManager.addNestedPass<FuncOp>(createTopLevelSCFToCFGPass());

  // Now that control flow has been lowered, promote and extract_element
  // to tensor loads. This will be done again later once everything that can
  // be is lowered to device.
  passManager.addNestedPass<FuncOp>(IREE::Flow::createPromoteTensorLoadsPass());

  // We also don't handle calls well on the old codepath; until we remove the
  // use of the CFG we can continue inlining.
  passManager.addPass(mlir::createInlinerPass());

  passManager.addNestedPass<FuncOp>(tosa::createTosaMakeBroadcastablePass());
  passManager.addNestedPass<FuncOp>(tosa::createTosaToStandard());
  passManager.addNestedPass<FuncOp>(mlir::createCanonicalizerPass());
  passManager.addNestedPass<FuncOp>(IREE::Flow::createPromoteI1ToI8Pass());
  passManager.addNestedPass<FuncOp>(tosa::createTosaToLinalgOnTensors());
  passManager.addNestedPass<FuncOp>(mlir::createCanonicalizerPass());

  //----------------------------------------------------------------------------
  // Entry dialect cleanup
  //----------------------------------------------------------------------------
  passManager.addPass(createVerifyCompilerTOSAInputLegality());
}

namespace {
#define GEN_PASS_REGISTRATION
#include "iree/compiler/InputConversion/TOSA/Passes.h.inc"  // IWYU pragma: export
}  // namespace

void registerTOSAConversionPasses() {
  // Generated.
  registerPasses();

  // Pipelines.
  registerTOSAConversionPassPipeline();
}

}  // namespace iree_compiler
}  // namespace mlir

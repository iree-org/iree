// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree_tf_compiler/TFL/Passes.h"

#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Shape/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"
#include "tensorflow/compiler/mlir/tosa/tf_tfl_passes.h"
#include "tensorflow/compiler/mlir/tosa/transforms/passes.h"

namespace mlir {
namespace iree_integrations {
namespace TFL {

namespace {
#define GEN_PASS_REGISTRATION
#include "iree_tf_compiler/TFL/Passes.h.inc"  // IWYU pragma: export
}  // namespace

// All IREE-specific passes that lower TFL representations before reaching the
// IREE core should go here.
void buildTFLImportPassPipeline(OpPassManager &pm) {
  //----------------------------------------------------------------------------
  // Guarantee the call once functions are preserved.
  //----------------------------------------------------------------------------

  pm.addPass(createRetainCallOnceFuncsPass());

  //----------------------------------------------------------------------------
  // Input IR cleanup
  //----------------------------------------------------------------------------

  pm.addPass(createInlinerPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createSymbolDCEPass());

  //----------------------------------------------------------------------------
  // Convert useful metadata into forms IREE's main compiler understands
  //----------------------------------------------------------------------------

  pm.addPass(createConvertModuleMetadataPass());
  pm.nest<func::FuncOp>().addPass(createConvertFunctionMetadataPass());

  //----------------------------------------------------------------------------
  // Convert all TFL ops to TOSA ops
  //----------------------------------------------------------------------------

  pm.addPass(createLowerGlobalTensorsPass());

  mlir::tosa::TOSATFTFLLegalizationPipelineOptions tosaOptions;
  // Temporary work-around for https://github.com/openxla/iree/issues/8974
  tosaOptions.dequantize_tfl_softmax = true;
  mlir::tosa::createTFTFLtoTOSALegalizationPipeline(pm, tosaOptions);

  pm.nest<func::FuncOp>().addPass(mlir::tosa::createStripQuantTypesPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createReconcileUnrealizedCastsPass());

  //----------------------------------------------------------------------------
  // Lowering shape-related constructs
  //----------------------------------------------------------------------------

  // TODO(#3975): support dynamic shapes in tflite inputs.
  // pm.addPass(iree_compiler::Shape::createConvertHLOToShapePass());
  // pm.addPass(createCanonicalizerPass());
  // pm.addPass(iree_compiler::Shape::createConvertShapeToShapexPass());
  // pm.addPass(createCanonicalizerPass());

  //----------------------------------------------------------------------------
  // Remove the rest of the TFL goo and verify that all ops converted
  //----------------------------------------------------------------------------

  pm.nest<func::FuncOp>().addPass(createStripFunctionMetadataPass());
  pm.addPass(createStripModuleMetadataPass());
  pm.addPass(createVerifyFullyConvertedPass());
}

void registerTFLImportPassPipeline() {
  mlir::PassPipelineRegistration<> pipeline(
      "iree-tflite-import-pipeline",
      "Run IREE-specific passes for importing TFLite code into IREE",
      [](OpPassManager &passManager) {
        buildTFLImportPassPipeline(passManager);
      });
}

void registerAllPasses() {
  registerTFLImportPassPipeline();

  // Generated.
  registerPasses();

  createVerifyFullyConvertedPass();
}

}  // namespace TFL
}  // namespace iree_integrations
}  // namespace mlir

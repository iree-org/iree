// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree_tf_compiler/TF/Passes.h"

#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/Shape/Transforms/Passes.h"
#include "iree_tf_compiler/MHLO/Passes.h"
#include "iree_tf_compiler/dialect/tf_strings/ir/dialect.h"
#include "iree_tf_compiler/dialect/tf_tensorlist/ir/tf_tensorlist_dialect.h"
#include "mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "mlir/Dialect/Shape/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_saved_model_passes.h"

namespace mlir {
namespace iree_integrations {
namespace TF {

void registerAllDialects(mlir::DialectRegistry &registry) {
  registry.insert<tf_strings::TFStringsDialect>();
  registry.insert<tf_tensorlist::TFTensorListDialect>();
}

// All IREE-specific passes that lower TF representations before reaching the
// IREE core should go here.
void buildTFImportPassPipeline(OpPassManager &pm) {
  //----------------------------------------------------------------------------
  // Clean up tf_executor and extraneous unused functions.
  //----------------------------------------------------------------------------
  pm.addPass(createSymbolDCEPass());
  pm.addPass(tf_executor::CreateTFExecutorGraphPruningPass());
  pm.addPass(::mlir::TF::CreateGuaranteeAllFuncsOneUsePass());
  ::mlir::TF::CreateTFStandardPipeline(pm,
                                       ::mlir::TF::StandardPipelineOptions());
  pm.addPass(::mlir::TF::CreateDeviceIndexSelectorPass());

  //----------------------------------------------------------------------------
  // Try to get the IR in good condition.
  //----------------------------------------------------------------------------
  pm.addPass(createStripAssertsPass());
  pm.addPass(createInlinerPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(TFDevice::CreateDecomposeResourceOpsPass());
  pm.addPass(createPropagateResourceCastsPass());
  pm.addPass(::mlir::TF::CreateTFShapeInferencePass());

  //----------------------------------------------------------------------------
  // Lower to CFG.
  // After this point, most TF optimizations won't work properly besides
  // simple canonicalizations.
  //----------------------------------------------------------------------------
  pm.addPass(::mlir::TF::CreateTFFunctionalControlFlowToCFG());
  // Inline, as tf-functional-control-flow-to-cfg leaves in calls.
  pm.addPass(createInlinerPass());

  //----------------------------------------------------------------------------
  // Some further cleanups now that control flow is in better shape.
  //----------------------------------------------------------------------------
  pm.addPass(createSymbolDCEPass());
  pm.addPass(createCanonicalizerPass());

  //----------------------------------------------------------------------------
  // Lowering module specific TF behavior to intermediate dialects.
  //----------------------------------------------------------------------------
  pm.addPass(tf_strings::createConvertTFToTFStringsPass());
  pm.addPass(tf_tensorlist::createConvertTFToTFTensorListPass());

  //----------------------------------------------------------------------------
  // Legalize to XLA
  //----------------------------------------------------------------------------
  pm.addPass(createConvertToMHLOPass());
  pm.addPass(createCanonicalizerPass());

  //----------------------------------------------------------------------------
  // Now that the IR is starting to look nice, optimize global tensors.
  //----------------------------------------------------------------------------
  pm.addPass(tf_saved_model::CreateOptimizeGlobalTensorsPass());

  //----------------------------------------------------------------------------
  // Lowering shape-related constructs.
  //----------------------------------------------------------------------------
  // pm.addPass(iree_compiler::Shape::createConvertHLOToShapePass());
  // TODO(GH-2277): Lower HLO shape constraints instead of eliding them here.
  pm.addPass(createRemoveShapeConstraintsPass());
  pm.addPass(createCanonicalizerPass());
  // pm.addPass(iree_compiler::Shape::createConvertShapeToShapexPass());
  // pm.addPass(createCanonicalizerPass());

  //----------------------------------------------------------------------------
  // Lowering intermediate dialects to module specific dialects.
  //----------------------------------------------------------------------------
  pm.addPass(tf_strings::createConvertTFStringsToStringsPass());
  pm.addPass(tf_tensorlist::createConvertTFTensorListToTensorListPass());

  //----------------------------------------------------------------------------
  // Lowering tf_saved_model dialect to IREE dialects
  //----------------------------------------------------------------------------
  // First, eliminate tf_saved_model.global_tensor's and corresponding
  // tf_saved_model.bound_input's.
  pm.addPass(createLowerGlobalTensorsPass());

  // Lower exported functions.
  //
  // This pass must run second because:
  // - It assumes that tf_saved_model.bound_inputs have been eliminated
  // - It removes tf_saved_model.semantics from the module, which we can only
  //   do at the very end.
  pm.addPass(createSavedModelToIREEABIPass());
  // Inline the wrapper functions.
  pm.addPass(createInlinerPass());

  //----------------------------------------------------------------------------
  // Ensure that all Tensorflow has been legalized away
  //----------------------------------------------------------------------------
  pm.addPass(createStripModuleMetadataPass());
  pm.nest<ModuleOp>().addPass(createStripFunctionMetadataPass());
  pm.addPass(createVerifyFullyConvertedPass());

  MHLO::buildMHLOImportPassPipeline(pm);
}

void registerTFImportPassPipeline() {
  mlir::PassPipelineRegistration<> pipeline(
      "iree-import-tf-pipeline",
      "Run IREE-specific passes for importing TF code into IREE",
      [](OpPassManager &passManager) {
        buildTFImportPassPipeline(passManager);
      });
}

}  // namespace TF
}  // namespace iree_integrations
}  // namespace mlir

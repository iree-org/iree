// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree_tf_compiler/TFL/Passes.h"

#include "iree/compiler/Dialect/Shape/Conversion/Passes.h"
#include "iree/compiler/Dialect/Shape/Transforms/Passes.h"
#include "mlir/Dialect/Shape/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"
#include "tensorflow/compiler/mlir/tosa/tfl_passes.h"

namespace mlir {
namespace iree_integrations {
namespace TFL {

// All IREE-specific passes that lower TFL representations before reaching the
// IREE core should go here.
void buildTFLImportPassPipeline(OpPassManager &pm) {
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
  pm.nest<ModuleOp>().addPass(createConvertFunctionMetadataPass());

  //----------------------------------------------------------------------------
  // Convert all TFL ops to TOSA ops
  //----------------------------------------------------------------------------

  mlir::tosa::TOSATFLLegalizationPipelineOptions tosaOptions;
  mlir::tosa::createTFLtoTOSALegalizationPipeline(pm, tosaOptions);
  pm.addPass(createCanonicalizerPass());

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

  pm.addPass(createStripModuleMetadataPass());
  pm.nest<ModuleOp>().addPass(createStripFunctionMetadataPass());
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

}  // namespace TFL
}  // namespace iree_integrations
}  // namespace mlir
